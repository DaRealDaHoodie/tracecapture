#!/usr/bin/env python3
"""Normalize raw dumps and mix teachers into one Qwen chat JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.converters import convert_rows  # noqa: E402


def load_mix(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_local_glob(pattern: str):
    for p in sorted(ROOT.glob(pattern)):
        if p.is_file() and p.suffix in {".jsonl", ".json"}:
            yield from iter_jsonl(p)


def example_hash(ex: dict) -> str:
    # hash on assistant + truncated user to dedup
    msgs = ex.get("messages") or []
    user = ""
    asst = ""
    for m in msgs:
        if m.get("role") == "user":
            user = m.get("content") or ""
        if m.get("role") == "assistant":
            asst = m.get("content") or ""
    key = (user[:2000] + "||" + asst[:2000]).encode("utf-8", errors="ignore")
    return hashlib.sha256(key).hexdigest()


def truncate_example(ex: dict, max_chars: int) -> dict | None:
    """Rough length guard before tokenization."""
    total = sum(len(m.get("content") or "") for m in ex.get("messages") or [])
    if total <= max_chars:
        return ex
    # try trimming oldest history turns (keep system + last user + assistant)
    msgs = ex["messages"]
    system = [m for m in msgs if m["role"] == "system"][:1]
    asst = [m for m in msgs if m["role"] == "assistant"][-1:]
    users = [m for m in msgs if m["role"] == "user"]
    if not users or not asst:
        return None
    user = users[-1]
    # shrink user content from the left (keep the end — recent context)
    budget = max_chars - sum(len(m.get("content") or "") for m in system + asst)
    content = user["content"]
    if len(content) > budget:
        content = "…[truncated]…\n" + content[-(budget - 20) :]
    new = {
        **ex,
        "messages": system + [{"role": "user", "content": content}] + asst,
    }
    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mix.yaml"))
    ap.add_argument("--smoke", action="store_true", help="Small mix for pipeline test")
    args = ap.parse_args()
    mix = load_mix(Path(args.config))
    raw_dir = ROOT / mix["output"]["raw_dir"]
    cleaned_dir = ROOT / mix["output"]["cleaned_dir"]
    mixed_path = ROOT / mix["output"]["mixed_path"]
    holdout_path = ROOT / mix["output"]["holdout_path"]
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    mixed_path.parent.mkdir(parents=True, exist_ok=True)

    target = 2000 if args.smoke else int(mix["output"].get("target_train_rows", 40000))
    max_chars = int(mix["output"].get("max_seq_chars", 48000))
    holdout_frac = float(mix["output"].get("holdout_fraction", 0.05))
    system_prompt = mix.get("system_prompt")

    # Collect per-teacher examples
    by_teacher: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()

    enabled = [t for t in mix["teachers"] if t.get("enabled", True)]
    for t in enabled:
        tid = t["id"]
        print(f"\n=== convert {tid} ===")
        rows_iter = None
        if t.get("hf_id"):
            raw_path = raw_dir / (t["hf_id"].replace("/", "__") + ".jsonl")
            if not raw_path.exists():
                print(f"  missing {raw_path} — run 01_download_datasets.py first")
                continue
            rows_iter = iter_jsonl(raw_path)
        elif t.get("local_glob"):
            rows_iter = iter_local_glob(t["local_glob"])
        else:
            print("  no source")
            continue

        count = 0
        for ex in convert_rows(tid, rows_iter):
            d = ex.to_dict()
            if system_prompt:
                if d["messages"] and d["messages"][0]["role"] == "system":
                    d["messages"][0]["content"] = system_prompt.strip()
            d = truncate_example(d, max_chars)
            if not d:
                continue
            h = example_hash(d)
            if h in seen:
                continue
            seen.add(h)
            by_teacher[tid].append(d)
            count += 1
            if args.smoke and count >= 400:
                break
            if t.get("max_rows") and count >= int(t["max_rows"]) * 3:
                # converter expands; soft cap
                break
        # write cleaned per teacher
        out_c = cleaned_dir / f"{tid}.jsonl"
        with out_c.open("w", encoding="utf-8") as f:
            for d in by_teacher[tid]:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  kept {len(by_teacher[tid])} unique examples -> {out_c}")

    # Mix by share
    shares = {t["id"]: float(t.get("share", 0)) for t in enabled if t["id"] in by_teacher}
    total_share = sum(shares.values()) or 1.0
    rng = random.Random(mix.get("train", {}).get("seed", 42))

    mixed: list[dict] = []
    for tid, share in shares.items():
        pool = by_teacher[tid]
        if not pool:
            continue
        n = max(1, int(target * (share / total_share)))
        if len(pool) <= n:
            chosen = pool
        else:
            chosen = rng.sample(pool, n)
        print(f"mix {tid}: take {len(chosen)} / {len(pool)} (target {n})")
        mixed.extend(chosen)

    rng.shuffle(mixed)
    # holdout by source_id hash when possible
    holdout, train = [], []
    for d in mixed:
        sid = d.get("source_id") or example_hash(d)
        bucket = int(hashlib.md5(sid.encode()).hexdigest()[:8], 16) % 1000
        if bucket < holdout_frac * 1000:
            holdout.append(d)
        else:
            train.append(d)

    with mixed_path.open("w", encoding="utf-8") as f:
        for d in tqdm(train, desc="write train"):
            # SFT trainer only needs messages; keep meta for debugging
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with holdout_path.open("w", encoding="utf-8") as f:
        for d in holdout:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nTrain:   {len(train)} -> {mixed_path}")
    print(f"Holdout: {len(holdout)} -> {holdout_path}")
    print("Next: bash scripts/runpod_setup.sh && python scripts/03_train_unsloth.py")


if __name__ == "__main__":
    main()
