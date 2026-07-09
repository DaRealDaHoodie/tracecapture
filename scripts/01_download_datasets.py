#!/usr/bin/env python3
"""Download enabled teacher datasets from Hugging Face into data/raw/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_mix(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def save_jsonl(rows, path: Path, limit: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                break
            # datasets rows may be Arrow-ish dicts
            if hasattr(row, "items"):
                obj = {k: row[k] for k in row.keys()}
            else:
                obj = row
            # make JSON-serializable
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def download_hf(hf_id: str, out_dir: Path, max_rows: int | None, config: str | None) -> Path:
    out_path = out_dir / (hf_id.replace("/", "__") + ".jsonl")
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"[skip] already have {out_path}")
        return out_path

    print(f"[download] {hf_id} -> {out_path}")
    kwargs = {"path": hf_id, "split": "train", "streaming": True}
    if config:
        kwargs["name"] = config
    try:
        ds = load_dataset(**kwargs)
    except Exception as e1:
        print(f"  streaming train failed ({e1}); trying snapshot + local load")
        local = snapshot_download(repo_id=hf_id, repo_type="dataset")
        try:
            ds = load_dataset(local, split="train", streaming=True)
        except Exception:
            # last resort: any split
            ds = load_dataset(local, split="train")
            n = save_jsonl(ds, out_path, max_rows)
            print(f"  wrote {n} rows")
            return out_path

    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=hf_id):
            if max_rows is not None and n >= max_rows:
                break
            obj = {k: row[k] for k in row.keys()}
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            n += 1
    print(f"  wrote {n} rows")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mix.yaml"))
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Download at most 500 rows per dataset for a quick test",
    )
    args = ap.parse_args()
    mix = load_mix(Path(args.config))
    raw_dir = ROOT / mix["output"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    for t in mix["teachers"]:
        if not t.get("enabled", True):
            print(f"[skip disabled] {t['id']}")
            continue
        if not t.get("hf_id"):
            print(f"[skip no hf_id] {t['id']} (use local_glob later)")
            continue
        limit = t.get("max_rows")
        if args.smoke:
            limit = min(limit or 500, 500)
        try:
            download_hf(t["hf_id"], raw_dir, limit, t.get("config"))
        except Exception as e:
            print(f"[ERROR] {t['id']} ({t['hf_id']}): {e}")
            print("  Continue with other datasets. You can re-run this script.")

    print("\nDone. Next: python scripts/02_normalize_and_mix.py")


if __name__ == "__main__":
    main()
