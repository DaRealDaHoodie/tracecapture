#!/usr/bin/env python3
"""Download enabled teacher datasets from Hugging Face into data/raw/.

Handles datasets that do NOT use a default ``train`` split (e.g. NVIDIA
Nemotron sets with named splits like bash_only_tool / general).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Prefer these split names first when auto-discovering.
PREFERRED_SPLITS = (
    "train",
    "train_sft",
    "sft",
    "default",
    "general",
    "bash_only_tool",
    "bash_only_tool_skills",
    "agent_skills",
    "question_tool",
    "agent_skills_question_tool",
    "tool_calling",
    "thinking",
    "non_thinking",
    "pass",
    "success",
    "resolved",
)


def load_mix(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def row_to_dict(row: Any) -> dict:
    if hasattr(row, "keys"):
        return {k: row[k] for k in row.keys()}
    if isinstance(row, dict):
        return row
    return {"value": row}


def list_configs(hf_id: str) -> list[str | None]:
    try:
        names = get_dataset_config_names(hf_id)
        if names:
            return list(names)
    except Exception as e:
        print(f"  config discovery: {e}")
    return [None]


def list_splits(hf_id: str, config: str | None) -> list[str]:
    try:
        if config is None:
            return list(get_dataset_split_names(hf_id))
        return list(get_dataset_split_names(hf_id, config))
    except Exception as e:
        print(f"  split discovery ({config}): {e}")
        return []


def order_splits(available: list[str], preferred: list[str] | None) -> list[str]:
    if preferred:
        # honor explicit list, keep only those that exist if available known
        if available:
            return [s for s in preferred if s in available] or preferred
        return preferred
    if not available:
        return ["train"]
    # preferred order, then any leftovers
    ordered: list[str] = []
    for s in PREFERRED_SPLITS:
        if s in available and s not in ordered:
            ordered.append(s)
    for s in available:
        if s not in ordered and "test" not in s.lower() and "dev" not in s.lower():
            ordered.append(s)
    return ordered or available


def write_stream(ds, out_f, max_rows: int | None, already: int, desc: str) -> int:
    n = already
    for row in tqdm(ds, desc=desc):
        if max_rows is not None and n >= max_rows:
            break
        out_f.write(json.dumps(row_to_dict(row), ensure_ascii=False, default=str) + "\n")
        n += 1
    return n


def try_load_streaming(hf_id: str, config: str | None, split: str):
    kwargs: dict[str, Any] = {
        "path": hf_id,
        "split": split,
        "streaming": True,
        "trust_remote_code": False,
    }
    if config is not None:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def try_load_map(hf_id: str, config: str | None, split: str):
    kwargs: dict[str, Any] = {
        "path": hf_id,
        "split": split,
        "streaming": False,
        "trust_remote_code": False,
    }
    if config is not None:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def download_data_files(hf_id: str, data_files: list[str], out_path: Path, max_rows: int | None) -> Path:
    """Download explicit repo files (e.g. a single merged JSONL) — avoids multi-file 429s."""
    from huggingface_hub import hf_hub_download

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download files] {hf_id} {data_files} -> {out_path}")
    total = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for rel in data_files:
            local = hf_hub_download(
                repo_id=hf_id,
                repo_type="dataset",
                filename=rel,
            )
            print(f"  got {rel} -> {local}")
            with open(local, encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    out_f.write(line + "\n")
                    total += 1
                    if max_rows is not None and total >= max_rows:
                        print(f"  wrote {total} rows (capped)")
                        return out_path
    print(f"  wrote {total} rows -> {out_path}")
    return out_path


def download_repo_jsonl_glob(
    hf_id: str,
    out_path: Path,
    max_rows: int | None,
    pattern: str = "rollout-*.jsonl",
) -> Path:
    """Merge many agent-trace JSONL files from a HF dataset repo (e.g. GPT-5.5 Codex)."""
    import fnmatch

    from huggingface_hub import hf_hub_download, list_repo_files

    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(hf_id, repo_type="dataset")
    matched = [f for f in files if fnmatch.fnmatch(f.split("/")[-1], pattern) or fnmatch.fnmatch(f, pattern)]
    matched = [f for f in matched if f.endswith(".jsonl")]
    print(f"[download glob] {hf_id} pattern={pattern!r} matched={len(matched)} -> {out_path}")
    if not matched:
        raise RuntimeError(f"No files matched {pattern!r} in {hf_id}")

    total = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for rel in matched:
            if max_rows is not None and total >= max_rows:
                break
            try:
                local = hf_hub_download(repo_id=hf_id, repo_type="dataset", filename=rel)
            except Exception as e:
                print(f"  skip {rel}: {e}")
                continue
            with open(local, encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    # tag source file for converters
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        obj.setdefault("_source_file", rel)
                        out_f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
                    else:
                        out_f.write(line + "\n")
                    total += 1
                    if max_rows is not None and total >= max_rows:
                        break
    print(f"  wrote {total} event rows from {len(matched)} files -> {out_path}")
    if total == 0:
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(f"No rows from glob download of {hf_id}")
    return out_path


def download_hf(
    hf_id: str,
    out_dir: Path,
    max_rows: int | None,
    config: str | None,
    splits: list[str] | None,
    data_files: list[str] | None = None,
    file_glob: str | None = None,
) -> Path:
    out_path = out_dir / (hf_id.replace("/", "__") + ".jsonl")
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"[skip] already have {out_path}")
        return out_path

    if data_files:
        return download_data_files(hf_id, data_files, out_path, max_rows)
    if file_glob:
        return download_repo_jsonl_glob(hf_id, out_path, max_rows, pattern=file_glob)

    print(f"[download] {hf_id} -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Configs to try
    if config:
        configs: list[str | None] = [config]
    else:
        configs = list_configs(hf_id)
        # Prefer a config literally named "default" if present, else all
        if len(configs) > 1 and "default" in configs:
            # Still only use default first; if empty we can expand
            configs = ["default"] + [c for c in configs if c != "default"]

    total = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for cfg in configs:
            available = list_splits(hf_id, cfg)
            split_list = order_splits(available, splits)
            print(f"  config={cfg!r} splits={split_list} (available={available or 'unknown'})")

            for split in split_list:
                if max_rows is not None and total >= max_rows:
                    break
                remaining = None if max_rows is None else max_rows - total
                desc = f"{hf_id}:{cfg or '-'}:{split}"
                try:
                    ds = try_load_streaming(hf_id, cfg, split)
                    before = total
                    total = write_stream(ds, out_f, max_rows, total, desc)
                    print(f"    +{total - before} rows from streaming {split}")
                except Exception as e_stream:
                    print(f"    streaming {split} failed: {e_stream}")
                    try:
                        ds = try_load_map(hf_id, cfg, split)
                        before = total
                        # map-style Dataset
                        limit = remaining
                        count = 0
                        for row in tqdm(ds, desc=desc + " (map)"):
                            if limit is not None and count >= limit:
                                break
                            out_f.write(
                                json.dumps(row_to_dict(row), ensure_ascii=False, default=str)
                                + "\n"
                            )
                            count += 1
                            total += 1
                        print(f"    +{count} rows from map {split}")
                    except Exception as e_map:
                        print(f"    map {split} failed: {e_map}")
                        continue

            # If we already got data from first config, stop (don't duplicate all configs)
            if total > 0:
                break

    if total == 0:
        # clean empty file so re-run retries
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(
            f"No rows downloaded for {hf_id}. "
            "Check HF_TOKEN, dataset name, or set splits: in mix.yaml."
        )

    print(f"  wrote {total} rows -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mix.yaml"))
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Download at most 500 rows per dataset for a quick test",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only download these teacher ids (e.g. nemotron_opencode nemotron_agentic)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if raw jsonl already exists",
    )
    args = ap.parse_args()
    mix = load_mix(Path(args.config))
    raw_dir = ROOT / mix["output"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    only = set(args.only) if args.only else None
    errors: list[str] = []

    for t in mix["teachers"]:
        tid = t["id"]
        if only is not None and tid not in only:
            continue
        if not t.get("enabled", True):
            print(f"[skip disabled] {tid}")
            continue
        if not t.get("hf_id"):
            print(f"[skip no hf_id] {tid} (use local_glob later)")
            continue

        limit = t.get("max_rows")
        if args.smoke:
            limit = min(int(limit or 500), 500)

        out_path = raw_dir / (t["hf_id"].replace("/", "__") + ".jsonl")
        if args.force and out_path.exists():
            out_path.unlink()
            print(f"[force] removed {out_path}")

        splits = t.get("splits")
        if isinstance(splits, str):
            splits = [splits]
        data_files = t.get("data_files")
        if isinstance(data_files, str):
            data_files = [data_files]
        file_glob = t.get("file_glob")

        try:
            download_hf(
                t["hf_id"],
                raw_dir,
                limit,
                t.get("config"),
                splits,
                data_files=data_files,
                file_glob=file_glob,
            )
        except Exception as e:
            msg = f"{tid} ({t['hf_id']}): {e}"
            print(f"[ERROR] {msg}")
            print("  Continue with other datasets. You can re-run this script.")
            errors.append(msg)

    print("\nDone.")
    if errors:
        print(f"{len(errors)} dataset(s) failed:")
        for e in errors:
            print(f"  - {e}")
        print("Fix/retry those, or set enabled: false in configs/mix.yaml.")
    print("Next: python scripts/02_normalize_and_mix.py")


if __name__ == "__main__":
    main()
