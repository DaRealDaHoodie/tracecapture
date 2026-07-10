#!/usr/bin/env python3
"""Merge PEFT LoRA into base weights for plain vLLM serve (no --enable-lora).

Use when dynamic LoRA load fails on MoE / Unsloth weight names.

  python scripts/07_merge_lora_for_vllm.py
  MERGE_PATH=outputs/qwen36-agent-lora/merged_16bit bash scripts/06_serve_vllm.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mix.yaml"))
    ap.add_argument(
        "--adapter",
        default=str(ROOT / "outputs" / "qwen36-agent-lora" / "lora_adapter"),
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "qwen36-agent-lora" / "merged_16bit"),
    )
    ap.add_argument("--base", default=None, help="Override base model id")
    args = ap.parse_args()

    mix = yaml.safe_load(Path(args.config).read_text())
    base = args.base or mix["student"].get("model_id") or "Qwen/Qwen3.6-35B-A3B"
    adapter = Path(args.adapter)
    out = Path(args.out)
    if not adapter.is_dir():
        raise SystemExit(f"Adapter not found: {adapter}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Base:    {base}")
    print(f"Adapter: {adapter}")
    print(f"Out:     {out}")

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    print("Merging…")
    model = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(f"Done. Serve with:\n  MERGE_PATH={out} bash scripts/06_serve_vllm.sh")


if __name__ == "__main__":
    main()
