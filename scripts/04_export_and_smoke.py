#!/usr/bin/env python3
"""Load LoRA adapter, run a few smoke prompts, optionally merge weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mix.yaml"))
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "qwen36-agent-lora" / "lora_adapter"))
    ap.add_argument("--merge", action="store_true", help="Also save merged 16-bit model (large)")
    ap.add_argument("--prompt", default="List the files in the current directory using a tool call.")
    args = ap.parse_args()

    mix = yaml.safe_load(Path(args.config).read_text())
    model_name = mix["student"].get("unsloth_model_id") or mix["student"]["model_id"]
    adapter = Path(args.adapter)
    if not adapter.exists():
        raise SystemExit(f"Adapter not found: {adapter}")

    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        load_in_16bit=True,
        full_finetuning=False,
    )
    # load adapter
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter))
    FastModel.for_inference(model)

    system = mix.get("system_prompt", "You are a coding agent.")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=512, temperature=0.6, top_p=0.95)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    print("=== SMOKE GENERATION ===")
    print(text)
    print("========================")

    report = {
        "adapter": str(adapter),
        "prompt": args.prompt,
        "completion": text,
    }
    report_path = adapter.parent / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {report_path}")

    if args.merge:
        print("Merging LoRA into base (needs disk + VRAM)…")
        merged = model.merge_and_unload()
        merge_dir = adapter.parent / "merged_16bit"
        merged.save_pretrained(str(merge_dir))
        tokenizer.save_pretrained(str(merge_dir))
        print(f"Merged model at {merge_dir}")


if __name__ == "__main__":
    main()
