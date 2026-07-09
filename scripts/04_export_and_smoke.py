#!/usr/bin/env python3
"""Load LoRA adapter, run a few smoke prompts, optionally merge weights.

Works around Qwen3.6 multimodal processor pitfalls by building text-only
input_ids and calling generate with only those tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_text_inputs(tokenizer, messages: list[dict], device):
    """Tokenize chat for text-only generation (no vision path)."""
    # Prefer chat-template tokenize API when available
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        elif isinstance(encoded, dict):
            encoded = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in encoded.items()
            }
        # Keep only text tensors so VL processors never see a fake image
        out = {}
        for k in ("input_ids", "attention_mask"):
            if isinstance(encoded, dict) and k in encoded:
                out[k] = encoded[k]
            elif hasattr(encoded, k):
                out[k] = getattr(encoded, k)
        if "input_ids" in out:
            return out
    except Exception as e:
        print(f"apply_chat_template(tokenize=True) failed ({e}); falling back")

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Explicit text= avoids some processor __call__ paths treating the
    # prompt string as an image URL / base64 blob.
    try:
        encoded = tokenizer(text=prompt, return_tensors="pt")
    except TypeError:
        encoded = tokenizer(prompt, return_tensors="pt")
    return {
        k: v.to(device)
        for k, v in encoded.items()
        if k in ("input_ids", "attention_mask") and hasattr(v, "to")
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mix.yaml"))
    ap.add_argument(
        "--adapter",
        default=str(ROOT / "outputs" / "qwen36-agent-lora" / "lora_adapter"),
    )
    ap.add_argument("--merge", action="store_true", help="Also save merged 16-bit model")
    ap.add_argument(
        "--prompt",
        default="List the files in the current directory using a tool call.",
    )
    args = ap.parse_args()

    mix = yaml.safe_load(Path(args.config).read_text())
    model_name = mix["student"].get("unsloth_model_id") or mix["student"]["model_id"]
    adapter = Path(args.adapter)
    if not adapter.exists():
        raise SystemExit(f"Adapter not found: {adapter}")

    import torch
    from unsloth import FastModel
    from peft import PeftModel

    print(f"Base:    {model_name}")
    print(f"Adapter: {adapter}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        # Unsloth defaults load_in_4bit=True — must not mix with 16bit
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    FastModel.for_inference(model)
    model.eval()

    device = next(model.parameters()).device
    system = mix.get("system_prompt", "You are a coding agent.")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": args.prompt},
    ]

    inputs = build_text_inputs(tokenizer, messages, device)
    if "input_ids" not in inputs:
        raise SystemExit(f"Failed to build input_ids; got keys={list(inputs)}")

    print(f"input_ids shape: {tuple(inputs['input_ids'].shape)}")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    print("=== SMOKE GENERATION ===")
    print(text)
    print("========================")

    report = {
        "adapter": str(adapter),
        "base": model_name,
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
