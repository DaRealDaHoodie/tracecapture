#!/usr/bin/env python3
"""
Fine-tune Qwen3.6-35B-A3B with Unsloth LoRA on mixed agent traces.

Run on the GPU pod AFTER:
  bash scripts/runpod_setup.sh
  python scripts/01_download_datasets.py
  python scripts/02_normalize_and_mix.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_mix(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_messages_dataset(jsonl_path: Path):
    from datasets import Dataset

    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not msgs:
                continue
            rows.append({"messages": msgs})
    if not rows:
        raise SystemExit(f"No training rows in {jsonl_path}")
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mix.yaml"))
    ap.add_argument("--data", default=None, help="Override train JSONL path")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "qwen36-agent-lora"))
    ap.add_argument("--max-steps", type=int, default=None, help="Optional cap for smoke tests")
    ap.add_argument("--smoke", action="store_true", help="Tiny run to verify training works")
    args = ap.parse_args()

    mix = load_mix(Path(args.config))
    tcfg = mix["train"]
    data_path = Path(args.data or (ROOT / mix["output"]["mixed_path"]))
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run scripts/02_normalize_and_mix.py first.")

    # Import Unsloth first (patches transformers)
    from unsloth import FastModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig
    from transformers import DataCollatorForSeq2Seq

    model_name = mix["student"].get("unsloth_model_id") or mix["student"]["model_id"]
    max_seq = 2048 if args.smoke else int(tcfg.get("max_seq_length", 8192))
    print(f"Loading {model_name}  max_seq={max_seq}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        load_in_4bit=False,
        load_in_16bit=True,  # bf16 LoRA — recommended for MoE on 96GB
        full_finetuning=False,
    )

    # Qwen chat template
    try:
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    except Exception:
        pass  # keep model default chat template

    model = FastModel.get_peft_model(
        model,
        r=int(tcfg.get("lora_r", 32)),
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=int(tcfg.get("lora_alpha", 64)),
        lora_dropout=float(tcfg.get("lora_dropout", 0.05)),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(tcfg.get("seed", 42)),
        # Do not finetune router on MoE (Unsloth default is safe)
    )

    dataset = load_messages_dataset(data_path)
    if args.smoke:
        dataset = dataset.select(range(min(64, len(dataset))))

    def formatting_prompts_func(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = 1 if args.smoke else float(tcfg.get("num_train_epochs", 1))
    max_steps = args.max_steps if args.max_steps is not None else (-1 if not args.smoke else 20)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 16)),
            warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
            num_train_epochs=epochs,
            max_steps=max_steps,
            learning_rate=float(tcfg.get("learning_rate", 8e-5)),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=int(tcfg.get("logging_steps", 10)),
            save_steps=int(tcfg.get("save_steps", 200)),
            optim="adamw_8bit",
            weight_decay=float(tcfg.get("weight_decay", 0.01)),
            lr_scheduler_type="cosine",
            seed=int(tcfg.get("seed", 42)),
            output_dir=str(out_dir),
            report_to="none",
            # train only on assistant completions when supported
        ),
    )

    print("Starting training…")
    trainer.train()
    print("Saving adapter…")
    model.save_pretrained(str(out_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(out_dir / "lora_adapter"))
    print(f"Done. Adapter at {out_dir / 'lora_adapter'}")
    print("Next: python scripts/04_export_and_smoke.py")


if __name__ == "__main__":
    main()
