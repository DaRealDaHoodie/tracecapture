#!/usr/bin/env python3
"""
Fine-tune Qwen3.6-35B-A3B with Unsloth LoRA on mixed agent traces.

Memory notes for 1× RTX PRO 6000 (96GB):
- Use max_seq_length 4096 (8192 OOMs on MoE aux loss)
- lora_dropout must be 0 for MoE ParamWrapper
- Disable router aux loss (load_balancing_loss is VRAM-hungry at long seq)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reduce fragmentation on long MoE runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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


def disable_moe_aux_loss(model) -> None:
    """Turn off router load-balancing loss (big VRAM spike on long sequences)."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    for attr, val in (
        ("router_aux_loss_coef", 0.0),
        ("aux_loss_alpha", 0.0),
        ("output_router_logits", False),
    ):
        if hasattr(cfg, attr):
            setattr(cfg, attr, val)
            print(f"  set config.{attr} = {val}")
    # Some wrappers nest text_config
    tc = getattr(cfg, "text_config", None)
    if tc is not None:
        for attr, val in (
            ("router_aux_loss_coef", 0.0),
            ("aux_loss_alpha", 0.0),
            ("output_router_logits", False),
        ):
            if hasattr(tc, attr):
                setattr(tc, attr, val)
                print(f"  set text_config.{attr} = {val}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mix.yaml"))
    ap.add_argument("--data", default=None, help="Override train JSONL path")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "qwen36-agent-lora"))
    ap.add_argument("--max-steps", type=int, default=None, help="Optional cap for smoke tests")
    ap.add_argument("--smoke", action="store_true", help="Tiny run to verify training works")
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override max sequence length (default from config, recommend 4096 on 96GB)",
    )
    args = ap.parse_args()

    mix = load_mix(Path(args.config))
    tcfg = mix["train"]
    data_path = Path(args.data or (ROOT / mix["output"]["mixed_path"]))
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run scripts/02_normalize_and_mix.py first.")

    # Import Unsloth first (patches transformers)
    import torch
    from unsloth import FastModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig
    from transformers import DataCollatorForSeq2Seq

    model_name = mix["student"].get("unsloth_model_id") or mix["student"]["model_id"]
    if args.smoke:
        max_seq = 2048
    elif args.max_seq_length is not None:
        max_seq = args.max_seq_length
    else:
        # 8192 OOMs on MoE aux/activations on 96GB; 4096 is the safe default
        max_seq = int(tcfg.get("max_seq_length", 4096))

    use_4bit = bool(tcfg.get("load_in_4bit", False))
    print(f"Loading {model_name}  max_seq={max_seq}  4bit={use_4bit}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        load_in_4bit=use_4bit,
        load_in_16bit=not use_4bit,
        full_finetuning=False,
    )

    disable_moe_aux_loss(model)
    # base model may sit under .model after PEFT; also try before
    if hasattr(model, "model"):
        disable_moe_aux_loss(model.model)

    # Qwen chat template
    try:
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    except Exception:
        pass

    # Attention-only LoRA is much lighter; experts still route without LoRA.
    # Set train.lora_on_experts: true in mix.yaml to also adapt expert MLPs.
    lora_on_experts = bool(tcfg.get("lora_on_experts", False))
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    if lora_on_experts:
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    print(f"LoRA targets: {target_modules}")

    model = FastModel.get_peft_model(
        model,
        r=int(tcfg.get("lora_r", 16)),
        target_modules=target_modules,
        lora_alpha=int(tcfg.get("lora_alpha", 32)),
        lora_dropout=float(tcfg.get("lora_dropout", 0.0)),  # must be 0 for MoE
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(tcfg.get("seed", 42)),
    )

    disable_moe_aux_loss(model)
    if hasattr(model, "model"):
        disable_moe_aux_loss(model.model)
    if hasattr(model, "base_model"):
        disable_moe_aux_loss(model.base_model)
        if hasattr(model.base_model, "model"):
            disable_moe_aux_loss(model.base_model.model)

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

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = 1 if args.smoke else float(tcfg.get("num_train_epochs", 1))
    max_steps = args.max_steps if args.max_steps is not None else (-1 if not args.smoke else 20)

    # Free any fragmentation before trainer alloc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
            gradient_checkpointing=True,
            max_grad_norm=1.0,
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
