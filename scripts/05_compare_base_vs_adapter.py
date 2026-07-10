#!/usr/bin/env python3
"""
Compare base Qwen3.6 vs base+LoRA on the same prompts.

VRAM-safe design:
  - Default mode runs base and adapter in **separate Python processes**
    so CUDA memory is fully released between loads (in-process free is flaky).
  - Optional --load-in-4bit for tighter VRAM.

Usage (on the GPU pod):
  python scripts/05_compare_base_vs_adapter.py
  python scripts/05_compare_base_vs_adapter.py --load-in-4bit --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PROBES = [
    {
        "id": "list_dir",
        "user": "List the files in the current directory using a tool call.",
    },
    {
        "id": "read_file",
        "user": "Read the first 20 lines of README.md with a tool.",
    },
    {
        "id": "fix_bug",
        "user": (
            "There's a bug in app.py: the /health endpoint returns 500. "
            "Investigate with tools and propose a minimal fix."
        ),
    },
    {
        "id": "write_test",
        "user": (
            "Write a pytest for a function add(a, b) that returns a+b. "
            "Use a tool to create tests/test_add.py."
        ),
    },
    {
        "id": "git_status",
        "user": "Check git status and summarize whether the working tree is clean. Use tools.",
    },
]


def get_text_tokenizer(tokenizer_or_processor):
    tok = tokenizer_or_processor
    if hasattr(tok, "encode") and callable(tok.encode):
        return tok
    if hasattr(tok, "tokenizer") and hasattr(tok.tokenizer, "encode"):
        return tok.tokenizer
    if hasattr(tok, "text_tokenizer") and hasattr(tok.text_tokenizer, "encode"):
        return tok.text_tokenizer
    raise AttributeError(
        f"No encode() on {type(tok)}; tried .tokenizer / .text_tokenizer"
    )


def build_text_inputs(tokenizer, messages: list[dict], device):
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text_tok = get_text_tokenizer(tokenizer)
        prompt = text_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if not isinstance(prompt, str):
        prompt = str(prompt)

    text_tok = get_text_tokenizer(tokenizer)
    ids = text_tok.encode(prompt, add_special_tokens=False, return_tensors="pt")
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return {
        "input_ids": ids.to(device),
        "attention_mask": torch.ones_like(ids).to(device),
    }


def score_agent_output(text: str) -> dict:
    t = text or ""
    lower = t.lower()
    has_think = bool(
        re.search(r"<think>.*?</think>", t, flags=re.DOTALL | re.IGNORECASE)
        or re.search(r"<thinking>.*?</thinking>", t, flags=re.DOTALL | re.IGNORECASE)
    )
    has_tool = bool(
        re.search(r"<tool_call>", t, flags=re.IGNORECASE)
        or re.search(r"```(?:json|tool|bash)", t, flags=re.IGNORECASE)
        or re.search(
            r'["\']name["\']\s*:\s*["\'](Bash|Read|Write|Edit|Shell|run_terminal)',
            t,
        )
        or ("tool_call" in lower and "arguments" in lower)
    )
    order_ok = False
    if has_think and has_tool:
        think_end = re.search(r"</think>|</thinking>", t, flags=re.IGNORECASE)
        tool_pos = re.search(r"<tool_call>|tool_call", t, flags=re.IGNORECASE)
        if think_end and tool_pos and think_end.start() < tool_pos.start():
            order_ok = True
    elif has_think and not has_tool:
        order_ok = True

    non_empty = len(t.strip()) > 40
    score = (
        int(non_empty)
        + int(has_think)
        + int(has_tool)
        + int(order_ok)
        + int(len(t.strip()) > 120)
    )
    return {
        "score_0_5": score,
        "non_empty": non_empty,
        "has_think": has_think,
        "has_tool_call": has_tool,
        "think_before_tool": order_ok,
        "chars": len(t),
    }


def load_holdout_prompts(path: Path, n: int, seed: int) -> list[dict]:
    if not path.exists() or n <= 0:
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows[: n * 3]:
        msgs = r.get("messages") or []
        user = ""
        for m in msgs:
            if m.get("role") == "user":
                user = (m.get("content") or "").strip()
        if len(user) < 20:
            continue
        if len(user) > 4000:
            user = user[:4000] + "\n…[truncated for eval]…"
        out.append(
            {
                "id": f"holdout_{len(out)}",
                "user": user,
                "meta": {"teacher": r.get("teacher")},
            }
        )
        if len(out) >= n:
            break
    return out


def generate_batch(model, tokenizer, system: str, probes: list[dict], max_new: int) -> list[dict]:
    import torch

    device = next(model.parameters()).device
    model.eval()
    results = []
    for i, p in enumerate(probes):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": p["user"]},
        ]
        inputs = build_text_inputs(tokenizer, messages, device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                use_cache=True,
            )
        dt = time.time() - t0
        prompt_len = inputs["input_ids"].shape[-1]
        text_tok = get_text_tokenizer(tokenizer)
        text = text_tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        sc = score_agent_output(text)
        print(
            f"  [{i+1}/{len(probes)}] {p['id']}: score={sc['score_0_5']} "
            f"think={sc['has_think']} tool={sc['has_tool_call']} ({dt:.1f}s)",
            flush=True,
        )
        results.append(
            {
                "id": p["id"],
                "user": p["user"],
                "completion": text,
                "metrics": sc,
                "seconds": round(dt, 2),
            }
        )
        # drop decode tensors
        del out, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def load_model(model_name: str, adapter: str | None, max_seq: int, load_in_4bit: bool):
    from unsloth import FastModel
    from peft import PeftModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        load_in_16bit=not load_in_4bit,
        full_finetuning=False,
    )
    if adapter:
        print(f"Loading adapter: {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
    FastModel.for_inference(model)
    return model, tokenizer


def build_probes(args, mix) -> list[dict]:
    holdout_path = ROOT / mix["output"].get(
        "holdout_path", "data/mixed/holdout_qwen_agent.jsonl"
    )
    probes = list(DEFAULT_PROBES)
    probes.extend(load_holdout_prompts(holdout_path, args.n_holdout, args.seed))
    return probes


def run_phase(args, phase: str) -> Path:
    """Run one model load in THIS process and write a partial JSON."""
    mix = yaml.safe_load(Path(args.config).read_text())
    model_name = mix["student"].get("unsloth_model_id") or mix["student"]["model_id"]
    system = mix.get("system_prompt", "You are a coding agent.")
    probes = build_probes(args, mix)
    print(f"Phase={phase}  Probes={len(probes)}  4bit={args.load_in_4bit}", flush=True)

    adapter = Path(args.adapter)
    if phase == "adapter" and not adapter.exists():
        raise SystemExit(f"Adapter not found: {adapter}")

    use_adapter = str(adapter) if phase == "adapter" else None
    label = "BASE + ADAPTER" if use_adapter else "BASE MODEL"
    print(f"\n=== {label} ===", flush=True)

    model, tokenizer = load_model(
        model_name, use_adapter, args.max_seq_length, args.load_in_4bit
    )
    results = generate_batch(model, tokenizer, system, probes, args.max_new_tokens)

    partial = {
        "phase": phase,
        "base_model": model_name,
        "adapter": str(adapter) if phase == "adapter" else None,
        "probes": [{"id": p["id"], "user": p["user"]} for p in probes],
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out.with_suffix(f".{phase}.json")
    partial_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False))
    print(f"Wrote partial: {partial_path}", flush=True)
    return partial_path


def merge_partials(args) -> None:
    out = Path(args.out)
    base_p = out.with_suffix(".base.json")
    ad_p = out.with_suffix(".adapter.json")
    if not base_p.exists() or not ad_p.exists():
        raise SystemExit(f"Missing partials: {base_p.exists()=} {ad_p.exists()=}")

    base = json.loads(base_p.read_text())
    ad = json.loads(ad_p.read_text())

    def avg_score(runs):
        if not runs:
            return None
        return sum(r["metrics"]["score_0_5"] for r in runs) / len(runs)

    def rate(runs, key):
        if not runs:
            return None
        return sum(1 for r in runs if r["metrics"][key]) / len(runs)

    base_runs = base["results"]
    ad_runs = ad["results"]
    base_s = avg_score(base_runs)
    ad_s = avg_score(ad_runs)

    report = {
        "base_model": base.get("base_model"),
        "adapter": ad.get("adapter"),
        "probes": base.get("probes"),
        "base": base_runs,
        "adapter_run": ad_runs,
        "summary": {
            "base_avg_score_0_5": base_s,
            "adapter_avg_score_0_5": ad_s,
            "delta": (ad_s - base_s) if base_s is not None and ad_s is not None else None,
            "base_think_rate": rate(base_runs, "has_think"),
            "adapter_think_rate": rate(ad_runs, "has_think"),
            "base_tool_rate": rate(base_runs, "has_tool_call"),
            "adapter_tool_rate": rate(ad_runs, "has_tool_call"),
            "n_probes": len(base_runs),
            "load_in_4bit": args.load_in_4bit,
        },
        "side_by_side": [],
    }

    by_id_base = {r["id"]: r for r in base_runs}
    by_id_ad = {r["id"]: r for r in ad_runs}
    for pid in by_id_base:
        b, a = by_id_base[pid], by_id_ad.get(pid)
        if not a:
            continue
        report["side_by_side"].append(
            {
                "id": pid,
                "user": b["user"][:500],
                "base_score": b["metrics"]["score_0_5"],
                "adapter_score": a["metrics"]["score_0_5"],
                "base_preview": b["completion"][:400],
                "adapter_preview": a["completion"][:400],
            }
        )

    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== SUMMARY ===", flush=True)
    for k, v in report["summary"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"\nFull report: {out}", flush=True)
    print(
        "\nNote: scores measure agent format (think/tool), not coding correctness. "
        "Read side_by_side / full completions in the JSON.",
        flush=True,
    )


def run_orchestrator(args) -> None:
    """Spawn isolated processes so CUDA memory is fully freed between loads."""
    script = str(Path(__file__).resolve())
    common = [
        sys.executable,
        "-u",
        script,
        "--config",
        args.config,
        "--adapter",
        args.adapter,
        "--n-holdout",
        str(args.n_holdout),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-seq-length",
        str(args.max_seq_length),
        "--seed",
        str(args.seed),
        "--out",
        args.out,
    ]
    if args.load_in_4bit:
        common.append("--load-in-4bit")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not args.skip_base:
        print("\n>>> Subprocess: BASE", flush=True)
        r = subprocess.run(common + ["--phase", "base"], env=env)
        if r.returncode != 0:
            raise SystemExit(f"Base phase failed with code {r.returncode}")
    else:
        print("Skipping base (using existing partial)", flush=True)

    print("\n>>> Subprocess: ADAPTER (fresh process = clean VRAM)", flush=True)
    r = subprocess.run(common + ["--phase", "adapter"], env=env)
    if r.returncode != 0:
        raise SystemExit(f"Adapter phase failed with code {r.returncode}")

    print("\n>>> Merge partials", flush=True)
    merge_partials(args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mix.yaml"))
    ap.add_argument(
        "--adapter",
        default=str(ROOT / "outputs" / "qwen36-agent-lora" / "lora_adapter"),
    )
    ap.add_argument("--n-holdout", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "qwen36-agent-lora" / "compare_report.json"),
    )
    ap.add_argument(
        "--phase",
        choices=["all", "base", "adapter", "merge"],
        default="all",
        help="all=isolated subprocesses; base/adapter=single load; merge=combine JSON",
    )
    ap.add_argument(
        "--skip-base",
        action="store_true",
        help="In phase=all, skip base if compare_report.base.json already exists",
    )
    ap.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4bit for lower VRAM (recommended after OOM)",
    )
    args = ap.parse_args()

    # Auto-skip base if partial already exists
    base_partial = Path(args.out).with_suffix(".base.json")
    if args.phase == "all" and args.skip_base and not base_partial.exists():
        print("No base partial found; will run base.", flush=True)
        args.skip_base = False
    if args.phase == "all" and base_partial.exists() and not args.skip_base:
        # If user re-runs after OOM mid-adapter, allow --skip-base
        pass

    if args.phase == "all":
        # If base partial exists from crashed run, prefer skip to save time
        if base_partial.exists() and args.skip_base:
            print(f"Using existing {base_partial}", flush=True)
        run_orchestrator(args)
    elif args.phase == "merge":
        merge_partials(args)
    else:
        run_phase(args, args.phase)


if __name__ == "__main__":
    main()
