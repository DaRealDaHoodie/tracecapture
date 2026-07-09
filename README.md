# TraceCapture

Private multi-teacher fine-tune of **Qwen3.6-35B-A3B** on coding-agent traces
(Fable, GPT-5.5, open SWE agents, optional Grok exports).

You rent the GPU. The scripts do the rest.

## What you get

1. Download strong Hugging Face agent/SWE datasets
2. Convert them into one Qwen chat format with `<think>…</think>` + tool actions
3. Mix teachers by ratio (see `configs/mix.yaml`)
4. LoRA fine-tune with Unsloth on RTX PRO 6000 Blackwell (96GB)
5. Smoke-test the adapter

**Personal use only** (as you asked). Do not publish weights trained on
restricted dumps if you care about licenses; private local/pod use is fine.

---

## Hardware (RunPod)

| Option | Recommendation |
|--------|----------------|
| **1× RTX PRO 6000 Blackwell (96GB)** | Enough for full bf16 LoRA @ 8k context |
| **2× PRO 6000** | Faster epochs later; not required for v1 |
| Disk | **≥ 300GB** free (datasets + model + checkpoints) |
| RAM | ≥ 128GB system RAM preferred |
| Template | PyTorch 2.x + CUDA 12.x (RunPod PyTorch template is fine) |

You do **not** need 2 GPUs for the first successful model.

---

## You only need to do these human steps

### A. On your laptop (once)

1. Open a terminal in this repo (already at `tracecapture`).
2. Push to a **private** GitHub repo (optional but easiest for RunPod):

   ```bash
   cd /Users/tweaver/Developer/GitRepos/tracecapture
   git init
   git add .
   git commit -m "TraceCapture pipeline for private Qwen agent SFT"
   # create private repo on GitHub, then:
   git remote add origin git@github.com:YOU/tracecapture.git
   git push -u origin main
   ```

3. Get a [Hugging Face token](https://huggingface.co/settings/tokens)
   (read access). Some datasets are nicer with auth.

### B. On RunPod

1. **Deploy** a pod: RTX PRO 6000 Blackwell, large disk, PyTorch/CUDA image.
2. Open the pod **web terminal** (or Jupyter terminal).
3. Clone and run:

```bash
# paste your HF token (keeps downloads reliable)
export HF_TOKEN="hf_xxxxxxxx"
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace && cd /workspace

git clone https://github.com/YOU/tracecapture.git
cd tracecapture

# --- first time: prove the pipeline works (30–90 min) ---
export SMOKE=1
bash scripts/00_all.sh

# --- when smoke looks OK: full train (many hours) ---
export SMOKE=0
bash scripts/00_all.sh
```

If you cannot use git, upload a zip of this folder to `/workspace/tracecapture`
via RunPod’s file browser, then:

```bash
cd /workspace/tracecapture
export HF_TOKEN="hf_xxxxxxxx"
export SMOKE=1
bash scripts/00_all.sh
```

### C. After training

- Adapter weights: `outputs/qwen36-agent-lora/lora_adapter/`
- Smoke text: `outputs/qwen36-agent-lora/smoke_report.json`
- Download the `lora_adapter` folder to your machine (RunPod download / `scp`).

Optional merge (large disk):

```bash
python scripts/04_export_and_smoke.py --merge
```

---

## What each script does

| Script | Role |
|--------|------|
| `scripts/runpod_setup.sh` | Installs Python deps + Unsloth |
| `scripts/01_download_datasets.py` | Pulls HF teachers into `data/raw/` |
| `scripts/02_normalize_and_mix.py` | Converts + mixes → `data/mixed/train_qwen_agent.jsonl` |
| `scripts/03_train_unsloth.py` | LoRA SFT on Qwen3.6-35B-A3B |
| `scripts/04_export_and_smoke.py` | Quick generation check |
| `scripts/00_all.sh` | Runs all of the above |

Flags:

- `--smoke` / `SMOKE=1` — tiny download + short train (pipeline test)
- Re-running download **skips** files already present

---

## Teachers in the mix (`configs/mix.yaml`)

Default blend (approximate shares):

| Share | Teacher | Dataset |
|------:|---------|---------|
| 30% | Open SWE verified | `togethercomputer/CoderForge-Preview` |
| 15% | Open SWE thinking | `nvidia/Open-SWE-Traces` |
| 10% | OpenHands trajs | `nebius/SWE-rebench-openhands-trajectories` |
| 5% | Nemotron SWE | `nvidia/Nemotron-SFT-SWE-v2` |
| 15% | Fable CoT | `Glint-Research/Fable-5-traces` |
| 5% | Fable verified compile | `Crownelius/Complete-FABLE.5-traces-2M` |
| 5% | Claude Code merge | `nlile/misc-merged-claude-code-traces-v1` |
| 8% | GPT-5.5 agent | `AletheiaResearch/GPT-5.5-Codex` |
| 3%+2% | Tools / OpenCode | Nemotron agentic + OpenCode |
| 0% | Grok (optional) | Drop your exports into `data/local/grok/` |

To enable Grok local exports:

1. Put `.jsonl` sessions under `data/local/grok/`
2. In `configs/mix.yaml`, set `grok_local.enabled: true` and raise `share` a bit
3. Re-run `02_normalize_and_mix.py` then train

To turn a broken HF source off: `enabled: false` on that teacher.

---

## Dual GPU (optional)

1× is enough. For 2× PRO 6000 later:

- Install the same stack on a multi-GPU pod
- Use Hugging Face Accelerate / `torchrun` (can add a dedicated multi-GPU
  launcher when you need it)
- Or run two jobs (data prep on one, train on the other) — usually overkill

---

## If something fails

| Symptom | Fix |
|---------|-----|
| Dataset download 401/403 | `export HF_TOKEN=...` and re-run `01` |
| One dataset errors | Script continues; set that teacher `enabled: false` |
| CUDA OOM | In `configs/mix.yaml` train section: lower `max_seq_length` to `4096`, raise `gradient_accumulation_steps` |
| Unsloth install fails | Use a fresh PyTorch CUDA pod image; re-run `runpod_setup.sh` |
| Empty mix | Run `01` successfully first; check `data/raw/*.jsonl` sizes |
| Gibberish smoke output | Train longer / full data; confirm smoke used full `03` not only 20 steps |

---

## Project layout

```text
tracecapture/
  configs/mix.yaml          # teachers + train hyperparams
  scripts/                  # pipeline steps
  src/schema.py             # chat example format
  src/converters.py         # HF row → Qwen messages
  data/                     # created on the pod (gitignored)
  outputs/                  # LoRA adapter (gitignored)
  requirements.txt
  README.md
```

---

## What I (the agent) already did for you

- Multi-teacher mix config (Fable + GPT-5.5 + open SWE + tools)
- Download / normalize / mix / train / smoke scripts
- RunPod setup + one-command driver (`00_all.sh`)
- Qwen `<think>` wrapping and tool-call serialization for agent rows

## What you do next

1. Create RunPod **1× RTX PRO 6000 Blackwell**, **300GB+** disk  
2. Clone/upload this repo  
3. `export HF_TOKEN=...` and `SMOKE=1 bash scripts/00_all.sh`  
4. If smoke works, full run with `SMOKE=0`  
5. Download `outputs/qwen36-agent-lora/lora_adapter`

When the pod is up, paste any error log here and I’ll fix the pipeline live.
