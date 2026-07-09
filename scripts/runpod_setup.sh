#!/usr/bin/env bash
# Install GPU training stack on a RunPod (or any CUDA) machine.
set -euo pipefail

echo "==> TraceCapture RunPod setup"
echo "Python: $(python3 --version 2>/dev/null || true)"
echo "CUDA:   $(nvidia-smi -L 2>/dev/null || echo 'nvidia-smi not found')"

cd "$(dirname "$0")/.."

python3 -m pip install -U pip wheel setuptools

# Data deps
python3 -m pip install -r requirements.txt

# Unsloth (installs compatible torch/transformers)
# See https://unsloth.ai/docs
python3 -m pip install -U unsloth

# Common training extras
python3 -m pip install -U trl peft accelerate bitsandbytes datasets huggingface_hub

# Optional: login if you need gated models/datasets
if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
  echo "Hugging Face login OK"
else
  echo "Note: set HF_TOKEN if downloads fail for gated repos"
fi

mkdir -p data/raw data/cleaned data/mixed data/local/grok outputs

echo ""
echo "Setup complete."
echo "Next:"
echo "  python scripts/01_download_datasets.py        # or --smoke first"
echo "  python scripts/02_normalize_and_mix.py"
echo "  python scripts/03_train_unsloth.py"
echo "  python scripts/04_export_and_smoke.py"
