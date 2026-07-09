#!/usr/bin/env bash
# One-shot pipeline driver. Safe to re-run; downloads skip existing files.
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE="${SMOKE:-0}"
SKIP_SETUP="${SKIP_SETUP:-0}"

if [[ "$SKIP_SETUP" != "1" ]]; then
  bash scripts/runpod_setup.sh
fi

if [[ "$SMOKE" == "1" ]]; then
  echo "==> SMOKE MODE (small data + short train)"
  python3 scripts/01_download_datasets.py --smoke
  python3 scripts/02_normalize_and_mix.py --smoke
  python3 scripts/03_train_unsloth.py --smoke
  python3 scripts/04_export_and_smoke.py
else
  echo "==> FULL RUN"
  python3 scripts/01_download_datasets.py
  python3 scripts/02_normalize_and_mix.py
  python3 scripts/03_train_unsloth.py
  python3 scripts/04_export_and_smoke.py
fi

echo ""
echo "All steps finished. Adapter: outputs/qwen36-agent-lora/lora_adapter"
