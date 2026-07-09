#!/bin/bash
set -e
cd /workspace/tracecapture
LOG=data/pipeline.log
echo "$(date) waiting for download..." | tee -a "$LOG"
while pgrep -f "python3 scripts/01_download" >/dev/null; do sleep 20; done
echo "$(date) download finished" | tee -a "$LOG"
tr "\r" "\n" < data/download.log | grep -E "wrote |ERROR|\[skip\]|Done" | tee -a "$LOG" || true
ls -lh data/raw/ | tee -a "$LOG"
echo "$(date) starting normalize+mix" | tee -a "$LOG"
python3 scripts/02_normalize_and_mix.py 2>&1 | tee -a "$LOG"
echo "$(date) MIX COMPLETE" | tee -a "$LOG"
ls -lh data/mixed/ data/cleaned/ 2>/dev/null | tee -a "$LOG"
