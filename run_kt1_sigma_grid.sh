#!/usr/bin/env bash
set -Eeuo pipefail
cd "$HOME/agi"

PY=".venv/bin/python"
EV="phef/experiments/eval_recall_compact.py"
C7="phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt"
C23="phef/results/v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt"
mkdir -p artifacts/kt1/phef phef/logs

for SIG in 0.30 0.50 0.70 1.0 1.5; do
for PAIR in "7:$C7" "23:$C23"; do
  TAG="${PAIR%%:*}"; CK="${PAIR#*:}"
  for ES in 7 11; do
    OUT="artifacts/kt1/phef/kt1_sigma${SIG}_ckpt${TAG}_eval${ES}.json"
    LOG="phef/logs/kt1_sigma${SIG}_ckpt${TAG}_eval${ES}.log"
    [[ -e "$OUT" ]] && { echo "skip $OUT"; continue; }
    echo "=== sigma=$SIG ckpt$TAG eval$ES $(date -Is) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=phef "$PY" "$EV" \
      --checkpoint "$CK" \
      --n-keys 1073741824 --n-segments 14000000 \
      --overwrite-p 0.3 --n-values 64 --key-dim 32 --dim 128 \
      --eval-batch-size 2 --eval-batches 32 --seed "$ES" \
      --progress-every 8 --stream-hard-chunk-size 65536 \
      --query-noise-sigma "$SIG" --noise-seed 4242 \
      --out "$OUT" 2>&1 | tee "$LOG"
  done
done
done
echo "ALL KT-1 SIGMA CELLS DONE $(date -Is)"
