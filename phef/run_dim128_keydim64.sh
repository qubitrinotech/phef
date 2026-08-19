#!/bin/bash
cd ~/qubitrino-projects/agi/phef
export PYTHONPATH=.
PYTHON_EXEC=/home/linux/qubitrino-projects/agi/venv/bin/python

BATCH=64
EVAL_BATCH=128
SEG=16384
EPOCHS=100000
OUT_DIR=results/capacity_scan_dim128_keydim64
LOG_DIR=logs/capacity_scan_dim128_keydim64
mkdir -p "$OUT_DIR" "$LOG_DIR"

for seed in 0 1 2; do
  nohup $PYTHON_EXEC -u experiments/analyze_memory.py \
    --world gpu --dim 128 --key-dim 64 --n-keys 512 --n-values 64 \
    --n-segments $SEG --overwrite-p 0.3 --batch-size $BATCH \
    --eval-batch-size $EVAL_BATCH --eval-batches 20 --lr 1e-3 \
    --epochs $EPOCHS --acc-every 250 --acc-n 2048 --use-best \
    --seed $seed --select-metric soft \
    --out "$OUT_DIR/dim128_seg${SEG}_s${seed}_e${EPOCHS}.json" \
    > "$LOG_DIR/dim128_seg${SEG}_s${seed}_e${EPOCHS}.log" 2>&1 &
done
wait
echo "All key_dim=64 runs finished."
