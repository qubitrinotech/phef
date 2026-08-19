#!/bin/bash
cd ~/qubitrino-projects/agi/phef
export PYTHONPATH=.
PYTHON_EXEC=/home/linux/qubitrino-projects/agi/venv/bin/python

mkdir -p logs/capacity_scan_dim32
mkdir -p logs/capacity_scan_dim256

# Серия 1: dim=32, seg=16384 (3 сида)
for seed in 0 1 2; do
  echo "=== dim=32 seg=16384 seed=$seed start $(date) ==="
  $PYTHON_EXEC -u experiments/analyze_memory.py \
    --world gpu --dim 32 --key-dim 32 --n-keys 512 --n-values 64 \
    --n-segments 16384 --overwrite-p 0.3 --batch-size 64 \
    --eval-batch-size 128 --eval-batches 20 --lr 1e-3 \
    --epochs 50000 --acc-every 250 --acc-n 2048 --use-best \
    --seed $seed --select-metric soft \
    --out results/capacity_scan_dim32/dim32_seg16384_s${seed}_e50k.json \
    > logs/capacity_scan_dim32/dim32_seg16384_s${seed}_e50k.log 2>&1
  echo "=== dim=32 seg=16384 seed=$seed finished $(date) ==="
done

# Серия 2: dim=256, seg=4096 (3 сида)
for seed in 0 1 2; do
  echo "=== dim=256 seg=4096 seed=$seed start $(date) ==="
  $PYTHON_EXEC -u experiments/analyze_memory.py \
    --world gpu --dim 256 --key-dim 32 --n-keys 512 --n-values 64 \
    --n-segments 4096 --overwrite-p 0.3 --batch-size 64 \
    --eval-batch-size 128 --eval-batches 20 --lr 1e-3 \
    --epochs 50000 --acc-every 250 --acc-n 2048 --use-best \
    --seed $seed --select-metric soft \
    --out results/capacity_scan_dim256/dim256_seg4096_s${seed}_e50k.json \
    > logs/capacity_scan_dim256/dim256_seg4096_s${seed}_e50k.log 2>&1
  echo "=== dim=256 seg=4096 seed=$seed finished $(date) ==="
done

echo "=== Все проверки завершены $(date) ==="
