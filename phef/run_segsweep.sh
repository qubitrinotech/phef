#!/bin/bash
cd ~/qubitrino-projects/agi/phef
for S in 64 128 256 512 1024; do
  echo "=== n_segments $S начато $(date +%H:%M:%S) ==="
  for SEED in 0 1 2; do
    LC_ALL=C PYTHONPATH=. nohup python -u experiments/analyze_memory.py \
      --n-segments $S --n-keys 512 --key-dim 32 --dim 8 \
      --epochs 200000 --batch-size 256 --eval-batch-size 512 \
      --eval-batches 20 --use-best --acc-every 250 --acc-n 2048 \
      --seed $SEED --world gpu \
      --out results/v5/segsweep_dim8_gpu_seg${S}_s${SEED}_e200k.json \
      > logs/v5/segsweep_seg${S}_s${SEED}.log 2>&1 &
  done
  wait
  echo "=== n_segments $S закончено $(date +%H:%M:%S) ==="
done
echo "=== развёртка завершена $(date +%H:%M:%S) ==="
