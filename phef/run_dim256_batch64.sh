#!/bin/bash
cd ~/qubitrino-projects/agi/phef
export PYTHONPATH=.
PYTHON_EXEC=/home/linux/qubitrino-projects/agi/venv/bin/python

echo "=== Запуск dim=256 с batch_size=64 ==="

for seg in 16384 32768 65536; do
  echo "=== $(date): dim=256, seg=$seg ==="
  for seed in 0 1 2; do
    nohup $PYTHON_EXEC -u experiments/analyze_memory.py \
      --world gpu --dim 256 --key-dim 32 --n-keys 512 --n-values 64 \
      --n-segments $seg --overwrite-p 0.3 --batch-size 64 \
      --eval-batch-size 128 --eval-batches 20 --lr 1e-3 --epochs 100000 \
      --acc-every 250 --acc-n 2048 --use-best --seed $seed --select-metric soft \
      --out results/capacity_scan_dim256/dim256_seg${seg}_s${seed}_e100k.json \
      > logs/capacity_scan_dim256/dim256_seg${seg}_s${seed}_e100k.log 2>&1 &
  done
  wait
  echo "=== $(date): dim=256 seg=$seg завершён ==="
done

echo "=== $(date): Все прогоны dim=256 завершены! ==="
