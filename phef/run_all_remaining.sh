#!/bin/bash
cd ~/qubitrino-projects/agi/phef
export PYTHONPATH=.
mkdir -p results/capacity_scan_dim32 results/capacity_scan_dim128 results/capacity_scan_dim64_extended
mkdir -p logs/capacity_scan_dim32 logs/capacity_scan_dim128 logs/capacity_scan_dim64_extended

echo "=== Запуск всех оставшихся конфигураций ==="

# dim=32: seg=2048, 4096, 8192
for seg in 2048 4096 8192; do
  echo "=== $(date): dim=32, seg=$seg ==="
  for seed in 0 1 2; do
    nohup env PYTHONPATH=. python -u experiments/analyze_memory.py \
      --world gpu --dim 32 --key-dim 32 --n-keys 512 --n-values 64 \
      --n-segments $seg --overwrite-p 0.3 --batch-size 256 \
      --eval-batch-size 512 --eval-batches 20 --lr 1e-3 --epochs 30000 \
      --acc-every 250 --acc-n 2048 --use-best --seed $seed --select-metric soft \
      --out results/capacity_scan_dim32/dim32_seg${seg}_s${seed}_e30k.json \
      > logs/capacity_scan_dim32/dim32_seg${seg}_s${seed}_e30k.log 2>&1 &
  done
  wait
  echo "=== $(date): dim=32 seg=$seg завершён ==="
done

# dim=128: seg=4096, 8192
for seg in 4096 8192; do
  echo "=== $(date): dim=128, seg=$seg ==="
  for seed in 0 1 2; do
    nohup env PYTHONPATH=. python -u experiments/analyze_memory.py \
      --world gpu --dim 128 --key-dim 32 --n-keys 512 --n-values 64 \
      --n-segments $seg --overwrite-p 0.3 --batch-size 256 \
      --eval-batch-size 512 --eval-batches 20 --lr 1e-3 --epochs 50000 \
      --acc-every 250 --acc-n 2048 --use-best --seed $seed --select-metric soft \
      --out results/capacity_scan_dim128/dim128_seg${seg}_s${seed}_e50k.json \
      > logs/capacity_scan_dim128/dim128_seg${seg}_s${seed}_e50k.log 2>&1 &
  done
  wait
  echo "=== $(date): dim=128 seg=$seg завершён ==="
done

# dim=64: seg=8192
for seg in 8192; do
  echo "=== $(date): dim=64, seg=$seg ==="
  for seed in 0 1 2; do
    nohup env PYTHONPATH=. python -u experiments/analyze_memory.py \
      --world gpu --dim 64 --key-dim 32 --n-keys 512 --n-values 64 \
      --n-segments $seg --overwrite-p 0.3 --batch-size 256 \
      --eval-batch-size 512 --eval-batches 20 --lr 1e-3 --epochs 50000 \
      --acc-every 250 --acc-n 2048 --use-best --seed $seed --select-metric soft \
      --out results/capacity_scan_dim64_extended/dim64_seg${seg}_s${seed}_e50k.json \
      > logs/capacity_scan_dim64_extended/dim64_seg${seg}_s${seed}_e50k.log 2>&1 &
  done
  wait
  echo "=== $(date): dim=64 seg=$seg завершён ==="
done

echo "=== $(date): Все прогоны завершены! ==="