#!/bin/bash
cd ~/qubitrino-projects/agi/phef
export PYTHONPATH=.
PYTHON_EXEC=/home/linux/qubitrino-projects/agi/venv/bin/python

echo "=== Запуск dim=128 extended (16384, 32768) ==="

for seg in 16384 32768; do
  echo "=== $(date): dim=128, seg=$seg ==="
  # бюджет эпох: для 16384 - 100k, для 32768 - 150k
  if [ $seg -eq 16384 ]; then
    EPOCHS=100000
  else
    EPOCHS=150000
  fi
  for seed in 0 1 2; do
    nohup $PYTHON_EXEC -u experiments/analyze_memory.py \
      --world gpu --dim 128 --key-dim 32 --n-keys 512 --n-values 64 \
      --n-segments $seg --overwrite-p 0.3 --batch-size 256 \
      --eval-batch-size 512 --eval-batches 20 --lr 1e-3 --epochs $EPOCHS \
      --acc-every 250 --acc-n 2048 --use-best --seed $seed --select-metric soft \
      --out results/capacity_scan_dim128_extended/dim128_seg${seg}_s${seed}_e${EPOCHS}.json \
      > logs/capacity_scan_dim128_extended/dim128_seg${seg}_s${seed}_e${EPOCHS}.log 2>&1 &
  done
  wait
  echo "=== $(date): dim=128 seg=$seg завершён ==="
done

echo "=== $(date): Все прогоны dim=128 extended завершены! ==="
