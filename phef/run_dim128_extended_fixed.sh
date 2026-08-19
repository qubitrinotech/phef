#!/bin/bash
cd ~/qubitrino-projects/agi/phef
export PYTHONPATH=.
PYTHON_EXEC=/home/linux/qubitrino-projects/agi/venv/bin/python

BATCH=64
EVAL_BATCH=128

echo "=== Запуск dim=128 extended с batch_size=$BATCH ==="

for seg in 16384 32768; do
  echo "=== $(date): dim=128, seg=$seg ==="
  EPOCHS=100000
  if [ $seg -eq 32768 ]; then EPOCHS=150000; fi
  for seed in 0 1 2; do
    nohup $PYTHON_EXEC -u experiments/analyze_memory.py \
      --world gpu --dim 128 --key-dim 32 --n-keys 512 --n-values 64 \
      --n-segments $seg --overwrite-p 0.3 --batch-size $BATCH \
      --eval-batch-size $EVAL_BATCH --eval-batches 20 --lr 1e-3 \
      --epochs $EPOCHS --acc-every 250 --acc-n 2048 --use-best \
      --seed $seed --select-metric soft \
      --out results/capacity_scan_dim128_extended/dim128_seg${seg}_s${seed}_e${EPOCHS}.json \
      > logs/capacity_scan_dim128_extended/dim128_seg${seg}_s${seed}_e${EPOCHS}.log 2>&1 &
  done
  wait
  echo "=== $(date): dim=128 seg=$seg завершён ==="
done

echo "=== $(date): Все прогоны dim=128 extended завершены! ==="
