#!/usr/bin/env bash
set -euo pipefail

echo "### PHEF memory ###"
python -m experiments.train_recall \
  --n-keys 32 \
  --n-values 64 \
  --n-segments 16 \
  --overwrite-p 0.3 \
  --top-k 8 \
  --epochs 1000 \
  --batch-size 512

echo
echo "### Baseline: no memory ###"
python -m experiments.train_recall \
  --n-keys 32 \
  --n-values 64 \
  --n-segments 16 \
  --overwrite-p 0.3 \
  --epochs 1000 \
  --batch-size 512 \
  --no-memory
