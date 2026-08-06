#!/usr/bin/env bash
set -euo pipefail

mkdir -p results logs

run() {
  local tag="$1"; shift
  echo "════════ ${tag} ════════"
  python -u -m experiments.analyze_memory "$@" \
    --out "results/${tag}.json" 2>&1 | tee "logs/${tag}.log"
  echo
}

# ── Ось 1: коллизии через сжатие key_dim ────────────────────────
for kd in 16 8 4 2; do
  run "kd${kd}" \
    --n-segments 256 --n-keys 512 --key-dim "${kd}" \
    --epochs 8000 --batch-size 128
done

# ── Ось 2: временное разрешение при жёстком overwrite ───────────
run "overwrite95" \
  --n-segments 256 --n-keys 16 --key-dim 32 \
  --overwrite-p 0.95 --epochs 8000 --batch-size 128

run "overwrite80" \
  --n-segments 256 --n-keys 16 --key-dim 32 \
  --overwrite-p 0.80 --epochs 8000 --batch-size 128

# ── Ось 3: масштаб дистракторов ─────────────────────────────────
run "seg1024" \
  --n-segments 1024 --n-keys 2048 --key-dim 32 \
  --epochs 16000 --batch-size 64
