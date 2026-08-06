#!/usr/bin/env bash
set -uo pipefail
mkdir -p results/_scratch logs
START=$(date +%s); DONE=0; TOTAL=4

run() {
  local label="$1" epochs="$2"; shift 2
  local log="logs/${label}.log"

  python -u -m experiments.analyze_memory "$@" \
    --epochs "$epochs" --batch-size 256 \
    --out "results/_scratch/${label}.json" > "$log" 2>&1 &
  local pid=$!

  while kill -0 $pid 2>/dev/null; do
    local line ep loss
    line=$(grep -E "^epoch" "$log" 2>/dev/null | tail -1)
    ep=$(echo "$line" | awk '{print $2}'); ep=${ep:-0}
    loss=$(echo "$line" | awk '{print $4}')
    dev=$(head -1 "$log" 2>/dev/null | grep -o "device=[a-z]*" | cut -d= -f2)
    printf "\r\033[K[%d/%d] %-16s %-4s %5d/%s  loss %-7s  %dм" \
      "$DONE" "$TOTAL" "$label" "${dev:-?}" "$ep" "$epochs" \
      "${loss:-...}" $(( ($(date +%s)-START)/60 ))
    sleep 2
  done
  wait $pid

  DONE=$((DONE+1))
  local acc drop
  acc=$(grep "answer_acc" "$log" | awk '{print $3}')
  printf "\r\033[K[%d/%d] %-16s acc %s\n" "$DONE" "$TOTAL" "$label" "$acc"
}

run "s3_kd3_seg1024" 2000 \
  --n-segments 1024 --n-keys 2048 --key-dim 3 --seed 0

run "s3_kd3_seg2048" 2500 \
  --n-segments 2048 --n-keys 4096 --key-dim 3 --seed 0

run "s3_kd6_seg2048" 2500 \
  --n-segments 2048 --n-keys 4096 --key-dim 6 --seed 0

run "s3_kd32_seg2048" 2500 \
  --n-segments 2048 --n-keys 4096 --key-dim 32 --seed 0

echo
echo "Готово за $(( ($(date +%s)-START)/60 ))м"
echo
for t in s3_kd3_seg1024 s3_kd3_seg2048 s3_kd6_seg2048 s3_kd32_seg2048; do
  echo "── $t ──"
  grep -E "answer_acc|oracle m0.05|enc_m0.05|tie_rate|best_loss" \
    "logs/${t}.log"
  echo "  distance: $(grep -A9 'acc by distance' "logs/${t}.log" \
    | grep bucket | awk '{printf "%s ", $3}')"
done
