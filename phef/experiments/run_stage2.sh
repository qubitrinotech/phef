#!/usr/bin/env bash
set -uo pipefail

mkdir -p results results/_scratch logs

TOTAL=3
DONE=0
START=$(date +%s)

progress() {
  local label="$1" ep="$2" total_ep="$3" extra="$4"
  local width=28
  local filled=0
  [ "$total_ep" -gt 0 ] && filled=$(( ep * width / total_ep ))
  local pct=0
  [ "$total_ep" -gt 0 ] && pct=$(( ep * 100 / total_ep ))
  local el=$(( $(date +%s) - START ))

  printf "\r\033[K[%d/%d] %-14s [" "$DONE" "$TOTAL" "$label"
  printf "%${filled}s" | tr ' ' '#'
  printf "%$(( width - filled ))s" | tr ' ' '.'
  printf "] %3d%%  %5d/%d  %s  всего %dм" \
    "$pct" "$ep" "$total_ep" "$extra" $(( el / 60 ))
}

run() {
  local label="$1" epochs="$2" out="$3"; shift 3
  local log="logs/${label}.log"

  python -u -m experiments.analyze_memory "$@" \
    --epochs "$epochs" --out "$out" > "$log" 2>&1 &
  local pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    local line ep loss
    line=$(grep -E "^epoch" "$log" 2>/dev/null | tail -1)
    ep=$(echo "$line" | awk '{print $2}')
    loss=$(echo "$line" | awk '{print "loss "$4}')
    progress "$label" "${ep:-0}" "$epochs" "${loss:-запуск}"
    sleep 2
  done

  wait "$pid"
  DONE=$(( DONE + 1 ))

  local acc
  acc=$(grep "answer_acc" "$log" | awk '{print $3}')
  progress "$label" "$epochs" "$epochs" "acc ${acc:-ERR}"
  echo
}

run "kd1" 5000 "results/curve_kd1_s0.json" \
  --n-segments 256 --n-keys 512 --key-dim 1 \
  --batch-size 512 --seed 0

run "kd2_long" 15000 "results/_scratch/kd2_long.json" \
  --n-segments 256 --n-keys 512 --key-dim 2 \
  --batch-size 512 --seed 0

run "kd3_seg1024" 10000 "results/_scratch/kd3_seg1024.json" \
  --n-segments 1024 --n-keys 2048 --key-dim 3 \
  --batch-size 256 --seed 0

echo
echo "Готово за $(( ($(date +%s) - START) / 60 )) минут"
echo
for t in kd1 kd2_long kd3_seg1024; do
  echo "── $t ──"
  grep -E "answer_acc|oracle|tie_rate|escape" "logs/${t}.log"
done
