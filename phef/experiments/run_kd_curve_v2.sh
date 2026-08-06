#!/usr/bin/env bash
set -uo pipefail
mkdir -p results/v2 logs
START=$(date +%s); DONE=0; TOTAL=18

for kd in 1 2 3 4 5 6; do
  for seed in 0 1 2; do
    tag="v2_kd${kd}_s${seed}"
    log="logs/${tag}.log"

    python -u -m experiments.analyze_memory \
      --n-segments 256 --n-keys 512 --key-dim "${kd}" \
      --epochs 3000 --batch-size 512 --seed "${seed}" \
      --use-best \
      --out "results/v2/${tag}.json" > "$log" 2>&1 &
    pid=$!

    while kill -0 $pid 2>/dev/null; do
      ep=$(grep -E "^epoch" "$log" 2>/dev/null | tail -1 | awk '{print $2}')
      ep=${ep:-0}
      printf "\r\033[K[%2d/%d] %-14s %5d/5000  %dм" \
        "$DONE" "$TOTAL" "$tag" "$ep" $(( ($(date +%s)-START)/60 ))
      sleep 2
    done
    wait $pid

    DONE=$((DONE+1))
    acc=$(grep "answer_acc" "$log" | awk '{print $3}')
    printf "\r\033[K[%2d/%d] %-14s acc %s\n" "$DONE" "$TOTAL" "$tag" "$acc"
  done
done
echo "Готово за $(( ($(date +%s)-START)/60 ))м"
