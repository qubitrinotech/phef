#!/usr/bin/env bash
set -euo pipefail

mkdir -p results logs

KDS=(2 3 4 5 6)
SEEDS=(0 1 2)
TOTAL=$(( ${#KDS[@]} * ${#SEEDS[@]} ))
DONE=0
START=$(date +%s)

bar() {
  local done=$1 total=$2 label=$3
  local width=30
  local filled=$(( done * width / total ))
  local pct=$(( done * 100 / total ))
  local elapsed=$(( $(date +%s) - START ))
  local eta="--"

  if [ "$done" -gt 0 ]; then
    eta=$(( elapsed * (total - done) / done / 60 ))
    eta="${eta}m"
  fi

  printf "\r["
  printf "%${filled}s" | tr ' ' '#'
  printf "%$(( width - filled ))s" | tr ' ' '.'
  printf "] %3d%% %2d/%d  elapsed %dm  eta %s  %-16s" \
    "$pct" "$done" "$total" $(( elapsed / 60 )) "$eta" "$label"
}

for kd in "${KDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="curve_kd${kd}_s${seed}"
    bar "$DONE" "$TOTAL" "$tag"

    python -u -m experiments.analyze_memory \
      --n-segments 256 --n-keys 512 --key-dim "${kd}" \
      --overwrite-p 0.3 --epochs 3000 --batch-size 512 \
      --seed "${seed}" \
      --out "results/${tag}.json" > "logs/${tag}.log" 2>&1

    DONE=$(( DONE + 1 ))
    acc=$(grep "answer_acc" "logs/${tag}.log" | awk '{print $3}')
    bar "$DONE" "$TOTAL" "$tag acc=${acc}"
    echo
  done
done

echo
echo "Готово за $(( ($(date +%s) - START) / 60 )) минут"
