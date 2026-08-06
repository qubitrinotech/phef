#!/usr/bin/env bash
# Capacity sweep: досбор сидов на широких dim + длинный dim 8.
# Идемпотентно: конфигурации с существующим JSON пропускаются.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.

mkdir -p results/capacity logs

SEG=256
KEYS=512
KD=32
BATCH=256
EVAL_BS=512
EVAL_BATCHES=20

START=$(date +%s)

run_one() {
    local dim=$1 seed=$2 epochs=$3
    local tag="dim${dim}_kd32_seg${SEG}_s${seed}_e$(( epochs / 1000 ))k"
    local out="results/capacity/${tag}.json"
    local log="logs/cap_${tag}.log"

    if [ -f "$out" ]; then
        echo "skip  ${tag} (уже есть)"
        return 0
    fi

    echo "start ${tag}  epochs=${epochs}  $(date +%H:%M:%S)"

    python -u experiments/analyze_memory.py \
        --n-segments "$SEG" --n-keys "$KEYS" --key-dim "$KD" --dim "$dim" \
        --epochs "$epochs" --batch-size "$BATCH" \
        --eval-batch-size "$EVAL_BS" --eval-batches "$EVAL_BATCHES" \
        --use-best --acc-every 250 --acc-n 2048 --seed "$seed" \
        --out "$out" > "$log" 2>&1

    local status=$?

    if [ $status -ne 0 ]; then
        echo "FAIL  ${tag} (exit ${status}), см. ${log}"
        return 1
    fi

    local acc esc
    acc=$(grep -m1 "^answer_acc" "$log" | awk '{print $3}')
    esc=$(grep -m1 "^escape_epoch" "$log" | awk '{print $3}')
    echo "done  ${tag}  acc=${acc}  escape=${esc}  ($(( ($(date +%s) - START) / 60 ))м всего)"
}

echo "=== A: длинный dim 8, 30000 эпох, 3 сида ==="
for seed in 0 1 2; do
    run_one 8 "$seed" 30000
done

echo
echo "=== B: досбор сидов на широких dim, 12000 эпох ==="
for dim in 128 64 32; do
    for seed in 1 2; do
        run_one "$dim" "$seed" 12000
    done
done

echo
echo "Готово за $(( ($(date +%s) - START) / 60 )) минут"
echo "Сводка:"
python -u experiments/collect_results.py \
    --root results/capacity --pattern "*.json" \
    --out output/capacity_final.csv
