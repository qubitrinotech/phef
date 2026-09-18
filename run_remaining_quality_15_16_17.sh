#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$HOME/agi"
cd "$ROOT"

PYTHON_BIN=".venv/bin/python"
EVALUATOR="phef/experiments/eval_recall_compact.py"
CHECKPOINT_7="phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt"
CHECKPOINT_23="phef/results/v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt"

N_KEYS=1073741824
OVERWRITE_P=0.3
N_VALUES=64
KEY_DIM=32
DIM=128
EVAL_BATCH_SIZE=2
EVAL_BATCHES=16
CHUNK=65536
PROGRESS_EVERY=1

mkdir -p phef/logs phef/results

require_idle_gpu() {
  local active
  active="$(pgrep -af 'eval_recall_compact.py' || true)"

  if [[ -n "$active" ]]; then
    echo "ERROR: active evaluator/capacity process detected:" >&2
    echo "$active" >&2
    exit 1
  fi

  echo "=== GPU preflight $(date -Is) ==="
  nvidia-smi
}

validate_report() {
  local out="$1"

  PYTHONPATH=phef "$PYTHON_BIN" - "$out" "$CHUNK" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_chunk = int(sys.argv[2])

assert path.exists(), f"missing JSON: {path}"
report = json.loads(path.read_text())

assert report["schema"] == "compact_eval_v2_streamed_hard", report["schema"]
assert report["stream_hard_chunk_size"] == expected_chunk, report["stream_hard_chunk_size"]

for condition in ("real", "shuffled", "masked"):
    block = report[condition]
    assert block["episodes"] == 32, (condition, block["episodes"])
    for key in (
        "answer_acc",
        "n_eff_mean",
        "n_eff_min",
        "n_eff_max",
        "wall_seconds",
    ):
        assert key in block, (condition, key)

assert report["real"]["retrieval_acc"] is not None
assert report["shuffled"]["retrieval_acc"] is not None
assert report["masked"]["retrieval_acc"] is None

print("validated:", path)
print("n_eff_theory:", report["n_eff_theory"])
for condition in ("real", "shuffled", "masked"):
    print(condition, report[condition])
PY
}

run_cell() {
  local n_segments="$1"
  local checkpoint="$2"
  local eval_seed="$3"
  local tag="$4"

  local out="phef/results/${tag}.json"
  local log="phef/logs/${tag}.log"

  echo
  echo "============================================================"
  echo "START: ${tag}"
  echo "time:  $(date -Is)"
  echo "out:   ${out}"
  echo "log:   ${log}"
  echo "============================================================"

  [[ -f "$checkpoint" ]] || {
    echo "ERROR: checkpoint not found: $checkpoint" >&2
    exit 1
  }

  [[ ! -e "$out" ]] || {
    echo "ERROR: refusing to overwrite existing JSON: $out" >&2
    exit 1
  }

  [[ ! -e "$log" ]] || {
    echo "ERROR: refusing to overwrite existing log: $log" >&2
    exit 1
  }

  require_idle_gpu

  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=phef "$PYTHON_BIN" "$EVALUATOR" \
    --checkpoint "$checkpoint" \
    --n-keys "$N_KEYS" \
    --n-segments "$n_segments" \
    --overwrite-p "$OVERWRITE_P" \
    --n-values "$N_VALUES" \
    --key-dim "$KEY_DIM" \
    --dim "$DIM" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --eval-batches "$EVAL_BATCHES" \
    --seed "$eval_seed" \
    --progress-every "$PROGRESS_EVERY" \
    --stream-hard-chunk-size "$CHUNK" \
    --out "$out" \
    2>&1 | tee "$log"
  local pipeline_status=("${PIPESTATUS[@]}")
  set -e

  local evaluator_status="${pipeline_status[0]}"
  local tee_status="${pipeline_status[1]}"

  if [[ "$evaluator_status" -ne 0 ]]; then
    echo "ERROR: evaluator failed for ${tag}; exit=${evaluator_status}" >&2
    exit "$evaluator_status"
  fi

  if [[ "$tee_status" -ne 0 ]]; then
    echo "ERROR: tee failed for ${tag}; exit=${tee_status}" >&2
    exit "$tee_status"
  fi

  grep -F "saved: ${out}" "$log" >/dev/null || {
    echo "ERROR: terminal saved marker missing for ${tag}" >&2
    exit 1
  }

  validate_report "$out"

  echo "SUCCESS: ${tag} at $(date -Is)"
}

require_idle_gpu

run_cell \
  15000000 \
  "$CHECKPOINT_7" \
  11 \
  "streamed_quality_s15000000_c65536_b2x16_ckpt7_evalseed11"

run_cell \
  15000000 \
  "$CHECKPOINT_23" \
  11 \
  "streamed_quality_s15000000_c65536_b2x16_ckpt23_evalseed11"

run_cell \
  16000000 \
  "$CHECKPOINT_7" \
  7 \
  "streamed_quality_s16000000_c65536_b2x16_seed7"

run_cell \
  16000000 \
  "$CHECKPOINT_23" \
  7 \
  "streamed_quality_s16000000_c65536_b2x16_ckpt23_evalseed7"

run_cell \
  16000000 \
  "$CHECKPOINT_7" \
  11 \
  "streamed_quality_s16000000_c65536_b2x16_ckpt7_evalseed11"

run_cell \
  16000000 \
  "$CHECKPOINT_23" \
  11 \
  "streamed_quality_s16000000_c65536_b2x16_ckpt23_evalseed11"

run_cell \
  17000000 \
  "$CHECKPOINT_7" \
  7 \
  "streamed_quality_s17000000_c65536_b2x16_seed7"

run_cell \
  17000000 \
  "$CHECKPOINT_23" \
  7 \
  "streamed_quality_s17000000_c65536_b2x16_ckpt23_evalseed7"

run_cell \
  17000000 \
  "$CHECKPOINT_7" \
  11 \
  "streamed_quality_s17000000_c65536_b2x16_ckpt7_evalseed11"

run_cell \
  17000000 \
  "$CHECKPOINT_23" \
  11 \
  "streamed_quality_s17000000_c65536_b2x16_ckpt23_evalseed11"

echo
echo "============================================================"
echo "ALL REMAINING QUALITY CELLS COMPLETED SUCCESSFULLY"
echo "finished: $(date -Is)"
echo "============================================================"
