# PHEF handoff: streamed hard retrieval, validation, and next resource probe

**Date:** 15 September 2026  
**Repository:** `~/agi`  
**Branch:** `main`  
**Current implementation commit:** `662edb6 phef: stream hard retrieval in compact evaluator`  
**Hardware:** NVIDIA GeForce RTX 4090, 24 GiB VRAM  
**Purpose:** Continue PHEF capacity-frontier work safely from a verified streamed hard-retrieval implementation, without overstating what correctness smoke tests or resource probes establish.

---

## 1. Immediate state and what is committed

The new code path avoids materializing a dense `(batch_size, n_segments)` retrieval-score tensor during **hard, inference-only retrieval**. Instead, it scans memory events in chunks, retains the highest-scoring event per query, and embeds only the selected value.

The implementation is committed:

```text
662edb6 phef: stream hard retrieval in compact evaluator
```

The commit contains exactly these files:

```text
M  phef/phef/model.py
M  phef/experiments/eval_recall_compact.py
A  phef/experiments/test_chunked_hard_equivalence.py
```

Commit summary:

```text
3 files changed, 447 insertions(+), 20 deletions(-)
```

The commit deliberately does **not** include the handoff document, logs, or JSON artifacts. At the point of handoff, these paths remain untracked and should not be accidentally staged in a later code commit:

```text
?? docs/handoffs/2026-09-15_PHEF_capacity_frontier_handoff.md

?? phef/logs/cuda_parity_legacy_s4096_b2x1_seed314159.log
?? phef/logs/cuda_parity_streamed_s4096_c257_b2x1_seed314159.log
?? phef/logs/smoke_compact_v1_dense_cuda_s4096_b2x4.log
?? phef/logs/smoke_compact_v2_streamed_cpu_s17_b2x1.log
?? phef/logs/smoke_compact_v2_streamed_cuda_s4096_b2x4.log

?? phef/results/cuda_parity_legacy_s4096_b2x1_seed314159.json
?? phef/results/cuda_parity_streamed_s4096_c257_b2x1_seed314159.json
?? phef/results/smoke_compact_v1_dense_cuda_s4096_b2x4.json
?? phef/results/smoke_compact_v2_streamed_cpu_s17_b2x1.json
?? phef/results/smoke_compact_v2_streamed_cuda_s4096_b2x4.json
```

Before doing any new work, inspect the actual tree rather than relying solely on this snapshot:

```bash
cd ~/agi
git status --short
git log -1 --oneline
git show --stat --oneline HEAD
```

Expected HEAD is `662edb6` unless later work has intentionally advanced the branch.

---

## 2. Research context and terminology

PHEF is an experimental **predictive hysteresis memory** mechanism evaluated here on a controlled synthetic key–value associative-retrieval task. This work is not evidence for general-purpose natural-language memory, document memory, arbitrary lossless compression, or a comparison with external lookup/RAG/ANN systems.

The capacity-oriented question is:

> At what measured number of effective active memory entries can the selected PHEF checkpoints retrieve the latest value for a queried key, while matched shuffled and memory-masked controls remain near chance?

Keep these quantities distinct:

- `n_keys`: key-ID universe size. It affects collision statistics but is **not** the memory-capacity outcome.
- `n_segments`: number of events in an episode/memory trajectory.
- `n_eff_mean`: observed number of distinct/effective active memory entries. This is the primary capacity-axis quantity.

The task has `n_values=64`, hence the nominal answer chance rate is `1 / 64 = 0.015625`.

The evaluator performs three matched conditions, in a fixed order:

```text
real -> shuffled -> masked
```

- `real`: normal memory-based retrieval.
- `shuffled`: full memory episodes are shifted across batch elements while query/answer pairs remain fixed, breaking correct memory-query correspondence.
- `masked`: memory input is disabled.

A meaningful capacity claim requires the complete protocol and measured `n_eff_mean`; do not describe increasing `n_keys` as increasing memory capacity.

---

## 3. Existing capacity results and hardware boundary

The main evaluator is:

```text
phef/experiments/eval_recall_compact.py
```

Historically its output schema was `compact_eval_v1`. The new streamed option uses `compact_eval_v2_streamed_hard`; details are below.

### 3.1 Completed 8M milestone

A prior full-control milestone demonstrated high-accuracy exact retrieval at about **8.044M measured effective entries** on a single RTX 4090. It was replicated across two independently trained checkpoints and two evaluation RNG streams.

Fixed 8M protocol:

```text
n_keys: 1,073,741,824
n_segments: 11,534,336
overwrite_p: 0.3
n_values: 64
key_dim: 32
dim: 128
eval_batch_size: 2
eval_batches: 64
episodes: 128 per condition
progress_every: 8
n_eff_theory: 8,043,754.7
```

Checkpoints:

```text
phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt
  selected epoch: 4750
  selection accuracy: 0.9999186198

phef/results/v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt
  selected epoch: 5000
  selection accuracy: 0.9999186198
```

Cross-seed/evaluation-seed results:

| Training seed | Evaluation seed | Real correct / 128 | Real answer/retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Real `n_eff_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 125 / 128 | 0.9765625 | 0.0000000 | 0.0000000 | 0.0234375 | 8,043,761.7 |
| 23 | 7 | 126 / 128 | 0.9843750 | 0.0312500 | 0.0000000 | 0.0156250 | 8,043,761.7 |
| 7 | 11 | 126 / 128 | 0.9843750 | 0.0234375 | 0.0000000 | 0.0156250 | 8,043,833.2 |
| 23 | 11 | 127 / 128 | 0.9921875 | 0.0078125 | 0.0000000 | 0.0078125 | 8,043,833.2 |

Descriptive pooled real accuracy was `504 / 512 = 0.984375`; treat that pooled figure descriptively, not as a precise population estimate.

Representative old dense-path resource profile at 8M:

```text
worst cumulative peak GPU allocation after shuffled: about 20.31 GiB
physical GPU memory: 24,564 MiB
peak process RSS: about 10.07 GiB
full real + shuffled + masked wall time: about 25–26 minutes
```

The 8M configuration was already near the practical dense-path boundary at `eval_batch_size=2`. Do not linearly extrapolate it to 16M.

### 3.2 Completed 6.98M pilot

A full-control pilot at `n_segments=10,000,000` measured `n_eff_mean≈6,977,123` in real condition, with real answer/retrieval `125/128 = 0.9765625`, shuffled retrieval `0/128`, and masked answer `1/128`.

Its resource profile was approximately:

```text
real peak GPU allocation: 15.51 GiB
worst cumulative peak after shuffled: 17.62 GiB
peak RSS: 8.86 GiB
full protocol wall time: about 21.9 minutes
```

This is one intermediate full-control point. It contributes a data point but does not by itself define a capacity curve or establish a meaningful performance difference from the 8M replication.

### 3.3 9M dense resource boundary

A previous handoff records that a dense path resource probe near 9M encountered a GPU memory boundary. The relevant handoff is:

```text
docs/handoffs/2026-09-15_PHEF_capacity_frontier_handoff.md
```

Read it before making claims or choosing a risky depth. The reason for the current work is to reduce hard-evaluation memory pressure, not to claim in advance that every depth is now feasible.

---

## 4. Why streamed hard retrieval was added

In hard evaluation, one memory event is selected per batch row. The old dense path nevertheless creates dense retrieval scores over all events. At millions of segments, the `(batch_size, n_segments)` score tensor becomes a major VRAM cost.

A previous hard-evaluation optimization already avoided materializing all value embeddings. Instead of:

```python
memory_values = self.value_emb.weight.to(query_state.dtype)[values]
```

which has eventwise shape `(batch_size, n_segments, dim)`, the hard path selects an index first and embeds only selected value IDs:

```python
index = scores.argmax(dim=1)
selected_values = values[
    torch.arange(batch_size, device=values.device),
    index,
]
memory_state = self.value_emb(selected_values).to(query_state.dtype)
```

That earlier change preserved output behavior in a same-seed 8.39M probe while reducing observed peak allocation from about 17.46 GiB to 14.72 GiB for that probe.

The new streamed path removes the large dense score materialization itself for inference-only hard retrieval. It scans `n_segments` in chunks of `stream_hard_chunk_size` and retains the best local/global candidate.

This reduces memory pressure but does **not** change the scientific task, checkpoint, controls, or metric definitions.

---

## 5. Implementation details in commit 662edb6

### 5.1 Model: `phef/phef/model.py`

New helper:

```python
PHEF._streamed_hard_retrieval(...)
```

Conceptual behavior:

1. For each batch row, determine its source row according to the identity permutation (`real`) or rolled batch permutation (`shuffled`).
2. Materialize key embeddings for that source row’s compact codebook.
3. Process event positions in chunks.
4. Compute chunk scores using the same semantic and recency expression as the dense path:

```python
semantic = (memory_keys[event_slots] @ query_state[b]) / math.sqrt(self.dim)
chunk_scores = (
    self.score_scale * semantic
    + self.recency_scale * event_times.float() / max(1, self.n_segments - 1)
)
```

5. Find each chunk’s local maximum and update the global candidate only for a **strictly greater** score.

The strict comparison is essential:

```python
if best_score is None or local_score > best_score:
```

It preserves `torch.argmax` tie semantics: if multiple positions have the same maximum score, choose the **earliest global index**. A later equal-scoring candidate must not replace an earlier one, including when the two candidates lie in different chunks.

The helper returns:

```text
retrieval indices: LongTensor shaped (batch_size,)
selected values: tensor of selected value IDs
```

The streamed forward path embeds only the selected values and returns:

```python
(logits, index)
```

rather than dense retrieval scores.

### 5.2 API constraints and guards

The streamed path is intentionally narrow. `PHEF.forward()` supports:

```text
stream_hard_chunk_size=0                  # default: legacy behavior
return_retrieval_indices=False            # default
```

When `stream_hard_chunk_size > 0`, the implementation rejects unsafe or ambiguous use:

- negative chunk size;
- `hard=False`;
- no `return_retrieval_indices=True`;
- simultaneous `return_retrieval=True` and `return_retrieval_indices=True`;
- `model.training == True`;
- gradients enabled (`torch.is_grad_enabled()`);
- memory disabled (`not self.use_memory`);
- `mask_memory=True`;
- `return_retrieval_indices=True` without streaming enabled.

This is by design: streamed retrieval is evaluation-only/inference-only and does not silently substitute itself into training or gradient-bearing code.

### 5.3 Evaluator: `phef/experiments/eval_recall_compact.py`

New CLI option:

```text
--stream-hard-chunk-size INTEGER
```

Rules:

- default is `0`, which remains the legacy dense behavior;
- values below zero terminate with an error;
- a positive value enables streamed hard retrieval for `real` and `shuffled` conditions;
- `masked` remains on the regular masked/dense branch, because streamed retrieval requires enabled, unmasked memory.

Schemas:

```text
legacy dense path:        compact_eval_v1
positive chunk-size path: compact_eval_v2_streamed_hard
```

The report includes:

```text
schema
stream_hard_chunk_size
checkpoint / selected metadata
config
n_eff_theory
real / shuffled / masked blocks
```

Each condition contains:

```text
episodes
n_eff_mean
n_eff_min
n_eff_max
answer_acc
retrieval_acc
wall_seconds
```

In `masked`, `retrieval_acc` is `null` in JSON / `None` after Python JSON loading, since retrieval is not scored for masked memory.

The evaluator prints GPU allocation peak cumulatively for the process. It does **not** call `torch.cuda.reset_peak_memory_stats()` between conditions, so interpret the largest printed allocation as a process-level worst peak, not as independent per-condition peaks.

### 5.4 Regression script

New file:

```text
phef/experiments/test_chunked_hard_equivalence.py
```

It checks:

- dense vs streamed retrieval decisions across random CPU cases;
- both non-shuffled and shuffled paths;
- a crafted tie case, including a tie crossing chunk boundaries;
- API guards;
- diagnostics if a mismatch occurs, including batch permutation, dense and streamed selected indices, and dense top-2 candidates/scores.

Successful output observed after final correction:

```text
cpu chunked-hard equivalence matrix: PASS random_cases=8 tie_case=PASS api_guards=PASS
```

---

## 6. Validation completed after the final code change

The following validation was run after the final committed implementation was in place.

### 6.1 Syntax and diff hygiene

Completed successfully:

```bash
python -m py_compile \
  phef/phef/model.py \
  phef/experiments/eval_recall_compact.py \
  phef/experiments/test_chunked_hard_equivalence.py

git diff --check
```

### 6.2 CPU regression matrix

Command:

```bash
cd ~/agi
PYTHONPATH=phef .venv/bin/python phef/experiments/test_chunked_hard_equivalence.py
```

Observed output:

```text
cpu chunked-hard equivalence matrix: PASS random_cases=8 tie_case=PASS api_guards=PASS
```

This validates CPU-level retrieval-index equivalence, including earliest-index ties and the intended API restrictions.

### 6.3 CUDA evaluator parity smoke

Two actual CUDA evaluator runs were completed with the same checkpoint, world parameters and evaluation seed. The GPU was confirmed idle before launch; only the desktop process held a negligible amount of memory.

Common settings:

```text
checkpoint: phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt
checkpoint epoch: 4750
checkpoint selection accuracy: 0.999919
n_keys: 262144
n_segments: 4096
overwrite_p: 0.3
n_values: 64
key_dim: 32
dim: 128
eval_batch_size: 2
eval_batches: 1
episodes: 2 per condition
seed: 314159
progress_every: 1
device: cuda
```

Legacy output artifact:

```text
phef/results/cuda_parity_legacy_s4096_b2x1_seed314159.json
phef/logs/cuda_parity_legacy_s4096_b2x1_seed314159.log
```

Streamed output artifact:

```text
phef/results/cuda_parity_streamed_s4096_c257_b2x1_seed314159.json
phef/logs/cuda_parity_streamed_s4096_c257_b2x1_seed314159.log
```

The streamed run used:

```text
--stream-hard-chunk-size 257
```

Both runs reached a terminal `saved:` line and wrote JSON.

Observed equal condition metrics:

| Condition | Episodes | `n_eff_mean` | `n_eff_min`–`n_eff_max` | `answer_acc` | `retrieval_acc` |
|---|---:|---:|---:|---:|---:|
| `real` | 2 | 2846.0 | 2841–2851 | 1.0 | 1.0 |
| `shuffled` | 2 | 2828.5 | 2819–2838 | 0.0 | 0.0 |
| `masked` | 2 | 2836.0 | 2818–2854 | 0.0 | `None` |

Expected differences:

```text
legacy schema:   compact_eval_v1
streamed schema: compact_eval_v2_streamed_hard
legacy chunk size: 0
streamed chunk size: 257
config.out differs
wall_seconds may differ
```

A strict report-contract comparator passed after excluding precisely those allowed differences:

```text
cuda evaluator parity: PASS
schemas: compact_eval_v1 vs compact_eval_v2_streamed_hard
allowed differences: schema, stream_hard_chunk_size, config.out, wall_seconds
```

This validates evaluator-level behavior for all three matched control conditions under CUDA, not merely direct helper output.

### 6.4 Scope of validation

What is established:

1. The streamed hard path matches dense hard retrieval in the CPU matrix, including tie behavior.
2. The API guards correctly prohibit unsupported modes.
3. The compact evaluator routes real/shuffled streamed runs and masked dense/masked runs as intended.
4. Under the CUDA smoke configuration, legacy and streamed reports match on all meaningful metric and configuration fields.

What is **not** established:

1. A two-episode smoke is not a capacity estimate.
2. It does not prove resource feasibility at 9M, 16M, or any particular scale.
3. It does not demonstrate a performance improvement at a target depth; that requires a measured resource probe.
4. It does not replace full-control, multi-episode, cross-seed scientific evaluations.

---

## 7. Operating rules for CUDA work

There is one RTX 4090 with 24 GiB VRAM. Before **every** CUDA-heavy run, inspect state:

```bash
cd ~/agi
git status --short
pgrep -af 'eval_recall_compact.py|capacity' || true
nvidia-smi
```

Rules:

- Run only one CUDA-heavy evaluator/capacity job at a time.
- Do not launch if another relevant job is present.
- Do not let two jobs write to the same JSON or log path.
- Use unique output names.
- Store stdout/stderr logs in `phef/logs/` and JSON reports in `phef/results/`.
- A run is complete only if its log reaches terminal `saved: ...` and the JSON exists, parses, and contains complete `real`, `shuffled`, and `masked` blocks.
- Do not rerun automatically after failure while a process may still exist.
- Do not stage stale PID files if any are created locally.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` may print allocator mapping/OOM warnings near capacity. Treat them as nonfatal only if progress continues, resource use remains stable, all three conditions complete, and the JSON is written. Otherwise treat them as a failure signal.

Do not begin a speculative giant run. The immediate scientific/engineering objective is a **medium-depth full-control resource probe**, not an unsupported capacity claim.

---

## 8. Recommended next action: medium-depth resource probe

### 8.1 Goal

Measure whether the committed streamed hard-retrieval path changes the operational GPU/RSS boundary at a meaningful depth, while preserving the full `real -> shuffled -> masked` protocol.

This is a resource/semantics probe, not a final capacity evaluation. Use:

```text
eval_batch_size: 2
eval_batches: 1
episodes: 2 per condition
```

This keeps the mandatory batch size valid for shuffled mode and limits exposure if the allocation remains too large.

### 8.2 Choose depth conservatively

The next depth should be well above the 4,096-segment smoke and informative relative to the former dense-path boundary, but should not leap blindly to 16M.

A practical candidate is the former 9M-scale neighborhood, using a single-batch full-control probe. However, inspect the existing handoff and prior result filenames before choosing exact `n_segments`; avoid duplicating or overwriting an older probe. If a 9M dense resource probe existed, use a unique name and clearly label streamed mode.

The output must be interpreted as an operational feasibility observation, not evidence of capacity quality from two episodes.

### 8.3 Preflight

Run and inspect:

```bash
cd ~/agi
git status --short
git log -1 --oneline
pgrep -af 'eval_recall_compact.py|capacity' || true
nvidia-smi
mkdir -p phef/logs phef/results
```

Confirm:

- no active evaluator/capacity job;
- the GPU is not materially occupied;
- `HEAD` is `662edb6` unless deliberately advanced;
- chosen output paths do not already exist.

### 8.4 Probe template

First choose an exact segment count and unique `TAG`. Example variables only; do not blindly substitute an unreviewed value:

```bash
cd ~/agi

TAG='streamed_resource_probe_REPLACE_ME'
NSEGMENTS='REPLACE_ME'
CHUNK='65536'
OUT="phef/results/${TAG}.json"
LOG="phef/logs/${TAG}.log"

[ ! -e "$OUT" ] || { echo "Refusing to overwrite $OUT"; exit 1; }
[ ! -e "$LOG" ] || { echo "Refusing to overwrite $LOG"; exit 1; }

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=phef .venv/bin/python phef/experiments/eval_recall_compact.py \
  --checkpoint phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt \
  --n-keys 1073741824 \
  --n-segments "$NSEGMENTS" \
  --overwrite-p 0.3 \
  --n-values 64 \
  --key-dim 32 \
  --dim 128 \
  --eval-batch-size 2 \
  --eval-batches 1 \
  --seed 7 \
  --progress-every 1 \
  --stream-hard-chunk-size "$CHUNK" \
  --out "$OUT" \
  2>&1 | tee "$LOG"
```

A chunk size around `65536` is a reasonable starting resource/profiling choice, but it is not a correctness-sensitive value: the CPU test covers multiple chunk patterns and CUDA smoke passed at 257. Keep the chunk size explicit in the artifact filename or report metadata.

### 8.5 Post-run validation

Only after the process exits, validate that the run completed and inspect report values:

```bash
cd ~/agi

TAG='streamed_resource_probe_REPLACE_ME'
OUT="phef/results/${TAG}.json"
LOG="phef/logs/${TAG}.log"

grep -F "saved:" "$LOG"

PYTHONPATH=phef .venv/bin/python - <<PY
import json
from pathlib import Path

path = Path("$OUT")
report = json.loads(path.read_text())

assert report["schema"] == "compact_eval_v2_streamed_hard"
assert report["stream_hard_chunk_size"] > 0
for condition in ("real", "shuffled", "masked"):
    block = report[condition]
    assert block["episodes"] == 2, (condition, block["episodes"])
    for key in ("n_eff_mean", "n_eff_min", "n_eff_max", "answer_acc"):
        assert key in block, (condition, key)

print("schema:", report["schema"])
print("chunk:", report["stream_hard_chunk_size"])
print("n_eff_theory:", report["n_eff_theory"])
for condition in ("real", "shuffled", "masked"):
    print(condition, report[condition])
PY
```

Also preserve/record from the log:

- all `real`, `shuffled`, and `masked` progress/final lines;
- wall time per condition;
- peak RSS;
- cumulative `gpu_alloc_peak` after each condition;
- any allocator warnings and whether progress continued after them;
- terminal `saved:` line.

Do not call the probe successful if the JSON is absent, invalid, missing a condition, or the log lacks `saved:`.

### 8.6 Interpretation of the medium probe

A completed probe establishes only that this exact checkpoint, environment, segment depth, batch size, chunk size, and one-batch control protocol ran successfully with recorded resource use.

It can support statements such as:

> At this configuration, the streamed hard evaluator completed a two-episode-per-condition resource probe, with observed cumulative allocator peak X GiB and recorded full-control outputs.

It cannot support statements such as:

- “PHEF capacity is N million.”
- “The model has been statistically validated at N million.”
- “The new path makes 16M feasible.”
- “A lack of OOM in one probe demonstrates a general scaling law.”

If the medium probe succeeds with sufficient memory headroom, the next research step is to select a depth and protocol for a larger full-control evaluation. If it fails, record the failure boundary, allocator behavior, exact settings, and whether failure occurred before/within a condition; do not immediately retry with an uncontrolled variation.

---

## 9. Useful commands

### Re-run CPU regression

```bash
cd ~/agi
PYTHONPATH=phef .venv/bin/python phef/experiments/test_chunked_hard_equivalence.py
```

Expected successful output:

```text
cpu chunked-hard equivalence matrix: PASS random_cases=8 tie_case=PASS api_guards=PASS
```

### Inspect committed changes

```bash
cd ~/agi
git show --stat --oneline 662edb6
git show -- phef/phef/model.py \
  phef/experiments/eval_recall_compact.py \
  phef/experiments/test_chunked_hard_equivalence.py
```

### Check evaluator help

```bash
cd ~/agi
PYTHONPATH=phef .venv/bin/python phef/experiments/eval_recall_compact.py --help
```

### Validate an existing JSON is complete

```bash
cd ~/agi
PYTHONPATH=phef .venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path('REPLACE_WITH_REPORT.json')
report = json.loads(path.read_text())
print('schema:', report['schema'])
print('stream_hard_chunk_size:', report['stream_hard_chunk_size'])
for condition in ('real', 'shuffled', 'masked'):
    print(condition, report[condition])
PY
```

---

## 10. Claims discipline

Supported by current evidence:

1. The implementation at `662edb6` has a hard, chunk-streamed inference path designed to avoid dense retrieval-score materialization.
2. CPU random cases, tie behavior, and API guards passed.
3. CUDA compact-evaluator parity against the legacy dense path passed under the documented smoke configuration and full three-condition protocol.
4. The 8M milestone previously demonstrated high exact-retrieval accuracy on this controlled task at measured `n_eff≈8.044M`, across two checkpoints and two evaluation seeds.
5. The new path is ready for a measured medium-depth resource probe.

Not supported yet:

1. A new capacity frontier above the prior 8M result.
2. Feasibility of 9M/16M/full multi-batch evaluation on the current GPU.
3. A quantitative VRAM saving at a target millions-of-segments configuration.
4. Any general-memory, language-memory, or real-world retrieval claim.

The next chat should begin by checking `git status`, inspecting the previous capacity-frontier handoff if necessary, and selecting a single conservative medium-depth resource probe with unique artifacts.
