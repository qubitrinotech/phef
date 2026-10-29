# PHEF capacity-frontier handoff: 8M replication, 6.98M pilot, and 9M resource boundary

**Date:** 15 September 2026  
**Repository:** `~/agi`  
**Hardware:** NVIDIA GeForce RTX 4090, 24 GiB VRAM  
**Current branch:** `main`  
**Last known HEAD:** `d0d0173 phef: record 9M resource probe boundary`  
**Purpose:** continue rigorous capacity-frontier work for PHEF without overstating current synthetic results or risking concurrent CUDA jobs.

## 1. Operating rules

This repository runs CUDA-heavy PHEF evaluations on one RTX 4090 with 24 GiB VRAM.

Before **every** new CUDA-heavy run, execute and inspect:

```bash
cd ~/agi
git status --short
pgrep -af 'eval_recall_compact.py|capacity' || true
nvidia-smi
```

Rules:

- Run **only one** CUDA-heavy evaluation at a time.
- Do not launch a run while another PHEF evaluator or capacity job is active.
- Do not let two processes write to the same `--out` JSON or log path.
- Save stdout/stderr logs in `phef/logs/` and JSON reports in `phef/results/`.
- Do not call a run complete until the log has a terminal `saved: ...` line and the JSON is valid and has complete `real`, `shuffled`, and `masked` blocks.
- Use unique artifact names; do not overwrite old results absent an explicit reason.
- Do not commit stale `.pid` files. If a PID file is used locally, confirm the process has ended and remove it before staging.
- After a meaningful result, validate artifacts, stage only intended files, inspect staged diff, then commit.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can produce repeated `expandable_segments: memory mapping failed with OOM ...` warnings near capacity. In completed prior runs these warnings were nonfatal only when progress continued, resource use remained stable, all three conditions completed, and the JSON was written. Do not assume warnings are harmless without those conditions.

Do not start speculative giant runs. First perform a resource probe with the full control protocol.

## 2. Scientific objective and terminology

PHEF is an experimental **predictive hysteresis memory** mechanism evaluated here on a controlled synthetic key–value associative-retrieval task.

Long-term objective: map the actual capacity frontier of the learned memory mechanism at millions of active memory entries, without substituting a key-universe parameter for memory capacity.

Keep these quantities distinct:

- `n_keys`: universe of possible key IDs. It affects collision statistics but is **not** the capacity result.
- `n_segments`: number of sequence/memory-trajectory events.
- `n_eff_mean`: directly observed number of distinct active/effective memory entries; this is the primary x-axis capacity metric.

Task:

- An episode contains a sequence of key/value events.
- The evaluator chooses a query key from the episode.
- The model must return the value associated with the most recent matching key.
- `n_values=64`, so answer chance is `1/64 = 0.015625`.
- Primary metric: exact `retrieval_acc`.
- `answer_acc` is also reported.

Matched conditions:

- `real`: normal memory-based retrieval.
- `shuffled`: complete memory episodes are shifted across batch elements while query/answer pairs remain in place, breaking memory-query correspondence.
- `masked`: memory input is disabled.

A valid capacity claim requires measured `n_eff_mean` and the full matched `real`, `shuffled`, `masked` protocol. Do not characterize an increase in `n_keys` as capacity growth.

## 3. Evaluator and model facts

Main evaluator:

```text
phef/experiments/eval_recall_compact.py
schema: compact_eval_v1
```

Important evaluator behavior verified from source:

- `HARD_EVAL = True` is a module constant. There is no CLI `--hard-eval` flag.
- CLI uses `--seed` for evaluation RNG, not `--eval-seed`.
- `--eval-batch-size` must be at least 2. Current `PHEF.forward(shuffle=True)` uses `torch.randint(1, batch_size, ...)`, invalid for singleton batches.
- The evaluator always runs conditions in this order:

```text
real -> shuffled -> masked
```

- A single `CompactBranchingWorld` is used across the conditions; each receives newly sampled episodes from its continuing RNG stream.
- `eval_batches=1` with `eval_batch_size=2` gives 2 episodes per condition. This is a resource/semantics probe, not a statistical capacity evaluation.
- The evaluator writes JSON only at the end. A missing `saved:` means no completed evaluation report.
- The evaluator does **not** call `torch.cuda.reset_peak_memory_stats()` between conditions. Printed `gpu_alloc_peak` is cumulative for the current process and thus should be treated as a worst-case process peak, not a per-condition independent peak.

The compact evaluator differs from the older full-codebook evaluator because it materializes codebook rows only for slots occurring in each episode. `n_keys` remains meaningful in the slot collision process. RNG streams are not bit-identical to the old evaluator, so equivalence is distributional rather than bit-identical.

### Hard-evaluation memory optimization

`phef/phef/model.py` was optimized in the `hard=True` branch.

Old code materialized embeddings for every event:

```python
memory_values = self.value_emb.weight.to(query_state.dtype)[values]
```

That tensor has shape `(batch_size, n_segments, dim)`, even though hard retrieval selects just one event per batch item.

The revised hard branch selects the argmax first and embeds only selected value IDs:

```python
index = scores.argmax(dim=1)
selected_values = values[
    torch.arange(batch_size, device=values.device),
    index,
]
memory_state = self.value_emb(selected_values).to(query_state.dtype)
```

Soft mode remains unchanged. The optimization was smoke-tested and checked in an identical-seed 8.39M-segment probe. Metrics were unchanged while observed peak allocation dropped from 17.46 GiB to 14.72 GiB at that probe. The hard-branch equivalence is:

```text
value_emb.weight[values][row, index]
== value_emb(values[row, index])
```

## 4. Completed 8M milestone

### Research question

Can selected PHEF checkpoints retrieve the correct value from episodes containing roughly 8 million **measured effective memory entries**, while matched shuffled and masked controls remain at chance-level behavior?

### Fixed 8M protocol

```text
script: phef/experiments/eval_recall_compact.py
schema: compact_eval_v1
hard_eval: true

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

The observed real-condition `n_eff_mean` range was 8,043,761.7 to 8,043,833.2. Thus the result concerns roughly **8.044M effective active entries**, not merely a large key universe.

### Checkpoints

Both checkpoints used `n_train=32768` and v2 checkpoint selection with a separate selection world:

```text
selection recipe:
sel_n_keys: 262144
sel_acc_n: 4096
sel_window: 3
```

| Training seed | Checkpoint | Selected epoch | Selection accuracy |
|---:|---|---:|---:|
| 7 | `phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt` | 4750 | 0.9999186198 |
| 23 | `phef/results/v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt` | 5000 | 0.9999186198 |

The suffix `_e6000_best.pt` is an output filename. It does not mean the selected state necessarily came from epoch 6000.

### Cross-seed results

| Training seed | Evaluation seed | Selected epoch | Real correct / 128 | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Real `n_eff_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 4750 | 125 / 128 | 0.9765625 | 0.9765625 | 0.0000000 | 0.0000000 | 0.0234375 | 8,043,761.7 |
| 23 | 7 | 5000 | 126 / 128 | 0.9843750 | 0.9843750 | 0.0312500 | 0.0000000 | 0.0156250 | 8,043,761.7 |
| 7 | 11 | 4750 | 126 / 128 | 0.9843750 | 0.9843750 | 0.0234375 | 0.0000000 | 0.0156250 | 8,043,833.2 |
| 23 | 11 | 5000 | 127 / 128 | 0.9921875 | 0.9921875 | 0.0078125 | 0.0000000 | 0.0078125 | 8,043,833.2 |

Descriptive aggregate:

```text
real correct retrievals: 504 / 512
pooled descriptive real accuracy: 0.984375 = 98.4375%
per-cell real accuracy range: 0.9765625 to 0.9921875
```

This pooled value is descriptive, not a precise population estimate: observations are grouped into four cells, using two training checkpoints and two evaluation RNG streams.

### Interpretation

Supported:

1. PHEF performs high-accuracy exact retrieval in this controlled synthetic task at measured `n_eff≈8.044M`.
2. The result replicated across two independently trained checkpoints and two independent evaluation RNG streams.
3. High real-condition performance depends on correct memory-query correspondence under the tested shuffled and masked controls.
4. After the hard-eval optimization, the complete three-condition 8M protocol is operationally feasible on a single RTX 4090.

Why the controls matter:

- Shuffled retrieval was `0/128` in all four runs. This supports that real success depends on matching each query to its associated memory episode.
- Shuffled answer ranged from `1/128` to `4/128`, compatible with sampling around 64-class chance.
- Masked answer ranged from `1/128` to `3/128`, also compatible with chance.
- In all four real cells, `answer_acc == retrieval_acc`; the answer head did not compensate for incorrect retrieval.

Not supported:

1. General-purpose natural-language or document long-term memory.
2. Lossless compression of arbitrary text into a fixed neural state at this scale.
3. A precise population error-rate estimate beyond this 2×2, 128-episodes-per-cell design.
4. That 8M is a maximum capacity.
5. That 16M is feasible with the present evaluator and hardware.
6. A direct comparison with exact lookup, ANN retrieval, RAG, or external-memory systems.

### 8M resource profile

Representative completed full runs:

| Metric | Observed value |
|---|---:|
| Peak GPU allocation after shuffled | approximately 20.31 GiB |
| Physical GPU memory | 24,564 MiB |
| Peak process RSS | approximately 10.07 GiB |
| Real wall time per 128 episodes | approximately 525–536 s |
| Shuffled wall time per 128 episodes | approximately 510–528 s |
| Masked wall time per 128 episodes | approximately 503–508 s |
| Full three-condition wall time | approximately 25–26 min |

The 8M configuration is near capacity at `eval_batch_size=2`. Do not extrapolate linearly to 16M.

### 8M artifacts and commits

Primary handoff note:

```text
docs/handoffs/2026-09-14_PHEF_8M_cross_seed_replication.md
```

Relevant commits in chronological order:

```text
80c2de4 phef: validate 8M effective-memory retrieval on seed 7
4c57db4 phef: replicate 8M effective-memory retrieval on seed 23
edd96d6 phef: validate 8M retrieval on eval seed 11
fe66587 phef: complete 8M retrieval cross-seed replication
344179f docs: record PHEF 8M cross-seed replication
```

Result artifacts:

```text
phef/results/capacity8m_pilot_k32768_seed7_b2x64.json
phef/logs/capacity8m_pilot_k32768_seed7_b2x64.log

phef/results/capacity8m_pilot_k32768_seed23_b2x64.json
phef/logs/capacity8m_pilot_k32768_seed23_b2x64.log

phef/results/capacity8m_pilot_k32768_seed7_evalseed11_b2x64.json
phef/logs/capacity8m_pilot_k32768_seed7_evalseed11_b2x64.log

phef/results/capacity8m_pilot_k32768_seed23_evalseed11_b2x64.json
phef/logs/capacity8m_pilot_k32768_seed23_evalseed11_b2x64.log
```

Supporting evaluator-optimization artifacts:

```text
phef/results/smoke_hard_value_gather_s65536_b2x1.json
phef/results/probe_hard_value_gather_k32768_s8388608_b2x1.json
phef/results/probe_hard_value_gather_k32768_s11534336_b2x1.json
```

## 5. New 6.98M full-control capacity-curve pilot

### Commit

```text
8b86fe2 phef: add 6.98M capacity-curve pilot
```

### Protocol

```text
checkpoint: phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt
training seed: 7
selected epoch: 4750
evaluation seed: 19

n_keys: 1,073,741,824
n_segments: 10,000,000
overwrite_p: 0.3
n_values: 64
key_dim: 32
dim: 128
hard_eval: true
eval_batch_size: 2
eval_batches: 64
episodes: 128 per condition
progress_every: 8
```

### Results

| Condition | Episodes | `n_eff_mean` | Answer accuracy | Retrieval accuracy | Wall time |
|---|---:|---:|---:|---:|---:|
| Real | 128 | 6,977,123.203125 | 0.9765625 (125/128) | 0.9765625 (125/128) | 439.4885 s |
| Shuffled | 128 | 6,977,252.4609375 | 0.0078125 (1/128) | 0.0000000 (0/128) | 437.8425 s |
| Masked | 128 | 6,977,331.625 | 0.0078125 (1/128) | not scored | 437.3247 s |

```text
n_eff_theory: 6,977,232.103631973
real n_eff_min/max: 6,973,189 / 6,980,504
```

The real answer and retrieval accuracies were exactly equal. Shuffled retrieval was `0/128`; shuffled and masked answer accuracy were each `1/128`, adjacent to the 64-class chance baseline.

### Resource profile

```text
real peak GPU allocation: 15.51 GiB
worst cumulative GPU allocation peak after shuffled: 17.62 GiB
peak RSS: 8.86 GiB
full real + shuffled + masked wall time: about 21.9 min
per-batch runtime: approximately 6.83–6.87 s
```

### Artifacts

```text
phef/results/capacitycurve_probe_nseg10000000_seed7_evalseed19_b2x1.json
phef/logs/capacitycurve_probe_nseg10000000_seed7_evalseed19_b2x1.log

phef/results/capacitycurve_nseg10000000_seed7_evalseed19_b2x64.json
phef/logs/capacitycurve_nseg10000000_seed7_evalseed19_b2x64.log
```

### Interpretation

This is one completed, full-control intermediate point at measured `n_eff≈6.977M`. It establishes operational feasibility at that depth and contributes one point for a future capacity curve.

It does **not** by itself estimate a curve shape, establish a difference from 8.044M, or replace the 8M 2×2 cross-seed replication. The observed 97.65625% from one 128-episode cell should not be compared as a meaningful decrease or increase against the 98.4375% pooled 8M aggregate without matched additional cells and uncertainty analysis.

## 6. New 9M resource-boundary probe

### Commit

```text
d0d0173 phef: record 9M resource probe boundary
```

### Protocol

```text
checkpoint: phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt
training seed: 7
selected epoch: 4750
evaluation seed: 19

n_keys: 1,073,741,824
n_segments: 12,900,000
overwrite_p: 0.3
n_values: 64
key_dim: 32
dim: 128
hard_eval: true
eval_batch_size: 2
eval_batches: 1
episodes: 2 per condition
progress_every: 1
```

### Observed values

| Condition | Episodes | `n_eff_mean` | Answer accuracy | Retrieval accuracy | Cumulative GPU allocation peak |
|---|---:|---:|---:|---:|---:|
| Real | 2 | 8,990,361.0 | 1.000000 | 1.000000 | 19.88 GiB |
| Shuffled | 2 | 8,991,882.5 | 0.000000 | 0.000000 | **22.60 GiB** |
| Masked | 2 | 8,991,542.0 | 0.000000 | not scored | 22.60 GiB |

```text
n_eff_theory: 8,992,135.782106042
real/shuffled/masked wall time: 9.0 s / 8.8 s / 9.0 s
peak RSS by masked completion: 8.60 GiB
```

The run completed `real`, `shuffled`, and `masked`, and wrote a complete valid JSON.

### Artifacts

```text
phef/results/capacitycurve_probe_nseg12900000_seed7_evalseed19_b2x1.json
phef/logs/capacitycurve_probe_nseg12900000_seed7_evalseed19_b2x1.log
```

### Critical interpretation

This probe establishes only that the current evaluator completes **one batch** of the complete control protocol at measured `n_eff≈8.99M` on the RTX 4090.

It is **not** a 9M capacity result:

- Two episodes per condition cannot estimate retrieval accuracy.
- The shuffle path reaches 22.60 GiB cumulative allocation, leaving a narrow margin on a 24 GiB card.
- The successful short probe does not justify a `eval_batches=64` full run. A long run risks late OOM, allocator instability, partial artifacts, or poor reproducibility.

Decision already made: **do not launch a full 9M 128-episode evaluation with the current evaluator/memory profile.**

## 7. Current evidence summary

| Measured effective depth | Design | Result | Status |
|---:|---|---|---|
| approximately 5.855M | Existing probe at `n_segments=8,388,608` | Resource/supporting probe only | Not a comparable full pilot claim |
| **6.977M** | One full-control cell: train seed 7, eval seed 19, 128 episodes/condition | Real retrieval 125/128 = 97.65625%; shuffled retrieval 0/128 | Completed intermediate pilot |
| **8.044M** | 2 training seeds × 2 evaluation seeds; 512 real episodes total | 504/512 = 98.4375% descriptive pooled exact real retrieval; shuffled retrieval 0/128 per cell | Completed reproducible milestone |
| **8.99M** | Full-control one-batch probe: 2 episodes/condition | Completed technically; shuffled cumulative peak 22.60 GiB | Resource boundary, not performance evidence |

Do not infer that retrieval improves with depth because the 8M descriptive aggregate exceeds the single 6.98M cell. The designs differ and the sample sizes are limited.

## 8. Recommended next work

### Immediate priority: engineer a memory-safer evaluator/control path

The next task should be **design and code inspection**, not a new giant CUDA run.

Goal: determine why the current full protocol reaches 22.60 GiB at 8.99M and make a scientifically equivalent evaluation protocol with a stronger margin before attempting a 9M pilot or any 16M work.

Permissible directions:

1. **Profile allocation lifetimes and condition isolation.**
   - Inspect tensors retained across forward/evaluation iterations and conditions.
   - Determine whether evaluation conditions can run in isolated subprocesses without changing per-condition semantics.
   - Running each condition in a separate subprocess may reduce allocator/history effects, but is only useful if it does not change the data-generating or control semantics and if outputs are merged transparently.

2. **Memory-efficient equivalent score computation.**
   - Investigate streamed/chunked computation only if it can be proved equivalent to current `PHEF.forward` hard-mode output, including tie/argmax behavior and retrieval metric semantics.
   - Add small-scale deterministic regression tests before relying on it at large scale.

3. **Semantically validated singleton-compatible shuffled control.**
   - The current in-model shuffle requires `batch_size>=2`.
   - A singleton control must be designed to break memory-query correspondence with the same causal meaning, then validated against current shuffled behavior at batch size 2 and small/medium depths.
   - Do not simply omit shuffled control or claim a changed control is equivalent without validation.

4. **Hardware escalation.**
   - More VRAM can be valid, but any larger-scale claim must retain measured `n_eff_mean` and full controls.

Non-negotiable constraints:

- Do not call a larger `n_keys` setting a capacity increase.
- Do not remove, weaken, or silently change negative controls.
- Do not compare a revised protocol to old results as though it is identical without explicit equivalence tests.
- Do not claim 16M from a theoretical target; report measured `n_eff_mean`.
- Do not attempt direct 16M `b=2` current-evaluator runs on this GPU.

### After redesign: validation ladder

Use this order before a large run:

1. Read and understand relevant `PHEF.forward` and evaluator code.
2. Formulate exact semantics that the optimization/control redesign must preserve.
3. Implement the smallest scoped change.
4. Run small deterministic or fixed-seed regression/equivalence checks against the current implementation.
5. Run medium-depth resource probes with full controls.
6. Verify JSON schema and metrics parity.
7. Run a 9M one-batch full-control probe again.
8. Only if memory margin is robust, run one 9M full 128-episode pilot.
9. If that completes, replicate across checkpoint and evaluation seeds before making frontier claims.

### Secondary priority: strengthen statistics at 8M

If the engineering path is delayed, strengthen the established 8M claim using new evaluation seeds (for example, 19 and 29), both training checkpoints, and the unchanged full three-condition protocol. Begin with 128 episodes per new matrix cell, then decide whether 512 or 1024 episodes per cell is justified.

This has low methodological risk but does not advance the frontier.

### Longer term: a meaningful memory benchmark

Current results are about synthetic exact associative retrieval. A practical extension should include:

- stored source records and provenance;
- updates, overwrites, and conflicts;
- semantic/approximate queries;
- distractors and noisy retrieval;
- multi-hop retrieval;
- separation of retrieval accuracy from answer-generation quality;
- comparisons against exact lookup, ANN retrieval, and RAG/external-memory baselines.

## 9. Useful commands

### Inspect evaluator and model before modifying code

```bash
cd ~/agi
sed -n '1,520p' phef/experiments/eval_recall_compact.py
sed -n '1,360p' phef/phef/model.py
grep -RIn --exclude-dir=.git \
  -E 'def forward|shuffle|mask_memory|hard|return_retrieval|argmax|memory_values|scores' \
  phef
```

### Inspect current result reports

```bash
cd ~/agi
python -m json.tool phef/results/capacitycurve_nseg10000000_seed7_evalseed19_b2x64.json
python -m json.tool phef/results/capacitycurve_probe_nseg12900000_seed7_evalseed19_b2x1.json
```

### Verify a completed report structurally

```bash
cd ~/agi
python - <<'PY'
import json
from pathlib import Path

path = Path('REPLACE_WITH_RESULT.json')
report = json.loads(path.read_text())
assert report['schema'] == 'compact_eval_v1'
assert report['hard_eval'] is True
for name in ('real', 'shuffled', 'masked'):
    assert name in report
    print(name, report[name])
PY
```

### Check status before any CUDA work

```bash
cd ~/agi
git status --short
pgrep -af 'eval_recall_compact.py|capacity' || true
nvidia-smi
```

## 10. Initial instruction to the new assistant

Read this handoff and the repository documentation first. Do not start a CUDA-heavy job immediately.

First:

1. Run `git status --short`, `pgrep -af 'eval_recall_compact.py|capacity' || true`, and `nvidia-smi`.
2. Confirm that the working tree is clean and no relevant CUDA job is active.
3. Inspect `phef/experiments/eval_recall_compact.py` and the relevant forward path in `phef/phef/model.py`.
4. Propose one narrow, semantics-preserving evaluator/control-path investigation aimed at reducing the 8.99M shuffled-path memory boundary.
5. Before any code edit or CUDA run, state the exact equivalence property to test and the smallest non-CUDA or small-scale test that would falsify it.

Proceed one command or one clearly bounded step at a time. The user runs local commands and returns terminal output. Do not send a large batch of commands at once.
