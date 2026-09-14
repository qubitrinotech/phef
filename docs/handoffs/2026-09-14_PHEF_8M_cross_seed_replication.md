# PHEF: 8M effective-memory cross-seed replication

**Date:** 14 September 2026  
**Repository:** `~/agi`  
**Hardware:** NVIDIA GeForce RTX 4090, 24 GiB VRAM  
**Status:** completed, reproducible synthetic-memory milestone

## Executive result

PHEF achieved reliable exact associative retrieval at a directly measured effective memory depth of approximately **8.044 million active entries** in the current controlled synthetic key-value benchmark.

The result was reproduced in a crossed 2×2 design:

- two independently trained checkpoints (`training seed=7` and `training seed=23`);
- two independently generated evaluation worlds (`eval seed=7` and `eval seed=11`);
- 128 real episodes per cell, for 512 real episodes total;
- matched `real`, `shuffled`, and `masked` evaluation conditions for every cell.

The four real evaluations produced **504 correct exact retrievals out of 512**, a descriptive pooled accuracy of **98.4375%**. Individual cell accuracies ranged from 97.65625% to 99.21875%.

This is evidence for large-scale retrieval in this synthetic PHEF task. It is not evidence that PHEF is already a production-ready natural-language long-term-memory system.

## Pre-registered question

> Can selected PHEF checkpoints retrieve the correct value from episodes containing approximately 8 million effective memory entries, while matched shuffled and memory-masked controls remain at chance-level behavior?

## Model and task

PHEF is evaluated on a synthetic associative-retrieval task.

- A memory episode contains a sequence of key/value events.
- A query key is sampled from the episode.
- The model must return the value associated with the most recent matching key.
- Values have `n_values=64`, so answer chance is `1/64 = 0.015625`.
- `real` evaluates normal memory-based retrieval.
- `shuffled` permutes complete memory episodes across batch elements while leaving each query/answer pair in place, breaking memory-query correspondence.
- `masked` disables memory input.

The primary metric is exact `retrieval_acc`. Exact `answer_acc` is also reported. In the reported real evaluations, answer and retrieval accuracy were equal in every cell.

## Fixed evaluation protocol

All four cells used the same evaluation configuration except for the training checkpoint and evaluation RNG seed.

```text
script: phef/experiments/eval_recall_compact.py
schema: compact_eval_v1
hard_eval: true

n_keys = 1,073,741,824
n_segments = 11,534,336
overwrite_p = 0.3
n_values = 64
key_dim = 32
dim = 128
eval_batch_size = 2
eval_batches = 64
episodes per condition = 128
progress_every = 8
```

The evaluator reports both a theoretical estimate and an observed effective depth. The theoretical value was:

```text
n_eff_theory = 8,043,754.7
```

Observed real-condition `n_eff_mean` ranged from 8,043,761.7 to 8,043,833.2. Thus the reported result is explicitly about approximately 8.044M effective entries, not merely a large key-universe parameter.

## Checkpoints

Both checkpoints use `n_train=32768` and v2 checkpoint selection with a separate selection world. The relevant selection recipe includes:

```text
sel_n_keys = 262144
sel_acc_n = 4096
sel_window = 3
```

| Training seed | Checkpoint | Selected epoch | Selection accuracy |
|---:|---|---:|---:|
| 7 | `phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt` | 4750 | 0.9999186198 |
| 23 | `phef/results/v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt` | 5000 | 0.9999186198 |

The `_e6000_best.pt` suffix is an output filename, not a claim that the selected state necessarily came from epoch 6000.

## Cross-seed results

| Training seed | Eval seed | Selected epoch | Real correct / 128 | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Real `n_eff_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 4750 | 125 / 128 | 0.9765625 | 0.9765625 | 0.0000000 | 0.0000000 | 0.0234375 | 8,043,761.7 |
| 23 | 7 | 5000 | 126 / 128 | 0.9843750 | 0.9843750 | 0.0312500 | 0.0000000 | 0.0156250 | 8,043,761.7 |
| 7 | 11 | 4750 | 126 / 128 | 0.9843750 | 0.9843750 | 0.0234375 | 0.0000000 | 0.0156250 | 8,043,833.2 |
| 23 | 11 | 5000 | 127 / 128 | 0.9921875 | 0.9921875 | 0.0078125 | 0.0000000 | 0.0078125 | 8,043,833.2 |

Descriptive aggregate across the four real cells:

```text
correct real retrievals = 504 / 512
pooled descriptive real accuracy = 0.984375 = 98.4375%
real-cell range = 0.9765625 to 0.9921875
```

This aggregate is descriptive, not a precise population estimate: observations are grouped into four cells and originate from two training checkpoints and two evaluation RNG streams.

## Controls and interpretation

### Shuffled control

In every one of the four runs:

```text
shuffled retrieval accuracy = 0 / 128 = 0.0
```

The complete memory episode is shifted across batch elements while query and answer remain fixed. Zero retrieval hits across all four shuffled evaluations supports the conclusion that real-condition success depends on correct memory-query correspondence.

Shuffled answer accuracy ranged from 1/128 to 4/128 (0.78125% to 3.125%), which is compatible with sampling variation around the 64-class answer chance of 1.5625%.

### Masked control

With memory disabled, masked answer accuracy ranged from 1/128 to 3/128 (0.78125% to 2.34375%). These values are compatible with the same 1/64 chance baseline.

### Real answer and retrieval

For every real cell, `answer_acc == retrieval_acc`. Therefore the final answer head did not compensate for incorrect retrieval in the reported runs; the measured real answer accuracy reflects exact retrieval success in this task.

## Evaluator optimization

The 8M evaluation required a targeted optimization of the hard-evaluation branch in `phef/phef/model.py`.

Previously, hard evaluation materialized value embeddings for every event:

```python
memory_values = self.value_emb.weight.to(query_state.dtype)[values]
```

This has shape `(batch_size, n_segments, dim)` even though `hard=True` selects only one event per batch item. The revised hard branch computes the argmax first, gathers only the selected value IDs, and embeds only those IDs:

```python
index = scores.argmax(dim=1)
selected_values = values[
    torch.arange(batch_size, device=values.device),
    index,
]
memory_state = self.value_emb(selected_values).to(query_state.dtype)
```

The soft branch remains unchanged. A smoke test and an identical-seed 8.39M-segment probe preserved all measured task metrics while reducing observed peak allocation at that probe from 17.46 GiB to 14.72 GiB.

The optimization is semantics-preserving for the hard branch because:

```text
value_emb.weight[values][row, index]
== value_emb(values[row, index])
```

## Resource profile

The full 8M runs were completed on one RTX 4090 with no fatal CUDA out-of-memory exception.

Representative full-run resource profile:

| Metric | Observed value |
|---|---:|
| Peak GPU allocation after shuffled | 20.31 GiB |
| Physical GPU memory | 24,564 MiB |
| Peak process RSS | approximately 10.07 GiB |
| Real wall time | approximately 525–536 s per 128 episodes |
| Shuffled wall time | approximately 510–528 s per 128 episodes |
| Masked wall time | approximately 503–508 s per 128 episodes |
| Full three-condition wall time | approximately 25–26 minutes |

The allocator emitted repeated `expandable_segments` mapping warnings near the memory limit. They were nonfatal in these completed runs: progress continued, RSS and peak allocation remained stable, each of the three conditions completed, and the evaluator wrote terminal JSON files.

Operational constraints:

- Run one CUDA-heavy evaluation at a time.
- Do not allow two processes to write the same log or JSON path.
- Treat the current 8M configuration as near-capacity for `eval_batch_size=2`; do not extrapolate directly to 16M without further evaluator/memory work.

## Reproducibility artifacts

Relevant commits, in chronological order:

```text
80c2de4 phef: validate 8M effective-memory retrieval on seed 7
4c57db4 phef: replicate 8M effective-memory retrieval on seed 23
edd96d6 phef: validate 8M retrieval on eval seed 11
fe66587 phef: complete 8M retrieval cross-seed replication
```

Primary results and logs:

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

## Claims supported by this study

Supported:

1. PHEF can perform high-accuracy exact retrieval in this controlled synthetic task at directly measured `n_eff≈8.044M`.
2. The result reproduced across two independently trained checkpoints and two independent evaluation RNG streams.
3. The high real result depends on the memory/query correspondence under the shuffled and masked tests used here.
4. The complete three-condition protocol is operationally feasible on one RTX 4090 after the hard-evaluation memory optimization.

Not supported:

1. A claim of general-purpose natural-language or document long-term memory.
2. A claim that arbitrary text can be losslessly compressed into a fixed neural state at this scale.
3. A precise estimate of the population error rate beyond this limited 2×2, 128-episodes-per-cell design.
4. A claim that 8M is the maximum capacity or that 16M is currently feasible with the same evaluator and hardware configuration.
5. A direct comparison with external-memory systems such as exact lookup, ANN retrieval, or production RAG.

## Recommended next work

### 1. Strengthen statistical evidence at 8M

Run more episodes with new evaluation seeds before making finer-grained error claims. Good next increments are 512 or 1024 episodes per cell, preferably using new evaluation seeds and keeping the full real/shuffled/masked protocol.

### 2. Map the capacity curve

With fixed checkpoints, evaluate several depths around and above the current result. Useful initial targets are approximately:

```text
n_eff ≈ 5.86M  (n_segments=8,388,608)
n_eff ≈ 8.04M  (n_segments=11,534,336; completed)
n_eff ≈ 9–10M (select segment count from observed n_eff)
```

The purpose is to distinguish gradual error growth from a capacity cliff.

### 3. Engineer a 16M-capable evaluator

The current `eval_batch_size=2` full protocol is near GPU capacity at 8M due to the shuffled control and large sequence-shaped tensors. A credible 16M study requires an explicit resource design, such as a semantically validated singleton-compatible shuffled control, streamed/chunked scoring if the model semantics allow it, or a separate-device/more-memory configuration. Do not label an increased `n_keys` setting as an increase in effective memory depth.

### 4. Move beyond synthetic key-value retrieval

A practical memory benchmark should include stored source records, provenance, updates and conflicts, semantic or approximate queries, distractors, multi-hop retrieval, and comparisons against exact lookup and ANN/RAG baselines. Retrieval accuracy should remain separate from answer-generation quality.

## Follow-up: approximately 6.98M effective-memory capacity-curve pilot

**Date:** 15 September 2026
**Status:** completed single-cell pilot; not a cross-seed replication

This follow-up adds one controlled point below the completed 8.044M study:

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
```

Measured and theoretical effective depth:

```text
n_eff_theory: 6,977,232.1
real n_eff_mean: 6,977,123.2
shuffled n_eff_mean: 6,977,252.5
masked n_eff_mean: 6,977,331.6
```

Results:

| Condition | Episodes | Answer accuracy | Retrieval accuracy |
|---|---:|---:|---:|
| Real | 128 | 0.9765625 (125/128) | 0.9765625 (125/128) |
| Shuffled | 128 | 0.0078125 (1/128) | 0.0000000 (0/128) |
| Masked | 128 | 0.0078125 (1/128) | not scored |

The real answer and retrieval metrics were equal. The shuffled retrieval control was 0/128, while shuffled and masked answer accuracy were each 1/128, consistent with the 64-class chance baseline.

Resource profile for the full three-condition run:

```text
real peak GPU allocation: 15.51 GiB
worst cumulative GPU allocation peak after shuffled: 17.62 GiB
peak RSS: 8.86 GiB
real/shuffled/masked wall time: 439.5 s / 437.8 s / 437.3 s
full three-condition wall time: approximately 21.9 minutes
```

Artifacts:

```text
phef/results/capacitycurve_probe_nseg10000000_seed7_evalseed19_b2x1.json
phef/logs/capacitycurve_probe_nseg10000000_seed7_evalseed19_b2x1.log

phef/results/capacitycurve_nseg10000000_seed7_evalseed19_b2x64.json
phef/logs/capacitycurve_nseg10000000_seed7_evalseed19_b2x64.log
```

Interpretation: this is one full-control pilot at measured `n_eff≈6.977M`. It supports feasibility and provides an intermediate point for a future capacity curve. It does not by itself estimate a capacity trend, establish a difference from 8.044M, or replace the 8M 2×2 cross-seed replication.

## Follow-up: approximately 9.0M effective-memory resource probe

**Date:** 15 September 2026
**Status:** completed 2-episode-per-condition resource probe; not a capacity result

A short full-protocol probe tested the next capacity-curve target above the completed 8.044M study:

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
```

Measured effective depth:

```text
n_eff_theory: 8,992,135.8
real n_eff_mean: 8,990,361.0
shuffled n_eff_mean: 8,991,882.5
masked n_eff_mean: 8,991,542.0
```

The one-batch real, shuffled, and masked conditions all completed and the evaluator wrote a complete `compact_eval_v1` JSON. The worst cumulative GPU allocation peak was 22.60 GiB after shuffled, with real reaching 19.88 GiB. This leaves a small margin on a 24 GiB RTX 4090 and is materially closer to the boundary than the completed 8M full protocol.

Artifacts:

```text
phef/results/capacitycurve_probe_nseg12900000_seed7_evalseed19_b2x1.json
phef/logs/capacitycurve_probe_nseg12900000_seed7_evalseed19_b2x1.log
```

Interpretation: this confirms that the evaluator can complete one full-control batch at measured `n_eff≈8.99M`; it does not establish retrieval accuracy at 9M and does not justify a 128-episode full run with the current memory profile. Before a 9M pilot or 16M study, profile and redesign the evaluator/control path for a more robust memory margin.
