# KT-3A: bridge between soft and hard readout

**Status:** pre-run specification

## Purpose

Evaluate whether the two existing canonical PHEF checkpoints retain high hard-retrieval and hard-answer accuracy on a fixed external evaluation matrix, and measure the relation between the model's hard and soft answer readouts on identical generated batches.

KT-3A is a checkpoint-only evaluation. It does not train a model, compare PHEF with attention, Transformer, ANN, RAG, Mamba, SSM, or RNN, establish a capacity frontier, or change the completed KT-1 conclusion.

## Inputs and scope

Canonical checkpoints:

- `phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt`
- `phef/results/v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt`

The checkpoint model contract is the existing `phef.model.PHEF` model. Evaluation uses dense scores only. For every generated batch, the same checkpoint and the same batch are evaluated twice:

1. `hard=True, return_retrieval=True` to obtain hard logits and dense retrieval scores.
2. `hard=False` to obtain soft-readout logits.

No streamed-hard mode is used in KT-3A.

## Fixed matrix

| Field | Value |
|---|---:|
| Checkpoints | 2: train seeds 7 and 23 |
| Evaluation streams | 7 and 11 |
| Cells | 4: checkpoint × stream |
| `n_keys` | 262144 |
| `n_segments` | 65536 |
| `n_values` | 64 |
| `overwrite_p` | 0.3 |
| `key_dim` | 32 |
| `dim` | 128 |
| Evaluation batch size | 16 |
| Evaluation batches per cell | 64 |
| Episodes per cell | 1024 |
| Total episodes | 4096 |
| Device | CUDA unless `--cpu` is supplied |

Each cell uses a fresh `BranchingWorld` seeded with its declared evaluation-stream seed. The model checkpoint does not change during evaluation.

## Metrics

All accuracy metrics are represented in the cell artifact by integer counts and the common total, not only floating-point estimates.

| Metric | Definition |
|---|---|
| Hard answer | `hard_logits.argmax(dim=-1) == answers` |
| Soft answer | `soft_logits.argmax(dim=-1) == answers` |
| Hard retrieval | `scores.argmax(dim=-1) == answer_positions` |
| Retrieval-value | `values[row, scores.argmax(dim=-1)] == answers` |
| Hard-soft gap | `(hard_answer_correct - soft_answer_correct) / episodes` |
| Tie rate | Mean batch-row rate of `scores >= max(scores) - 1e-6` at more than one position |
| Normalized entropy | Mean softmax-score entropy divided by `ln(n_segments)` |

The artifact also records answer-distance buckets. Buckets divide normalized distance `((n_segments - 1) - answer_position) / max(1, n_segments - 1)` into 8 equal bins. For each bin, the artifact stores hard-answer correct count, hard retrieval slot-hit count, and total count.

## Required cell artifact fields

Each cell JSON must contain:

- schema identifier and protocol identifier;
- checkpoint path, checkpoint epoch and checkpoint selection accuracy;
- train seed label and evaluation seed;
- full fixed configuration;
- episodes and all integer metric counts;
- tie numerator and denominator;
- entropy sum and denominator;
- distance-bucket count arrays;
- wall seconds, peak CUDA allocation in GiB, peak RSS in GiB;
- Git commit, branch, dirty status, Python version, platform, NumPy version and Torch version.

## Aggregation and reporting

The statistics program accepts exactly the four predeclared cells:

- `(checkpoint seed 7, eval seed 7)`;
- `(checkpoint seed 7, eval seed 11)`;
- `(checkpoint seed 23, eval seed 7)`;
- `(checkpoint seed 23, eval seed 11)`.

It validates that all four exist exactly once, have the fixed configuration, use the expected checkpoint seed/evaluation seed pair, and contain internally valid counts. It aggregates counts over 4096 episodes.

Hard retrieval, hard answer and soft answer are reported as `successes / n`, together with 95% Wilson intervals. The aggregate hard-soft gap is computed from aggregate hard and soft answer counts. Tie rate is computed as aggregate tie rows divided by aggregate tie rows examined.

No interpretation is made until all four cells are complete and aggregation succeeds.

## Fixed verdict rules

`BRIDGE_PASS` requires all of the following:

1. All four predeclared cells are complete and pass artifact/configuration validation.
2. Every cell has tie rate no greater than 0.01.
3. The aggregate 95% Wilson lower bound for hard retrieval is at least 0.95.
4. The aggregate 95% Wilson lower bound for hard answer is at least 0.95.
5. The absolute aggregate hard-soft gap is no greater than 0.05.

`BRIDGE_FAIL` applies only when all four cells are valid and complete but one or more `BRIDGE_PASS` criteria are not met.

`BRIDGE_INCONCLUSIVE` applies if any required artifact is missing, duplicated, malformed, inconsistent with the fixed matrix, or has invalid metric counts.

The matrix, thresholds, metrics and verdict rules must not be changed after the first cell begins.

## Interpretation limits

A pass supports only this statement:

> For the two specified canonical PHEF checkpoints, at the declared canonical-scale external evaluation matrix, hard retrieval and hard answer meet the registered accuracy and consistency criteria, with the measured hard-soft answer gap within the registered tolerance.

A pass does not demonstrate superiority over attention, Transformers, ANN methods, databases, RAG, Mamba, SSM, RNNs, or any real-data baseline. It does not establish a capacity frontier and does not alter KT-1's fixed conclusions.
