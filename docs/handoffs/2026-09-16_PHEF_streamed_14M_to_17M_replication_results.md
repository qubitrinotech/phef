# PHEF: streamed hard retrieval — репликация на 14M–17M segments

**Дата:** 16 сентября 2026  
**Репозиторий:** `~/agi`  
**Ветка:** `main`  
**Коммит реализации streamed path:** `662edb6 phef: stream hard retrieval in compact evaluator`  
**Предыдущий 14M documentation commit:** `86a9169 docs: record streamed 14M replication results`  
**14M artifacts commit:** `bd8434c phef: add 14M streamed evaluation artifacts`  
**15M resource-probe commit:** `d212883 phef: record 15M streamed resource probe`  
**16M resource-probe commit:** `7697c27 phef: record 16M streamed resource probe`  
**17M resource-probe commit:** `6419e2f phef: record 17M streamed resource probe`  
**Оборудование:** NVIDIA GeForce RTX 4090, 24 GiB VRAM

## Область применимости

Эта заметка фиксирует расширенную серию CUDA evaluations для PHEF на контролируемой синтетической задаче точного ассоциативного извлечения latest value по паре «ключ–значение». Все результаты получены compact evaluator с streamed hard retrieval.

Результаты относятся только к этой synthetic task, этой реализации, указанным checkpoints, параметрам мира и одному GPU environment. Они не являются доказательством общей memory capability для естественного языка, документов, произвольных данных или реальных агентных задач. Это не benchmark и не comparison с RAG, ANN, external lookup, key–value storage либо lossless compression.

## Реализация и валидация

`662edb6` добавляет streamed hard retrieval для hard, inference-only evaluation. Вместо материализации плотного `(batch_size, n_segments)` тензора retrieval scores события обрабатываются чанками, а для каждого query сохраняется текущий лучший candidate. Затем embedding извлекается только для выбранного value.

До runs, приведённых здесь, были выполнены:

- CPU equivalence matrix dense versus streamed, включая random cases, shuffled path и tie case через границы chunks.
- API guards для неподдерживаемых режимов.
- CUDA evaluator parity smoke между legacy dense и streamed path во всех conditions `real → shuffled → masked`.

Положительный chunk size использует report schema `compact_eval_v2_streamed_hard`. Streamed retrieval применяется только к enabled/unmasked memory в `real` и `shuffled`; `masked` condition сохраняет обычную masked ветвь, поэтому `retrieval_acc` в ней намеренно равен `null`.

## Термины и protocol

- `n_segments` — число events в одной memory trajectory.
- `n_eff_mean` — измеренное среднее число effective/active memory entries; это главная величина по оси capacity.
- `n_keys=1,073,741,824 = 2^30` — размер key-ID universe, влияющий на collision statistics; это не число фактически сохранённых memory entries и не результат capacity measurement.
- `n_values=64`, поэтому nominal answer chance равен `1/64 = 0.015625`.
- `real` использует исходное соответствие memory и query.
- `shuffled` cyclically shifts complete memory episodes между batch rows при сохранении query/answer pairs, разрушая корректное memory-query correspondence.
- `masked` отключает memory input.

Каждый full quality run выполняет matched conditions в фиксированном порядке:

```text
real → shuffled → masked
```

Общий protocol для каждой depth и каждой ячейки 2×2:

```text
schema:                   compact_eval_v2_streamed_hard
device:                   cuda
n_keys:                   1,073,741,824
overwrite_p:              0.3
n_values:                 64
key_dim:                  32
dim:                      128
eval_batch_size:          2
eval_batches:             16
episodes per condition:   32
stream_hard_chunk_size:   65,536
conditions:               real → shuffled → masked
```

Каждая depth имеет четыре заранее определённые replication cells:

| Training checkpoint | Training seed | Selected epoch | Evaluation seed |
|---|---:|---:|---:|
| `v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt` | 7 | 4750 | 7 |
| `v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt` | 23 | 5000 | 7 |
| `v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt` | 7 | 4750 | 11 |
| `v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt` | 23 | 5000 | 11 |

Оба checkpoint-а имеют recorded selection accuracy `0.9999186198`.

## Результаты по глубинам

Каждая строка ниже — descriptive pooled total четырёх заранее определённых cells: 2 independently trained checkpoints × 2 evaluation RNG streams, то есть 128 episodes в каждом condition. Pooled values удобны для сводки, но не являются точной population estimate; individual cells приведены далее.

| `n_segments` | Средний real `n_eff_mean` | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer |
|---:|---:|---:|---:|---:|---:|---:|
| 14M | 9,755,783.50 | 125/128 = 0.9765625 | 125/128 = 0.9765625 | 1/128 = 0.0078125 | 0/128 = 0.0000000 | 1/128 = 0.0078125 |
| 15M | 10,448,940.19 | 122/128 = 0.9531250 | 122/128 = 0.9531250 | 3/128 = 0.0234375 | 0/128 = 0.0000000 | 1/128 = 0.0078125 |
| 16M | 11,142,046.72 | 125/128 = 0.9765625 | 125/128 = 0.9765625 | 2/128 = 0.0156250 | 0/128 = 0.0000000 | 4/128 = 0.0312500 |
| 17M | 11,834,283.70 | 126/128 = 0.9843750 | 126/128 = 0.9843750 | 3/128 = 0.0234375 | 0/128 = 0.0000000 | 2/128 = 0.0156250 |

Across all four depth points, real answer accuracy equals real retrieval accuracy in the pooled totals and in the individual cells. Thus, observed real answer errors coincide with retrieval-selection errors in this protocol rather than a divergence between selected memory event and classifier output.

Across all depths, shuffled retrieval is `0/128`. Shuffled and masked answer counts remain near the nominal 64-class chance baseline on these finite samples. The 16M masked count is `4/128`, above the nominal expected count of 2 but still small relative to real retrieval and not accompanied by a masked retrieval metric; it should be reported descriptively rather than interpreted as evidence of leakage.

## Individual replication cells

### 14M segments

| Training seed | Evaluation seed | Real answer / retrieval | Shuffled answer / retrieval | Masked answer | Real `n_eff_mean` | Full wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 31/32 = 0.968750 / 31/32 = 0.968750 | 0/32 = 0.000000 / 0/32 = 0.000000 | 0/32 = 0.000000 | 9,755,875.59 | 472.0 s |
| 23 | 7 | 31/32 = 0.968750 / 31/32 = 0.968750 | 1/32 = 0.031250 / 0/32 = 0.000000 | 1/32 = 0.031250 | 9,755,875.59 | 464.7 s |
| 7 | 11 | 31/32 = 0.968750 / 31/32 = 0.968750 | 0/32 = 0.000000 / 0/32 = 0.000000 | 0/32 = 0.000000 | 9,755,691.41 | 477.0 s |
| 23 | 11 | 32/32 = 1.000000 / 32/32 = 1.000000 | 0/32 = 0.000000 / 0/32 = 0.000000 | 0/32 = 0.000000 | 9,755,691.41 | 477.5 s |

### 15M segments

| Training seed | Evaluation seed | Real answer / retrieval | Shuffled answer / retrieval | Masked answer | Real `n_eff_mean` | Full wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 30/32 = 0.937500 / 30/32 = 0.937500 | 1/32 = 0.031250 / 0/32 = 0.000000 | 0/32 = 0.000000 | 10,448,758.88 | 512.6 s |
| 23 | 7 | 32/32 = 1.000000 / 32/32 = 1.000000 | 1/32 = 0.031250 / 0/32 = 0.000000 | 0/32 = 0.000000 | 10,448,758.88 | 511.7 s |
| 7 | 11 | 29/32 = 0.906250 / 29/32 = 0.906250 | 1/32 = 0.031250 / 0/32 = 0.000000 | 0/32 = 0.000000 | 10,449,121.50 | 513.1 s |
| 23 | 11 | 31/32 = 0.968750 / 31/32 = 0.968750 | 0/32 = 0.000000 / 0/32 = 0.000000 | 1/32 = 0.031250 | 10,449,121.50 | 504.3 s |

### 16M segments

| Training seed | Evaluation seed | Real answer / retrieval | Shuffled answer / retrieval | Masked answer | Real `n_eff_mean` | Full wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 32/32 = 1.000000 / 32/32 = 1.000000 | 2/32 = 0.062500 / 0/32 = 0.000000 | 0/32 = 0.000000 | 11,142,098.75 | 528.4 s |
| 23 | 7 | 31/32 = 0.968750 / 31/32 = 0.968750 | 0/32 = 0.000000 / 0/32 = 0.000000 | 0/32 = 0.000000 | 11,142,098.75 | 521.8 s |
| 7 | 11 | 31/32 = 0.968750 / 31/32 = 0.968750 | 0/32 = 0.000000 / 0/32 = 0.000000 | 2/32 = 0.062500 | 11,141,994.69 | 522.8 s |
| 23 | 11 | 31/32 = 0.968750 / 31/32 = 0.968750 | 0/32 = 0.000000 / 0/32 = 0.000000 | 2/32 = 0.062500 | 11,141,994.69 | 522.1 s |

### 17M segments

| Training seed | Evaluation seed | Real answer / retrieval | Shuffled answer / retrieval | Masked answer | Real `n_eff_mean` | Full wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 32/32 = 1.000000 / 32/32 = 1.000000 | 1/32 = 0.031250 / 0/32 = 0.000000 | 0/32 = 0.000000 | 11,834,360.16 | 560.0 s |
| 23 | 7 | 32/32 = 1.000000 / 32/32 = 1.000000 | 1/32 = 0.031250 / 0/32 = 0.000000 | 1/32 = 0.031250 | 11,834,360.16 | 559.8 s |
| 7 | 11 | 30/32 = 0.937500 / 30/32 = 0.937500 | 1/32 = 0.031250 / 0/32 = 0.000000 | 0/32 = 0.000000 | 11,834,207.25 | 566.6 s |
| 23 | 11 | 32/32 = 1.000000 / 32/32 = 1.000000 | 0/32 = 0.000000 / 0/32 = 0.000000 | 1/32 = 0.031250 | 11,834,207.25 | 566.3 s |

## Resource observations

The evaluator reports GPU allocation peak cumulatively for the process and does not reset peak memory statistics between conditions. Therefore `gpu_alloc_peak` is a process-level worst peak, not an independent per-condition measurement.

### Two-episode resource probes

| Probe depth | Real `n_eff_mean` | Cumulative GPU allocation peak | Peak RSS | Full protocol wall time |
|---:|---:|---:|---:|---:|
| 12.9M segments | 8,993,033.0 | 15.60 GiB | 8.57 GiB | 26.746 s |
| 14M segments | 9,756,961.0 | 16.92 GiB | 9.21 GiB | 29.563 s |
| 15M segments | 10,450,242.5 | 18.12 GiB | 9.80 GiB | about 32.2 s |
| 16M segments | 11,143,095.0 | 19.32 GiB | 10.40 GiB | about 33.7 s |
| 17M segments | 11,835,588.0 | 20.52 GiB | 10.92 GiB | about 36.6 s |

### Full quality runs

Full quality-run wall time increased with depth:

| Depth | Observed full-run wall-time range |
|---:|---:|
| 14M | 464.7–477.5 s |
| 15M | 504.3–513.1 s |
| 16M | 521.8–528.4 s |
| 17M | 559.8–566.6 s |

The 15M full run observed a cumulative GPU allocation peak of `18.12 GiB` and RSS stabilizing around `12.74 GiB`, matching the 15M resource probe allocation peak. The final 17M long-run cell completed all conditions and passed JSON validation despite repeated nonfatal `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` warnings that a 20 MiB mapping could not be made. Its allocation peak was approximately `20.53 GiB` and RSS reached approximately `14.29 GiB`.

These warnings must be reported, not hidden. They were nonfatal in the observed run because progress continued through all 16 batches of each condition, the evaluator emitted `saved:`, and JSON validation passed. They nevertheless indicate proximity to an allocator/mapping boundary; no linear extrapolation from these points to 18M or beyond is justified.

## Interpretation

Supported by the current evidence:

1. Streamed hard retrieval completed full matched-control evaluations at 14M, 15M, 16M and 17M segments on a single RTX 4090 under the documented protocol.
2. Across the 2×2 matrix at each depth, high real exact answer/retrieval accuracy was retained at observed real `n_eff_mean` from about 9.756M through about 11.834M.
3. Shuffled retrieval was `0/128` at every tested depth, while shuffled and masked answer counts remained near the finite-sample 64-class chance baseline.
4. The implementation and evaluator routing have CPU equivalence, API-guard and CUDA evaluator-parity evidence described above.
5. Operational resource cost increased with depth; 17M long runs completed but showed allocator-mapping warnings, making this a cautious upper tested point rather than evidence of an open-ended scaling regime.

Not supported by the current evidence:

1. A claim about general-purpose memory, language memory, document memory, arbitrary compression or real-world retrieval.
2. A claim that `n_keys=2^30` is the number of stored memory entries.
3. A population estimate or statistical significance claim based only on the four-cell pooled totals.
4. A general scaling law, a definitive capacity frontier, or feasibility at 18M and beyond.
5. A comparison to RAG, ANN, external lookup or key–value database systems.

## Artifact inventory

### 14M quality matrix

```text
phef/logs/streamed_quality_s14000000_c65536_b2x16_seed7.log
phef/results/streamed_quality_s14000000_c65536_b2x16_seed7.json
phef/logs/streamed_quality_s14000000_c65536_b2x16_ckpt23_evalseed7.log
phef/results/streamed_quality_s14000000_c65536_b2x16_ckpt23_evalseed7.json
phef/logs/streamed_quality_s14000000_c65536_b2x16_ckpt7_evalseed11.log
phef/results/streamed_quality_s14000000_c65536_b2x16_ckpt7_evalseed11.json
phef/logs/streamed_quality_s14000000_c65536_b2x16_ckpt23_evalseed11.log
phef/results/streamed_quality_s14000000_c65536_b2x16_ckpt23_evalseed11.json
```

### 15M quality matrix

```text
phef/logs/streamed_quality_s15000000_c65536_b2x16_seed7.log
phef/results/streamed_quality_s15000000_c65536_b2x16_seed7.json
phef/logs/streamed_quality_s15000000_c65536_b2x16_ckpt23_evalseed7.log
phef/results/streamed_quality_s15000000_c65536_b2x16_ckpt23_evalseed7.json
phef/logs/streamed_quality_s15000000_c65536_b2x16_ckpt7_evalseed11.log
phef/results/streamed_quality_s15000000_c65536_b2x16_ckpt7_evalseed11.json
phef/logs/streamed_quality_s15000000_c65536_b2x16_ckpt23_evalseed11.log
phef/results/streamed_quality_s15000000_c65536_b2x16_ckpt23_evalseed11.json
```

### 16M quality matrix

```text
phef/logs/streamed_quality_s16000000_c65536_b2x16_seed7.log
phef/results/streamed_quality_s16000000_c65536_b2x16_seed7.json
phef/logs/streamed_quality_s16000000_c65536_b2x16_ckpt23_evalseed7.log
phef/results/streamed_quality_s16000000_c65536_b2x16_ckpt23_evalseed7.json
phef/logs/streamed_quality_s16000000_c65536_b2x16_ckpt7_evalseed11.log
phef/results/streamed_quality_s16000000_c65536_b2x16_ckpt7_evalseed11.json
phef/logs/streamed_quality_s16000000_c65536_b2x16_ckpt23_evalseed11.log
phef/results/streamed_quality_s16000000_c65536_b2x16_ckpt23_evalseed11.json
```

### 17M quality matrix

```text
phef/logs/streamed_quality_s17000000_c65536_b2x16_seed7.log
phef/results/streamed_quality_s17000000_c65536_b2x16_seed7.json
phef/logs/streamed_quality_s17000000_c65536_b2x16_ckpt23_evalseed7.log
phef/results/streamed_quality_s17000000_c65536_b2x16_ckpt23_evalseed7.json
phef/logs/streamed_quality_s17000000_c65536_b2x16_ckpt7_evalseed11.log
phef/results/streamed_quality_s17000000_c65536_b2x16_ckpt7_evalseed11.json
phef/logs/streamed_quality_s17000000_c65536_b2x16_ckpt23_evalseed11.log
phef/results/streamed_quality_s17000000_c65536_b2x16_ckpt23_evalseed11.json
```

### Resource-probe artifacts

```text
phef/logs/streamed_resource_probe_s12900000_c65536_b2x1_seed7.log
phef/results/streamed_resource_probe_s12900000_c65536_b2x1_seed7.json
phef/logs/streamed_resource_probe_s14000000_c65536_b2x1_seed7.log
phef/results/streamed_resource_probe_s14000000_c65536_b2x1_seed7.json
phef/logs/streamed_resource_probe_s15000000_c65536_b2x1_seed7.log
phef/results/streamed_resource_probe_s15000000_c65536_b2x1_seed7.json
phef/logs/streamed_resource_probe_s16000000_c65536_b2x1_seed7.log
phef/results/streamed_resource_probe_s16000000_c65536_b2x1_seed7.json
phef/logs/streamed_resource_probe_s17000000_c65536_b2x1_seed7.log
phef/results/streamed_resource_probe_s17000000_c65536_b2x1_seed7.json
```
