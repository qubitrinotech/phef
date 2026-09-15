# PHEF: результаты репликации streamed hard retrieval на 14M сегментов

**Дата:** 15 сентября 2026  
**Репозиторий:** `~/agi`  
**Коммит реализации:** `662edb6 phef: stream hard retrieval in compact evaluator`  
**Оборудование:** NVIDIA GeForce RTX 4090, 24 GiB VRAM

## Область применимости

Эта заметка фиксирует CUDA evaluation artifacts, полученные после коммита
streamed hard retrieval. PHEF оценивается только на контролируемой
синтетической задаче ассоциативного извлечения «ключ–значение», реализованной
compact evaluator.

Результаты не доказывают общую способность к памяти для языка, документов или
реальных задач. Они не являются сравнением с RAG, ANN, внешним lookup или
системами поиска.

Streamed path предназначен только для inference-only hard retrieval. Вместо
материализации плотного тензора retrieval scores размера
`(batch_size, n_segments)` он обрабатывает события чанками и хранит лучший
кандидат для каждого запроса. До запусков, зафиксированных здесь, были успешно
пройдены CPU equivalence tests, проверка tie behavior, API guards и CUDA parity
между legacy dense evaluator и streamed evaluator.

## Термины и метрики

- `n_keys` — размер виртуального пространства key IDs. В этих запусках
  `n_keys=1,073,741,824 = 2^30`; это параметр collision statistics, а не
  количество фактически хранимых записей и не результат измерения capacity.
- `n_segments` — число событий в одной memory trajectory.
- `n_eff_mean` — измеренное среднее число effective/active memory entries;
  это основная величина по оси capacity.
- `n_values=64`, поэтому nominal answer chance равен
  `1/64 = 0.015625`.
- Каждый evaluator run выполняет matched conditions в фиксированном порядке:
  `real -> shuffled -> masked`.
- `shuffled` сдвигает complete memory episodes между строками batch, сохраняя
  query/answer pairs. `masked` отключает memory input. В condition `masked`
  поле `retrieval_acc` намеренно равно `null`.

## Общий протокол 14M

```text
schema:                   compact_eval_v2_streamed_hard
device:                   cuda
n_keys:                   1,073,741,824
n_segments:               14,000,000
overwrite_p:              0.3
n_values:                 64
key_dim:                  32
dim:                      128
n_eff_theory:             9,755,413.64
eval_batch_size:          2
eval_batches:             16
episodes на condition:    32
stream_hard_chunk_size:   65,536
```

Использованные checkpoints:

```text
phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt
  training seed: 7
  selected epoch: 4750
  checkpoint selection accuracy: 0.9999186198

phef/results/v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt
  training seed: 23
  selected epoch: 5000
  checkpoint selection accuracy: 0.9999186198
```

## Матрица качества 2×2

Каждая cell ниже имеет terminal marker `saved:` в логе и JSON report, который
успешно распарсился и содержит полные блоки `real`, `shuffled` и `masked`.

| Training seed | Evaluation seed | Real answer / retrieval | Shuffled answer / retrieval | Masked answer | Real `n_eff_mean` |
|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 31/32 = 0.968750 / 31/32 = 0.968750 | 0/32 = 0.000000 / 0/32 = 0.000000 | 0/32 = 0.000000 | 9,755,875.59 |
| 23 | 7 | 31/32 = 0.968750 / 31/32 = 0.968750 | 1/32 = 0.031250 / 0/32 = 0.000000 | 1/32 = 0.031250 | 9,755,875.59 |
| 7 | 11 | 31/32 = 0.968750 / 31/32 = 0.968750 | 0/32 = 0.000000 / 0/32 = 0.000000 | 0/32 = 0.000000 | 9,755,691.41 |
| 23 | 11 | 32/32 = 1.000000 / 32/32 = 1.000000 | 0/32 = 0.000000 / 0/32 = 0.000000 | 0/32 = 0.000000 | 9,755,691.41 |

Описательные totals по четырём заранее заданным cells:

| Величина | Результат |
|---|---:|
| Real answer accuracy | 125/128 = 0.9765625 |
| Real retrieval accuracy | 125/128 = 0.9765625 |
| Shuffled answer accuracy | 1/128 = 0.0078125 |
| Shuffled retrieval accuracy | 0/128 = 0.0000000 |
| Masked answer accuracy | 1/128 = 0.0078125 |

Pooled values являются только описательной сводкой. Они суммируют два
independently trained checkpoints и два evaluation RNG streams, но не должны
представляться как точная population estimate. Поэтому выше сохранены также
результаты отдельных cells, а не только pooled total.

Единственный correct answer в shuffled condition не совпал с correct shuffled
retrieval; единственный correct answer в masked condition получен при
отключённой памяти. Оба наблюдения совместимы с редкими chance-level answers
в 64-class task.

## Наблюдения по ресурсам

### Medium-depth resource probes

| Probe | `n_segments` | Real `n_eff_mean` | GPU allocation peak | Peak RSS | Full protocol wall time |
|---|---:|---:|---:|---:|---:|
| Streamed probe | 12,900,000 | 8,993,033.0 | 15.60 GiB | 8.57 GiB | 26.746 s |
| Streamed probe | 14,000,000 | 9,756,961.0 | 16.92 GiB | 9.21 GiB | 29.563 s |

Каждый resource probe использовал `eval_batch_size=2`, `eval_batches=1`,
то есть два episodes на condition. Это измерения operational feasibility,
а не оценки capacity quality.

### Quality runs на 14M

Все четыре 14M quality runs показали cumulative process-level GPU allocation
peak `16.92 GiB` и peak RSS `11.96 GiB`.

| Training seed | Evaluation seed | Full protocol wall time |
|---:|---:|---:|
| 7 | 7 | 471.999965 s |
| 23 | 7 | 464.680367 s |
| 7 | 11 | 477.038103 s |
| 23 | 11 | 477.477788 s |

Evaluator не вызывает `torch.cuda.reset_peak_memory_stats()` между conditions.
Следовательно, отображаемая GPU allocation является cumulative process-level
peak, а не независимым измерением по каждому condition. Повторяемость
наблюдений на 14M не оправдывает линейную экстраполяцию на 16M или другую
неизмеренную глубину.

## Provenance artifacts

### Resource probes

```text
phef/logs/streamed_resource_probe_s12900000_c65536_b2x1_seed7.log
phef/results/streamed_resource_probe_s12900000_c65536_b2x1_seed7.json

phef/logs/streamed_resource_probe_s14000000_c65536_b2x1_seed7.log
phef/results/streamed_resource_probe_s14000000_c65536_b2x1_seed7.json
```

### Матрица качества 14M

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

## Поддерживаемые и неподдерживаемые выводы

Поддерживается:

1. В commit `662edb6` streamed hard retrieval выполнил указанный compact
   evaluator protocol на 14M segments на одной RTX 4090.
2. В рамках этой controlled task и этого protocol матрица 2×2 показала
   высокую exact real answer/retrieval accuracy при измеренном real
   `n_eff_mean` около 9.756M, тогда как shuffled retrieval оставался 0/32
   во всех cells, а shuffled/masked answers находились около 64-class chance
   baseline.
3. Указанная 14M configuration повторяемо показала `16.92 GiB` cumulative GPU
   allocation peak и `11.96 GiB` peak RSS.

Не поддерживается:

1. Утверждение об общей памяти, language memory, document memory, lossless
   compression, RAG, ANN или external lookup.
2. Утверждение, что `n_keys=2^30` — количество сохранённых memory entries.
3. Общий scaling law или feasibility claim для 16M либо иной неизмеренной
   глубины.
4. Точная population estimate, полученная из pooled four-cell result.
5. Утверждение, что только этот результат устанавливает окончательный
   capacity frontier.
