# PHEF — актуальный handoff

**Дата:** 13 сентября 2026  
**Репозиторий:** `~/agi`  
**Окружение:** RTX 4090 (24 GiB), Python в `~/agi/.venv`  
**Цель:** проверить, может ли PHEF надёжно извлекать записи на очень большой эффективной глубине памяти и приблизиться к practically useful long-term memory, а не бесконечно оптимизировать малые synthetic regimes.

---

## 1. Краткий текущий вывод

В синтетической key–value задаче PHEF показывает валидный retrieval-сигнал: реальные результаты высокие, а `shuffled`/`masked` controls остаются около chance. После замены старой насыщавшейся метрики выбора чекпоинта на v2 результаты по train support стали монотонными:

- `n_train=4096`: средняя дальняя error 0.011068;
- `n_train=16384`: средняя дальняя error 0.003255;
- `n_train=32768`: средняя дальняя error 0.001628.

Эти значения получены на одной дальней точке `n_segments=524288`, с `n_keys=1073741824`, `n_eff_mean≈366941.1`, 2048 эпизодов и общим eval seed 7.

**Главный практический вывод:** старое кажущееся ухудшение при большем `n_train` было артефактом старого checkpoint selection, который насыщался и выбирал слишком раннюю эпоху. V2-отбор выбирает поздние чекпоинты и даёт устойчивое улучшение при 4096 → 16384 → 32768.

**Следующий крупный milestone:** не ещё одна мелкая scan-серия, а capacity test с целевым `n_eff` порядка 8M и затем 16M, если 8M проходит. Важно: `n_keys=8,388,608` — это размер universe ключей, а не автоматически 8M фактически активных записей; эффективная глубина измеряется `n_eff_mean`.

---

## 2. Модель и задача

PHEF — экспериментальная predictive hysteresis memory для synthetic associative retrieval.

Задача в текущих запусках:

- `n_keys`: размер universe ключей / training world или eval world, в зависимости от скрипта;
- `n_segments`: число сегментов/шагов памяти в эпизоде;
- `overwrite_p=0.3`: вероятность overwrite;
- `n_values=64`: answer chance = `1/64 = 0.015625`;
- модель получает query и memory state, должна извлечь соответствующее value;
- `real` — нормальная проверка;
- `shuffled` — нарушение соответствия memory/retrieval; ожидается answer около chance, retrieval 0;
- `masked` — memory mask; ожидается answer около chance.

Параметры архитектуры и базового оптимизатора в текущем воспроизводимом рецепте:

```text
n_values=64
n_segments=65536               # в обучении
overwrite_p=0.3
key_dim=32
dim=128
hard_eval=true
epochs=6000
batch_size=32
eval_batch_size=16             # локальная training evaluation
eval_batches=64
lr=0.0003
weight_decay=0.0001
retrieval_weight=1.0
retrieval_weight_late=-1.0
retrieval_weight_switch_epoch=0
aux_value_weight=0.01
use_best=true
acc_every=250
acc_n=512
escape_factor=2.0
```

---

## 3. Почему был нужен v2 checkpoint selection

### Проблема старого отбора

Старый отбор использовал in-distribution/local validation, которая часто достигала `1.0` рано. В частности, исторические `ntrain=16384` и `ntrain=32768` были оценены по checkpoint, выбранным около epoch 2500 при saturated score. Это приводило к ложному выводу, будто больший train support ухудшает дальнюю генерализацию.

### V2-отбор

Скрипт:

```text
phef/experiments/train_recall_v2.py
```

Ключевые v2 параметры:

```text
--sel-n-keys 262144
--sel-acc-n 4096
--sel-window 3
```

- Selection world имеет отдельный, более крупный `n_keys=262144`.
- Разрешение одной selection оценки: `1/4096 = 0.000244140625`.
- Отбор идёт по среднему последних трёх selection measurements.
- Скрипт сохраняет текущий лучший checkpoint по ходу работы, а после конца обучения восстанавливает выбранный state в путь `--best-out`.
- Имя `*_e6000_best.pt` — это **имя выходного файла**, а не обязательно epoch checkpoint. Например, seed 23 мог выбрать epoch 4750, но состояние epoch 4750 лежит в файле с suffix `_e6000_best.pt`.

V2 всё ещё почти насыщается при `n_train=32768`, так как отношение `sel_n_keys/n_train=8`. Для будущих обучений с train world 32768 и выше следует рассмотреть `sel_n_keys=1048576` или больше.

---

## 4. Полные результаты v2 study

### Единый дальний eval protocol

Все следующие результаты получены по одному протоколу:

```text
script: phef/experiments/eval_recall_compact.py
n_keys=1073741824
n_segments=524288
n_values=64
overwrite_p=0.3
key_dim=32
dim=128
eval_batch_size=2
eval_batches=1024
episodes=2048
progress_every=512
eval seed=7
n_eff_mean≈366941.1
```

### Дальняя error по train size и seed

| n_train | seed 7 | seed 11 | seed 23 | mean error | sample SD | range |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 0.015137 | 0.004883 | 0.013184 | 0.011068 | 0.00547 | 0.010254 |
| 16384 | 0.004883 | 0.001953 | 0.002930 | 0.003255 | 0.00150 | 0.002930 |
| 32768 | 0.000977 | 0.001953 | 0.001953 | 0.001628 | 0.000564 | 0.000976 |

Equivalent real accuracies at `n_train=32768`:

| seed | real answer | real retrieval | errors / 2048 |
|---:|---:|---:|---:|
| 7 | 0.999023 | 0.999023 | 2 |
| 11 | 0.998047 | 0.998047 | 4 |
| 23 | 0.998047 | 0.997559 | 4 |

### Matched-pairs comparison

| seed | error 4096 | error 16384 | error 32768 | 16384−4096 | 32768−16384 |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.015137 | 0.004883 | 0.000977 | −0.010254 | −0.003906 |
| 11 | 0.004883 | 0.001953 | 0.001953 | −0.002930 | 0.000000 |
| 23 | 0.013184 | 0.002930 | 0.001953 | −0.010254 | −0.000977 |

Negative delta means lower error at larger `n_train`.

Interpretation:

- All three seeds improve for 4096 → 16384.
- For 16384 → 32768, two improve and one is tied at 2048-episode resolution.
- The large apparent bad historical results at higher `n_train` should not be used for scaling claims; they used saturated old checkpoint selection.
- Inter-seed variance is substantial at 4096, much smaller at 16384 and 32768.
- At 32768 only 2–4 errors occur in 2048 episodes, so future evaluation needs 8192 or 16384 episodes to resolve further gains.

### Controls

Across all recorded v2 far evaluations:

- `shuffled answer` is near answer chance `0.015625`;
- `shuffled retrieval = 0.0`;
- `masked answer` is near chance;
- controls are PASS.

This supports that high real accuracy comes through memory/retrieval rather than a no-memory shortcut.

---

## 5. Current best checkpoints

For the present synthetic regime, the most useful candidates are the matched `n_train=32768` v2 checkpoints:

```text
phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt
phef/results/v2sel_k32768_sel262144_s65536_seed11_e6000_best.pt
phef/results/v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt
```

Selected epochs:

| n_train | seed 7 | seed 11 | seed 23 |
|---:|---:|---:|---:|
| 4096 | 6000 | 6000 | 5500 |
| 16384 | 4750 | 6000 | 4750 |
| 32768 | 4750 | 4750 | 5000 |

For 32768, final selection scores are nearly saturated (~0.9999), so use the checkpoints but recognize that exact late epoch selection is not fully resolved by `sel_n_keys=262144`.

---

## 6. Git history

Relevant commits, in order:

```text
a23f9cb  sel metric v2: non-saturating selection at N_sel=262144
5986474  measure seed variance at fixed ntrain 4096
a993781  replicate ntrain 16384 with v2 selection
5b10a33  complete matched ntrain 4096 vs 16384 v2 study
f8c0a65  complete v2 ntrain 32768 seed series
```

All generated PHEF logs/results/checkpoints are intentionally versioned. `.gitignore` has exceptions for `phef/results/**` and `phef/logs/**`; `.pt` files in this project are small (~236 KiB) and are tracked.

Before any new run:

```bash
cd ~/agi
git status --short
```

After meaningful completed experiments, explicitly stage logs, JSON results, and `.pt` best checkpoints, then commit.

---

## 7. 8M / 16M clarification

The user’s research objective is memory at 4–8M+ elements, analogous to loading very large information into persistent AI memory and retrieving it later without losing relevant context.

Important distinction:

- `n_keys` is a universe of possible keys / key space.
- `n_segments` controls sequence/memory depth in an episode.
- `n_eff_mean` is the directly reported effective number of memory entries/active unique items and is the relevant empirical capacity measure.

Existing 8M file:

```text
phef/results/early_s65536_seed7_rw10_keys8388608_b2_probe_retry2.json
```

Its configuration was:

```text
checkpoint = phef/results/early_s65536_seed7_rw10_e6000_best.pt
n_keys = 8388608
n_segments = 65536
eval_batch_size = 2
eval_batches = 1
seed = 7
```

It only contained **2 episodes** and did **not** report `n_eff_mean` in its old schema. It is a smoke/probe that `n_keys=8,388,608` runs, not a proof of 8M effective memory entries or a statistically meaningful capacity measurement.

Current far eval already uses `n_keys=1073741824` (2^30), which is far larger than 8M or 16M as a key universe. Therefore increasing `n_keys` from 8M to 16M is not a capacity scale-up relative to the current long-range test. To test 8M/16M memory **depth**, the relevant quantity to scale is `n_segments` until the output reports target `n_eff_mean`.

With `overwrite_p=0.3`, current `n_segments=524288` gives `n_eff_mean≈366941`, i.e. roughly 70% of segments. A rough initial conversion is:

```text
target n_segments for n_eff≈8M  ≈ 8,000,000 / 0.70 ≈ 11.4M
target n_segments for n_eff≈16M ≈ 16,000,000 / 0.70 ≈ 22.9M
```

These are only rough planning values. The actual target must be confirmed from `n_eff_mean` in the evaluator, and the generator/VRAM/runtime capability must be probed before long runs.

---

## 8. Recommended next experiment: capacity frontier

Do **not** begin another small `n_train` scan. Run a resource probe toward an 8M effective-memory target using a strong current checkpoint, then do a rigorous evaluation only after confirming actual `n_eff_mean` and feasibility.

### Design goals

Primary question:

> Can the selected PHEF model retrieve the correct value from an episode with approximately 8 million effective memory entries on one RTX 4090, while shuffled/masked controls remain at chance?

Pre-register a practical success criterion:

1. `n_eff_mean` meets or closely approaches the target (8M first).
2. `real` retrieval and answer are far above chance.
3. `shuffled retrieval=0` and shuffled/masked answer are near `1/64`.
4. No CUDA OOM; report wall time and peak VRAM.
5. For full measurement, use at least 8192 episodes; use 16384 if runtime permits, because a 2048-episode test cannot resolve rare errors near 10^-3 well.
6. After a successful seed-7 pilot, repeat at least one independent training seed (preferably seed 23) before claiming stable capacity.
7. Only after an 8M PASS, attempt 16M.

### Candidate model

Start with:

```text
phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt
```

Then replicate on seed 23 if successful:

```text
phef/results/v2sel_k32768_sel262144_s65536_seed23_e6000_best.pt
```

### First action: discover evaluator limits before a giant run

Run help and inspect any constraints/defaults. Do not assume the evaluator can allocate multi-million segments efficiently.

```bash
cd ~/agi && PYTHONPATH=phef .venv/bin/python phef/experiments/eval_recall_compact.py --help
```

Search evaluator internals for segment/length assumptions and n_eff calculation:

```bash
cd ~/agi && grep -R -n -E 'n_eff|n_segments|segments|chunk|batch|CUDA|memory' \
  phef/experiments/eval_recall_compact.py phef 2>/dev/null | head -160
```

### Probe protocol — do not run a full 8M eval immediately

Use a one-episode or two-episode resource probe at a candidate segment count. Begin lower than the rough target if unknown, then escalate. The exact candidate should be selected after inspecting the evaluator; likely initial candidates are powers of two such as 1,048,576; 2,097,152; 4,194,304; 8,388,608; 16,777,216 segments.

Generic probe pattern:

```bash
cd ~/agi && PYTHONPATH=phef PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python phef/experiments/eval_recall_compact.py \
  --checkpoint phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt \
  --n-keys 1073741824 --n-segments <CANDIDATE_SEGMENTS> --n-values 64 \
  --overwrite-p 0.3 --key-dim 32 --dim 128 \
  --eval-batch-size 1 --eval-batches 1 \
  --progress-every 1 --seed 7 \
  --out phef/results/probe_k32768_s<CANDIDATE_SEGMENTS>_b1x1.json
```

After every probe, record:

- `n_eff_mean`;
- elapsed time;
- `gpu_alloc_peak` / `nvidia-smi` memory;
- whether OOM occurred;
- whether the result JSON exists.

Do not launch several large probes concurrently.

### Full 8M evaluation — only after probe succeeds

Once a candidate segment count produces approximately 8M `n_eff_mean` and fits in memory, run the full real + shuffled + masked evaluation first on seed 7. The exact number of eval batches should be chosen based on per-batch timing from the probe. Target 8192 episodes initially; with `eval_batch_size=1`, that is `eval_batches=8192`; with batch size 2, 4096 batches.

Example shape (replace placeholders after probe):

```bash
cd ~/agi && nohup bash -c '
set -euo pipefail
PYTHONPATH=phef PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python phef/experiments/eval_recall_compact.py \
  --checkpoint phef/results/v2sel_k32768_sel262144_s65536_seed7_e6000_best.pt \
  --n-keys 1073741824 --n-segments <SEGMENTS_FOR_8M_NEFF> --n-values 64 \
  --overwrite-p 0.3 --key-dim 32 --dim 128 \
  --eval-batch-size <SAFE_BATCH> --eval-batches <BATCHES_FOR_8192_EPISODES> \
  --progress-every <REASONABLE_INTERVAL> --seed 7 \
  --out phef/results/capacity8m_k32768_seed7.json
' > phef/logs/capacity8m_k32768_seed7.log 2>&1 &
echo "started pid=$!"
```

Use the existing compact evaluator schema for consistency and require complete `real`, `shuffled`, and `masked` blocks before treating the output as valid.

### 16M decision

Proceed to ~16M only when:

- 8M actual `n_eff_mean` is confirmed;
- 8M controls PASS;
- runtime/memory are acceptable;
- seed 7 has nontrivial performance above chance;
- ideally seed 23 replicates the qualitative success.

If 8M fails, diagnose the failure form rather than changing many knobs:

- resource failure: OOM/time;
- `n_eff_mean` not near target: generation/overwriting issue;
- real retrieval falls but controls PASS: true capacity/generalization limit;
- controls fail: methodological bug/shortcut;
- accuracy floor too coarse: increase episode count.

---

## 9. How this relates to real LLM memory

The project should distinguish three claims:

1. **Huge context window:** feed every token into attention. This is expensive and not equivalent to reliable recall.
2. **External long-term memory:** persist original records externally, retrieve a small relevant subset, and place that subset in the model’s working context.
3. **PHEF capacity:** synthetic evidence that a learned memory/retrieval mechanism can retrieve an associated value at large effective memory depth.

A realistic production AI memory system would not compress 8–16M arbitrary text fragments into a fixed neural state and reproduce everything perfectly. It would store originals plus metadata and provenance, retrieve candidates, rerank/verify them, and put only the relevant items into a limited working context.

PHEF’s current synthetic result is not yet natural-language memory. It is evidence about a prerequisite: controlled key–value retrieval with negative controls at increasing memory depth. To move toward text, future work needs a benchmark with:

- text/document chunks as values or structured records;
- exact IDs plus semantic/approximate queries;
- source provenance and versioning;
- updates/conflicts/temporal facts;
- multi-hop retrieval and distractors;
- retrieval accuracy separate from answer quality;
- baseline comparisons such as exact hash lookup and ANN/RAG.

---

## 10. Operational conventions

### Run environment

Use:

```bash
cd ~/agi
PYTHONPATH=phef PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python ...
```

### Background jobs

Use one CUDA-heavy job at a time. Example:

```bash
nohup bash -c 'set -euo pipefail; ...' > phef/logs/<name>.log 2>&1 &
echo "started pid=$!"
```

Check with:

```bash
ps -o pid,etime,stat,cmd -p <PID> 2>&1 || echo 'wrapper process gone'
tail -50 phef/logs/<name>.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

### If accidentally duplicated a job

Do not allow two processes to write the same JSON/log. Stop wrappers and child Python workers; then remove only partial files of the duplicated experiment and rerun once. Do not remove established results/checkpoints.

### JSON extraction pattern

Output schemas differ across older and newer evaluator versions. Robustly obtain real result as:

```python
r = d.get('results', d).get('real', d.get('real', {}))
answer_acc = r.get('answer_acc')
error = 1.0 - answer_acc
```

### Commit convention

For a completed result, include:

- source changes, if any;
- exact stdout log;
- output JSON;
- selected `.pt` checkpoint;
- concise commit message that records protocol and key measured numbers.

Do not claim an eval complete unless log has finished all `real`, `shuffled`, and `masked` blocks and printed `saved:` / terminal completion marker.

---

## 11. Recommended immediate next-chat response

If continuing immediately, first inspect `eval_recall_compact.py --help` and the code path that creates segments and computes `n_eff_mean`. Then formulate a *small, resource-safe* capacity probe, not a speculative giant full run.

The user explicitly prefers moving toward 8M/16M-scale memory rather than endless small-regime tuning. Maintain that direction. The current question is whether the hardware and evaluator can realize an actual `n_eff_mean≈8M`, not whether `n_keys` can be set to 8M.
