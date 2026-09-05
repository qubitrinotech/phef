# PHEF — хендофф: S=65536 PASS, контроль shuffle и дальнейшая работа

Дата: 27 августа 2026.
Репозиторий: `~/qubitrino-projects/agi`.
Рабочий каталог: `~/qubitrino-projects/agi/phef`.
Ветка: `main`.
Последний отправленный коммит: `3c712b6 phef: pass S=65536 recall with value decode auxiliary loss`.

## 0. ГЛАВНОЕ ПРАВИЛО

# НЕ ГАДАТЬ. НЕ ФАНТАЗИРОВАТЬ. НЕ НАДУМЫВАТЬ.

Это правило важнее скорости и удобства.

1. Каждое содержательное утверждение помечать как **[ФАКТ]**, **[ВЫВОД]** или **[НЕИЗВЕСТНО]**.
   - **[ФАКТ]** — прямо подтверждён выводом команды, показанным пользователем в этом чате.
   - **[ВЫВОД]** — логическое следствие подтверждённых фактов; не выдавать за замер.
   - **[НЕИЗВЕСТНО]** — данных недостаточно. Не подменять пробел правдоподобной версией.
2. Агент не имеет прямого доступа к машине пользователя. Все команды выполняет пользователь и присылает их вывод.
3. Не угадывать содержание файлов, кода, JSON, checkpoint `.pt`, конфигураций или логов. Нужен фрагмент — запросить точную read-only команду (`sed`, `grep`, `python json.load`, `torch.load`) и дождаться вывода.
4. Всегда сверять фактический `config` в целевом JSON, а не пересказ из старого хендоффа.
5. Перед любым изменением кода, созданием, удалением, перемещением или коммитом файлов сначала показать точное действие/diff и запросить явное подтверждение пользователя.
6. Если гипотеза опровергнута замером, прямо написать, что она опровергнута. Не воскрешать её без новых данных.
7. Не экстраполировать скорость, память, качество или устойчивость между конфигурациями. Нужен отдельный замер конкретного режима.
8. Не объяснять числовые расхождения «на глаз». Если причина не проверена — `[НЕИЗВЕСТНО]` и конкретная проверка.
9. Команды давать готовыми для копирования, предпочтительно одним блоком и одной строкой запуска. Ответы — по-русски и кратко.
10. Для GPU перед Python всегда: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
11. Единственный допустимый Python:
   ```bash
   /home/linux/qubitrino-projects/agi/venv/bin/python
   ```
12. Долгие запуски — только через `nohup ... > лог 2>&1 &`; не запускать обучение через pipe с `tail`.
13. Перед созданием логов в `/tmp/probe_logs` всегда включать `mkdir -p /tmp/probe_logs` в команду. `/tmp` не является постоянным хранилищем.
14. Не склеивать heredoc создания файла и проверочные команды в один блок: в предыдущих сессиях это повреждало конец скрипта при вставке. Создание и проверку давать отдельными блоками.
15. Длинные команды с `LC_ALL=C` и `PYTHONPATH=.` давать одной строкой, без переносов с обратным слэшем и пробелом после него.
16. Перед коммитом обязательны: `git status --short`, `git diff --check`, синтаксис изменённых `.py`. Не добавлять результаты, черновики, `.bak` и временные файлы без отдельного согласования.

## 1. Окружение

- Git-корень: `~/qubitrino-projects/agi`.
- PHEF: `~/qubitrino-projects/agi/phef`.
- Пакет: `phef/phef/`.
- Скрипты: `phef/experiments/`.
- Remote: `gitlab.qubitrinotech.ru:root/agi.git`.
- В путях `git diff` PHEF имеет префикс `phef/`.
- GPU: RTX 4090, доступно torch около 23.52 GiB.
- Базовый префикс запуска:
  ```bash
  cd ~/qubitrino-projects/agi/phef && LC_ALL=C PYTHONPATH=.
  ```

## 2. Что уже сделано

### Отправленные коммиты

| Коммит | Содержание |
|---|---|
| `8e18019` | CPU-генерация slots в `world_gpu` |
| `150e9c7` | `batch()` возвращает `(slots, codebook)` вместо развёрнутых `event_keys` |
| `d440829` | pointer-doubling в `_sample_slots` вместо цикла по сегментам |
| `84dda15` | первый PASS малого масштаба: hard-eval, 1000 эпох |
| `3c712b6` | S=65536 PASS: no-fixed-point shuffle, hard-selection/checkpoint, auxiliary decoding loss, evaluator и итоговые артефакты |

[ФАКТ] `git push origin main` завершился успешно: `d440829..3c712b6 main -> main`.

### Производительность

[ФАКТ] До оптимизаций `batch()` на S=131072 занимал около 3487 мс из-за Python-цикла в GPU `_sample_slots`.

[ФАКТ] После переноса генерации slots на CPU, представления `(slots, codebook)` и pointer-doubling эпоха уменьшилась с 541.32 мс до 38.44 мс; 1.85 → 26.02 эпох/с.

[ФАКТ] Паритет распределительных метрик после оптимизаций был побитово идентичен; среди них `oracle_raw_top1=1.0`, `max_key_norm_error=0.0`, `mean_unique_slots=512.0`.

## 3. Главный результат S=65536

### Целевой режим

[ФАКТ] Итоговое обучение использовало:

```text
n_keys=512
n_values=64
n_segments=65536
batch_size=32
epochs=12000
lr=3e-4
hard_eval=True
eval_batch_size=16
eval_batches=64
use_best=True
acc_every=250
acc_n=512
aux_value_weight=0.1
seed=0
```

[ФАКТ] Лучший checkpoint был сохранён на epoch 3500 с `sel_acc=1.000000`.

[ФАКТ] После восстановления best checkpoint финальная оценка обучения дала:

```text
real answer      = 1.000000
real retrieval   = 1.000000
shuffled answer  = 0.018555
masked answer    = 0.017578
control status   = PASS
```

[ФАКТ] Независимая оценка checkpoint на 16384 эпизодах (`eval_batch_size=16`, `eval_batches=1024`) дала:

```text
checkpoint_epoch = 3500
checkpoint_sel_acc = 1.000000
real answer      = 0.999329
real retrieval   = 0.999329
shuffled answer  = 0.014587
shuffled retrieval= 0.000977
masked answer    = 0.013306
answer_chance    = 1/64 = 0.015625
```

[ВЫВОД] На текущей конфигурации S=65536 формально подтверждён: real answer/retrieval > 0.90, а shuffled/masked < `2 * chance = 0.03125`.

### Корректная формулировка результата

[ВЫВОД] Корректно утверждать: «PHEF при `n_keys=512`, `n_values=64`, `n_segments=65536`, hard retrieval и `aux_value_weight=0.1` достигает 0.999329 answer и retrieval accuracy на независимых 16384 эпизодах; shuffled и masked контроли находятся около chance».

[НЕИЗВЕСТНО] Этот один seed не доказывает устойчивость к другим seed, другим `overwrite_p`, большему S, иной размерности, другим числу значений или отсутствие всех возможных путей leakage. Такие заявления требуют отдельных экспериментов.

## 4. Установленные причины и исправления

### Shuffle-контроль

[ФАКТ] Раньше при `shuffle=True` применялся `torch.randperm(batch_size)`, который допускает fixed points.

[ФАКТ] На batch size 16 прежняя большая оценка checkpoint давала shuffled answer 0.045593 при real answer 0.504333 и masked 0.015869.

[ВЫВОД] Fixed points объясняли завышенный shuffled score: в среднем около 1/16 эпизодов сохраняли собственную память.

[ФАКТ] В `phef/model.py` теперь используется случайный ненулевой циклический сдвиг:

```python
shift = torch.randint(1, batch_size, (), device=values.device)
permutation = torch.roll(
    torch.arange(batch_size, device=values.device),
    shifts=int(shift.item()),
)
```

[ФАКТ] После исправления на старом checkpoint shuffled answer упал до 0.016113 при masked 0.015869.

[ВЫВОД] В рамках проверки fixed points были достаточной причиной прежнего повышения shuffled-контроля.

### Почему прежний checkpoint был около 0.50

[ФАКТ] Старый checkpoint `cap_k512_s65536_best3.pt` имел retrieval около 0.999878, но answer около 0.504333.

[ФАКТ] Его штатный `head` корректно декодировал только 35 из 64 value embeddings; новый линейный head на фиксированных `value_emb` достигал 100% уже к 10 шагу.

[ФАКТ] Усиление gate не устраняло проблему: accuracy была около 0.555 даже при почти полном memory-вкладе; `head(memory)` и `head(LayerNorm(memory))` были около 0.558.

[ВЫВОД] Bottleneck был не в retrieval, LayerNorm, gate или разделимости `value_emb`, а в недостаточном прямом обучающем сигнале штатной головы к декодированию value embeddings.

### Auxiliary loss

[ФАКТ] В `experiments/train_recall.py` добавлен параметр:

```python
--aux-value-weight
```

[ФАКТ] Его default равен `0.0`, поэтому базовое поведение без флага сохраняется.

[ФАКТ] При обучении вычисляется:

```python
aux_value_loss = F.cross_entropy(
    model.head(model.norm(model.value_emb.weight)),
    torch.arange(args.n_values, device=device),
)
```

и добавляется в общий loss с весом `args.aux_value_weight`.

[ВЫВОД] `--aux-value-weight 0.1` предоставил прямой обучающий сигнал к декодированию всех 64 embeddings и устранил выявленный bottleneck на S=65536.

## 5. Важный код

### `phef/phef/model.py`

[ФАКТ] `PHEF.forward` принимает:

```python
slots, codebook, values, timestamps, query_keys,
shuffle=False, mask_memory=False, hard=False,
return_retrieval=False
```

[ФАКТ] `hard=True` выбирает `scores.argmax(dim=1)` и соответствующий value embedding.

[ФАКТ] Мягкое чтение материализует `memory_values` формы `(bs, n_segments, dim)`, поэтому при большом evaluation batch может OOM.

[ВЫВОД] Для S=65536 использовать `--hard-eval` и `--eval-batch-size 16`, пока не будет отдельно проверен и реализован memory-safe soft path.

### `phef/experiments/train_recall.py`

[ФАКТ] При `--hard-eval` `quick_acc()` теперь передаёт `hard=HARD_EVAL`, поэтому датчик `--use-best` измеряет ту же геометрию чтения, что evaluation.

[ФАКТ] Периодический epoch-level `evaluate` использует `n_batches=8`, то есть при eval batch 16 — 128 эпизодов, вместо прежних 32.

[ФАКТ] `--best-out PATH.pt` сохраняет checkpoint с ключами `epoch`, `sel_acc`, `state_dict` при новом максимуме `sel_acc`; JSON report содержит `best_checkpoint`.

### `phef/experiments/eval_recall_checkpoint.py`

[ФАКТ] Новый скрипт загружает `.pt` checkpoint, всегда включает hard evaluation, проводит real/shuffled/masked оценки и сохраняет JSON.

[ФАКТ] Скрипт должен запускаться с `PYTHONPATH=.`.

[ВЫВОД] Не менять этот evaluator, пока не выполнена read-only проверка актуального кода и не согласовано, что именно нужно измерить.

## 6. С чего начать

Сначала выполнить **только read-only** проверку и прислать полный вывод:

```bash
cd ~/qubitrino-projects/agi && git status --short && git log --oneline -5 && git branch -vv && git diff --check && git diff -- phef/phef/model.py phef/experiments/train_recall.py phef/experiments/eval_recall_checkpoint.py && cd phef && for f in phef/model.py phef/world.py experiments/train_recall.py experiments/eval_recall_checkpoint.py; do /home/linux/qubitrino-projects/agi/venv/bin/python -c "import ast; ast.parse(open('$f').read()); print('ok $f')"; done && /home/linux/qubitrino-projects/agi/venv/bin/python - <<'PY'
import json, os, torch
for f in ["results/cap_k512_s65536_aux01.json", "results/cap_k512_s65536_aux01_eval16384.json"]:
    print("\n===", f, "===")
    with open(f) as h:
        d = json.load(h)
    print(json.dumps(d, indent=2, sort_keys=True))
p = "results/cap_k512_s65536_aux01.pt"
print("\n===", p, "===")
print("exists=", os.path.exists(p), "bytes=", os.path.getsize(p) if os.path.exists(p) else None)
if os.path.exists(p):
    c = torch.load(p, map_location="cpu", weights_only=True)
    print("keys=", sorted(c))
    print("epoch=", c.get("epoch"), "sel_acc=", c.get("sel_acc"), "state_dict_tensors=", len(c.get("state_dict", {})))
PY
```

[ФАКТ] На момент передачи ожидаются многочисленные **untracked** старые результаты, handoff-файлы, `.bak` и `logs/microtest/`. Они не были включены в коммит намеренно.

[ВЫВОД] Не удалять и не добавлять эти файлы в коммит без отдельного решения пользователя.

## 7. Какие файлы приложить

Минимальный пакет для нового чата:

1. Этот файл: `PHEF_handoff_S65536_PASS.md`.
2. `phef/phef/model.py`.
3. `phef/phef/world.py`.
4. `phef/experiments/train_recall.py`.
5. `phef/experiments/eval_recall_checkpoint.py`.
6. `phef/results/cap_k512_s65536_aux01.json`.
7. `phef/results/cap_k512_s65536_aux01_eval16384.json`.
8. При необходимости воспроизведения/диагностики: `phef/results/cap_k512_s65536_aux01.pt`.

Полезные исторические хендоффы, если нужно восстановить путь оптимизаций:

- `docs/handoffs/18_PHEF_HANDOFF.md`.
- `docs/handoffs/19_PHEF — хендофф ёмкость памяти и устойчивость обучения.md`.
- `docs/handoffs/20_PHEF-handoff-S65536-checkpoint-shuffle-control.md`.

[ВЫВОД] Не нужно прикладывать `train_recall.py.bak`, `/tmp`-логи, smoke results и временные диагностические скрипты: они не являются частью воспроизводимого утверждения.

## 8. Что делать дальше

### Первый рекомендуемый шаг: воспроизводимость по seed

[ВЫВОД] Основной следующий научный вопрос — не «улучшить уже 0.999», а проверить воспроизводимость PASS на независимых seeds.

Перед запуском нового обучения:

1. Прочитать актуальный код и JSON командой из §6.
2. Согласовать точные seeds, число повторов, имена JSON/checkpoint/логов и критерий успеха.
3. Не перезаписывать артефакты `aux01`.

Предлагаемый минимальный дизайн после подтверждения пользователя:

- Повторить идентичную конфигурацию для seed 1 и seed 2.
- Для каждого seed сохранить best checkpoint через `--best-out`.
- После обучения выполнить независимую оценку checkpoint на 16384 эпизодах с тем же seed.
- Считать run PASS только если real answer/retrieval > 0.90 и shuffled/masked < 0.03125.

[НЕИЗВЕСТНО] Нельзя заранее утверждать, что seed 1 и 2 пройдут: пока подтверждён только seed 0.

### Второй шаг: абляция веса auxiliary loss

[ВЫВОД] Если seed-reproduction подтвердится, полезна узкая абляция `aux_value_weight` (например 0.01, 0.03, 0.1) при одном новом seed или при меньшем числе эпох, чтобы определить минимально достаточный вес.

[НЕИЗВЕСТНО] Не выбирать веса или длительность прогона окончательно без предварительного согласования и чтения актуальных результатов.

### Дальнейшее масштабирование

[ВЫВОД] Только после воспроизводимости имеет смысл отдельно исследовать S > 65536, изменение `n_values`, `n_keys`, `overwrite_p` или soft-read path.

[НЕИЗВЕСТНО] PASS на S=65536 не доказывает PASS на S=131072 или другом режиме. Для каждого нового масштаба нужны измерение VRAM, throughput, обучение, независимый checkpoint evaluation и controls.

## 9. Опровергнутые гипотезы

Не воскрешать без новых замеров:

1. «В S=65536 нет тайбрейкера» — `recency_scale` обучается; адресация достигла ~1.0.
2. «Голова принципиально не может декодировать value embeddings» — свежая голова на фиксированных embeddings достигла 100% к шагу 10.
3. «LayerNorm — причина ~0.50 accuracy» — `head(memory)` и `head(LayerNorm(memory))` давали почти одинаково ~0.558 на старом checkpoint.
4. «Gate — единственная причина ~0.50 accuracy» — усиление gate подняло старый checkpoint лишь примерно до 0.555.
5. «Высокий shuffled score доказывает query leakage» — до no-fixed-point контроля это было менее экономное объяснение; после исправления shuffle score стал около chance.
6. «12000 эпох обязательно улучшают S=65536» — без auxiliary loss качество деградировало после пика; с `--use-best` должен использоваться сохранённый лучший checkpoint.
7. «Мягкое чтение — адекватный selection-сигнал при S=65536» — ранее оно было шумовым; hard selection обязателен в текущем целевом режиме.
8. «Коллапс bf16 неизбежен около S=4096–16384» — не подтверждён.
9. «Кросс-энтропия неизбежно разгоняет масштаб при lr=3e-4» — не подтверждено прежними замерами.

## 10. Полезные команды

### Быстрая проверка итогового checkpoint без обучения

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/eval_recall_checkpoint.py --checkpoint results/cap_k512_s65536_aux01.pt --n-keys 512 --n-values 64 --n-segments 65536 --overwrite-p 0.3 --key-dim 32 --dim 128 --eval-batch-size 16 --eval-batches 1024 --seed 0 --out results/cap_k512_s65536_aux01_eval16384_repeat.json > /tmp/probe_logs/cap_s65536_aux01_eval16384_repeat.log 2>&1 &
```

После завершения:

```bash
cd ~/qubitrino-projects/agi/phef && grep -E "^(device=|real answer|real retrieval|shuffled answer|shuffled retrieval|masked answer|saved:)" /tmp/probe_logs/cap_s65536_aux01_eval16384_repeat.log
```

### Статус перед коммитом

```bash
cd ~/qubitrino-projects/agi && git status --short && git diff --check && git diff --stat && git log --oneline -5
```

### Синтаксис файлов

```bash
cd ~/qubitrino-projects/agi/phef && for f in phef/model.py phef/world.py experiments/train_recall.py experiments/eval_recall_checkpoint.py; do /home/linux/qubitrino-projects/agi/venv/bin/python -c "import ast; ast.parse(open('$f').read()); print('ok $f')"; done
```
