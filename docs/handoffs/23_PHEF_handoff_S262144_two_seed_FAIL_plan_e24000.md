# PHEF — самодостаточный хендофф 23: S=262144 двух-seed independent FAIL, план главной проверки epochs=24000

Дата передачи: 28 августа 2026.

Репозиторий: `~/qubitrino-projects/agi`.
Рабочий каталог PHEF: `~/qubitrino-projects/agi/phef`.
Ветка: `main`.
Последний отправленный commit: `d41245c phef: record S=262144 negative result for seeds 7-8 (independent FAIL, clean controls)`.
Remote: `gitlab.qubitrinotech.ru:root/agi.git`.

---

## 0. ГЛАВНОЕ ПРАВИЛО — НЕ ГАДАТЬ

# НЕ ГАДАТЬ. НЕ ФАНТАЗИРОВАТЬ. НЕ ПОДМЕНЯТЬ НЕИЗВЕСТНОЕ ПРАВДОПОДОБНОЙ ВЕРСИЕЙ.

Это правило важнее скорости, удобства и желания «помочь побыстрее».

### 0.1. Маркировка утверждений

Каждое содержательное утверждение обязательно помечать одной из меток:

- **[ФАКТ]** — прямо подтверждён выводом команды, логом, JSON, checkpoint metadata, `git`-выводом или иным первичным артефактом, показанным пользователем в этом чате.
- **[ВЫВОД]** — логическое следствие одного или нескольких подтверждённых фактов. Это не новый замер и не должно формулироваться как измеренная величина.
- **[НЕИЗВЕСТНО]** — данных недостаточно. Нельзя заменять пробел догадкой. Вместо этого запросить конкретную минимальную проверку.

Примеры корректности:

- «[ФАКТ] `real answer=0.188660`» — если это число есть в присланном JSON/логе.
- «[ВЫВОД] Seed 7 не проходит порог, потому что `real answer < 0.90`» — если эти числа измерены.
- «[НЕИЗВЕСТНО] Peak VRAM при S=262144 неизвестен» — пока нет отдельного измерения именно этого режима.

### 0.2. Граница доступа

- Агент **не имеет прямого доступа** к машине пользователя, shell, Git, GPU, файлам, JSON, checkpoint `.pt`, конфигурациям и `/tmp`-логам.
- Все команды выполняет пользователь и присылает полный вывод.
- Нельзя говорить «в файле находится…», «GPU использует…», «процесс завершился…» без присланного подтверждения.

### 0.3. Не угадывать содержимое

Не угадывать содержимое: Python-кода; JSON и checkpoint `.pt`; конфигураций; системных журналов; `.gitignore`; логов обучения; статуса процессов; потребления VRAM, времени, throughput или температуры.

Если нужен фрагмент, запросить точную **read-only** команду и дождаться результата. Предпочитать:

```bash
sed -n 'START,ENDp' FILE
```

```bash
grep -nE 'PATTERN' FILE
```

```bash
/home/linux/qubitrino-projects/agi/venv/bin/python -c "...json.load(...)..."
```

```bash
/home/linux/qubitrino-projects/agi/venv/bin/python -c "...torch.load(..., weights_only=True)..."
```

### 0.4. Config — только из целевого JSON

Нельзя подтверждать параметры эксперимента по памяти, старому хендоффу или пересказу. Перед выводом о конкретном run сверять фактический `config` в **целевом JSON** этого run/eval.

### 0.5. Изменения только после согласия

До любого действия, которое меняет состояние, обязательно:

1. Показать точное планируемое действие.
2. Явно перечислить затрагиваемые файлы/сервисы/артефакты.
3. Для кода — показать точный diff до применения.
4. Получить явное подтверждение пользователя.
5. После действия запросить проверку фактического результата.

К действиям с изменением состояния относятся: создание/редактирование/удаление/перемещение файлов; `git add`, `git commit`, `git push`; запуск обучения/eval, создающий JSON/checkpoint/log; остановка или удаление системного сервиса; изменение BIOS, GRUB, kernel parameters, драйверов, Xorg.

Не маскировать команду изменения в длинном блоке и не действовать по неявному «ок».

Практика, подтверждённая в работе над этим хендоффом: перед каждым write-действием агент просил буквальную фразу подтверждения, например «Подтверждаю запуск exact full-training-команды S=262144 seed 8». Эту практику сохранить.

### 0.6. Гипотеза, опровергнутая замером

Если гипотеза опровергнута — прямо так и написать. Не воскрешать её без новых данных.

### 0.7. Никакой экстраполяции

Нельзя переносить между конфигурациями: качество; устойчивость по seed; скорость; peak VRAM; время обучения; поведение loss; оптимальный epoch; надёжность controls.

Каждая новая конфигурация, например `epochs=24000` вместо `epochs=12000`, требует собственного запуска и независимого eval.

### 0.8. Числовые расхождения — не объяснять «на глаз»

Если числа расходятся, причина — **[НЕИЗВЕСТНО]**, пока не проверена. Нельзя объявлять причиной «случайность», «влияние GPU», «утечку», «gate», «LayerNorm», «мало эпох», «перегрузку CPU» без измерения, которое различает объяснения.

Важно: «мало эпох» для S=262144 — это именно **гипотеза, подлежащая проверке** (см. §7), а не установленная причина.

### 0.9. Формат ответов и команд

- Отвечать по-русски и кратко.
- Команды давать готовыми для копирования.
- Предпочитать один блок и одну строку запуска для длинных команд.
- Не давать много тяжёлых команд одновременно.
- Для рекурсивного `grep` всегда ограничивать область и вывод; не сканировать большие каталоги во время обучения.

### 0.10. Строгие правила Python/GPU

Единственный разрешённый Python:

```bash
/home/linux/qubitrino-projects/agi/venv/bin/python
```

Перед любым Python, использующим GPU, обязательно:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Базовый префикс PHEF:

```bash
cd ~/qubitrino-projects/agi/phef && LC_ALL=C PYTHONPATH=.
```

Долгие run запускать только через `nohup`, с логом и в фоне. Перед созданием лога всегда включать `mkdir -p /tmp/probe_logs` в ту же команду. `/tmp` — непостоянное хранилище, поэтому воспроизводимые результаты сохраняются в `phef/results/` и коммитятся как JSON.

Нельзя запускать обучение через pipe с `tail`.

### 0.11. Heredoc, пути и shell

- Не склеивать heredoc создания файла и проверочные команды в один блок.
- Длинные команды с `LC_ALL=C` и `PYTHONPATH=.` давать **одной строкой**, без `\` и переносов.
- Учитывать текущий каталог: в Git-корне пути имеют префикс `phef/`, внутри `~/qubitrino-projects/agi/phef` — `results/...`, `experiments/...`, `phef/...`.

### 0.12. Правила Git

Перед commit обязательны: `git status --short`; `git diff --check`; `git diff --cached --check`; `git diff --cached --stat`; `git diff --cached`.

Также проверять синтаксис изменённых `.py`:

```bash
/home/linux/qubitrino-projects/agi/venv/bin/python -c "import ast; ast.parse(open('FILE.py').read()); print('ok FILE.py')"
```

Не добавлять без отдельного согласования: `.pt` checkpoints; `.bak`; временные скрипты; `/tmp`-логи; `logs/microtest/`; исторические/черновые результаты; handoff-файлы, не обновлённые под новый результат; technical probe JSON.

Commit всегда формировать только из белого списка явно названных файлов.

---

## 1. Окружение и репозиторий

- Git-root: `~/qubitrino-projects/agi`.
- PHEF: `~/qubitrino-projects/agi/phef`.
- Пакет: `phef/phef/`.
- Скрипты: `phef/experiments/`.
- Remote: `gitlab.qubitrinotech.ru:root/agi.git`.
- В путях Git из root PHEF имеет префикс `phef/`.
- GPU compute: NVIDIA GeForce RTX 4090, 24564 MiB total по `nvidia-smi`.
- Видеовывод desktop: Intel UHD 770 через `i915`/X11.
- Kernel на момент прошлой системной диагностики: `7.0.0-30-generic` Ubuntu.
- NVIDIA driver на момент прошлой системной диагностики: `595.84`, CUDA по `nvidia-smi`: `13.2`.
- Hostname в приглашении shell: `linux-desktop2`; пользователь `linux`.

[ФАКТ] `git status --short` 28 августа показывал только untracked записи (`??`) и ни одного изменённого tracked файла; `git diff --check` был пуст.

---

## 2. Отправленные commits

| Commit | Содержание |
|---|---|
| `8e18019` | CPU-генерация slots в `world_gpu` |
| `150e9c7` | `batch()` возвращает `(slots, codebook)` |
| `d440829` | pointer-doubling в `_sample_slots` |
| `84dda15` | первый PASS малого масштаба: hard-eval, 1000 epochs |
| `3c712b6` | S=65536 PASS: no-fixed-point shuffle, hard-selection/checkpoint, auxiliary decoding loss, evaluator, seed-0 артефакты |
| `16e4cf0` | S=65536: воспроизводимые JSON для seed 1–2 |
| `d81b3c6` | S=131072: train/eval JSON для seeds 4–6 |
| `d41245c` | S=262144: train/eval JSON для seeds 7–8, отрицательный результат |

[ФАКТ] `git push origin main` для последнего commit завершился успешно: `d81b3c6..d41245c main -> main`.

[ФАКТ] Commit `d41245c` содержал ровно 4 файла, 560 insertions, все как `create mode 100644`:

- `phef/results/cap_k512_s262144_aux001_seed7.json`
- `phef/results/cap_k512_s262144_aux001_seed7_eval16384.json`
- `phef/results/cap_k512_s262144_aux001_seed8.json`
- `phef/results/cap_k512_s262144_aux001_seed8_eval16384.json`

[ФАКТ] `.pt` checkpoints S=131072 и S=262144, `probe_s262144_aux001_seed7.json`, handoff-документы, `train_recall.py.bak`, `logs/microtest/` и исторические результаты остались untracked и намеренно не коммитились.

---

## 3. Проверенный метод, код и CLI

### 3.1. Базовый режим large-S

```text
n_keys=512
n_values=64
overwrite_p=0.3
key_dim=32
dim=128
batch_size=32
lr=3e-4
weight_decay=1e-4
retrieval_weight=1.0
aux_value_weight=0.01
hard_eval=True
eval_batch_size=16
use_best=True
acc_every=250
acc_n=512
```

[ФАКТ] Независимый evaluator использует hard evaluation, `eval_batch_size=16`, `eval_batches=1024`, то есть 16384 эпизода.

### 3.2. CLI `experiments/train_recall.py`

[ФАКТ] `--help` подтвердил флаги: `--n-keys`, `--n-values`, `--n-segments`, `--overwrite-p`, `--key-dim`, `--hard-eval`, `--dim`, `--epochs`, `--batch-size`, `--eval-batch-size`, `--eval-batches`, `--lr`, `--weight-decay`, `--retrieval-weight`, `--aux-value-weight`, `--use-best`, `--acc-every`, `--acc-n`, `--escape-factor`, `--no-memory`, `--seed`, `--cpu`, `--out`, `--best-out`.

[ФАКТ] `--out` — «путь к JSON-отчёту; без него результат только в stdout».
[ФАКТ] `--best-out` — «путь к .pt лучшего checkpoint по sel_acc; без него checkpoint не сохраняется».
[ФАКТ] `--escape-factor` — «порог выхода с плато в кратности chance (2.0 => 2/n_values)».
[ФАКТ] Default `--epochs` равен 1500.

### 3.3. CLI `experiments/eval_recall_checkpoint.py`

[ФАКТ] `--help` подтвердил: обязательные `--checkpoint` и `--out`; опциональные `--n-keys`, `--n-values`, `--n-segments`, `--overwrite-p`, `--key-dim`, `--dim`, `--eval-batch-size`, `--eval-batches`, `--seed`, `--cpu`.

### 3.4. Структура training loop

[ФАКТ] `sed -n '245,385p' experiments/train_recall.py` показал:

- цикл `for epoch in range(1, args.epochs + 1)` без `break`/`return`;
- forward под `torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)`;
- `answer_loss = F.cross_entropy(logits, answers)`;
- `retrieval_loss = F.cross_entropy(retrieval_scores.float(), answer_positions)`;
- `aux_value_loss = F.cross_entropy(model.head(model.norm(model.value_emb.weight)), torch.arange(args.n_values, device=device))`;
- `loss = answer_loss + retrieval_weight * retrieval_loss + aux_value_weight * aux_value_loss`;
- `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`;
- блок selection-probe/best-checkpoint выполняется только при `args.use_best and epoch % args.acc_every == 0`;
- progress-строка печатается при `epoch == 1 or epoch % 100 == 0`;
- после цикла: `if args.use_best and best_state is not None: model.load_state_dict(best_state)`.

[ВЫВОД] Поэтому при `epochs=25` и `acc_every=250` checkpoint и `history` не могут быть созданы; отсутствие `.pt` в такой конфигурации ожидаемо и не является признаком сбоя.

[ВЫВОД] Скаляры в report характеризуют **восстановленный best checkpoint**, а не модель на последней epoch, потому что `load_state_dict(best_state)` (строка 371) выполняется до чтения `model.recency_scale`/`score_scale`/`memory_gate` (строки 405–408).

### 3.5. Критерий PASS в коде

[ФАКТ] `grep` показал в `experiments/train_recall.py`:

```text
444: real["answer_acc"] > 0.90
447: and shuffled["answer_acc"] < chance * 2
448: and masked["answer_acc"] < chance * 2
453: report["control_status"] = "PASS" if passed else "FAIL"
```

[ФАКТ] Answer chance равен `1/64 = 0.015625`; порог controls равен `2 * chance = 0.03125`.

[ВЫВОД] Пороги для S=262144 не менялись и совпадают с протоколом S=131072.

---

## 4. Научные результаты

### 4.1. S=131072: трёх-seed independent PASS (подтверждён повторно)

[ФАКТ] Конфигурация: `n_keys=512`, `n_values=64`, `n_segments=131072`, `overwrite_p=0.3`, `key_dim=32`, `dim=128`, `batch_size=32`, `epochs=12000`, `lr=0.0003`, `hard_eval=true`, `eval_batch_size=16`, `use_best=true`, `acc_every=250`, `acc_n=512`, `aux_value_weight=0.01`, `weight_decay=0.0001`.

[ФАКТ] Independent eval каждого best checkpoint на 16384 эпизодах:

| Seed | Best epoch | Checkpoint sel_acc | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Статус |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 9750 | 0.990234375 | 0.9840087890625 | 0.9893798828125 | 0.01611328125 | 0.00079345703125 | 0.0142822265625 | PASS |
| 5 | 9750 | 0.98046875 | 0.9788818359375 | 0.99102783203125 | 0.015625 | 0.0009765625 | 0.01666259765625 | PASS |
| 6 | 10000 | 0.97265625 | 0.95849609375 | 0.96026611328125 | 0.01544189453125 | 0.0006103515625 | 0.015380859375 | PASS |

[ФАКТ] Checkpoint-файлы seeds 4–6 существуют, каждый 238999 байт, ключи `epoch`, `sel_acc`, `state_dict`, по 16 тензоров в `state_dict`.

[ФАКТ] В training history этих runs `sel_acc` к epoch 12000 деградировала до 0.029296875 (seed 4), 0.0234375 (seed 5), 0.060546875 (seed 6).

[ВЫВОД] Использование `--use-best` для large-S обязательно; финальная модель на 12000 epochs непригодна.

### 4.2. S=262144: technical probe

[ФАКТ] Probe: `n_segments=262144`, `seed=7`, `epochs=25`, `batch_size=32`, `aux_value_weight=0.01`, `hard_eval=true`, `eval_batch_size=16`, `eval_batches=64`, `use_best=true`.

[ФАКТ] Probe завершился, сохранил `results/probe_s262144_aux001_seed7.json` (1342 байта), без traceback и без CUDA OOM в присланном логе.

[ФАКТ] В probe JSON `history=[]`, `best_acc=0.0`, `best_epoch=0`, `escape_epoch=null`, `control_status=FAIL`, `memory_gate=6.978788587730378e-05`, `recency_scale=0.007494939025491476`, `score_scale=1.0060869455337524`.

[ФАКТ] Checkpoint `results/probe_s262144_aux001_seed7.pt` не был создан.

[ВЫВОД] Probe подтверждает только технический запуск режима S=262144 с `batch_size=32` на CUDA без OOM. Его метрики не являются тестом качества.

### 4.3. S=262144: двух-seed independent FAIL

[ФАКТ] Конфигурация обоих full runs идентична §3.1 при `n_segments=262144` и `epochs=12000`; отличался только seed.

[ФАКТ] Training-логи обоих runs дошли до epoch 12000 и напечатали `восстановлен best checkpoint`.

[ФАКТ] Independent eval каждого best checkpoint на 16384 эпизодах (`hard_eval=true`, `eval_batch_size=16`, `eval_batches=1024`):

| Seed | Best epoch | Checkpoint sel_acc | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Статус |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 7 | 1250 | 0.166015625 | 0.18865966796875 | 0.5360107421875 | 0.01617431640625 | 0.001220703125 | 0.01593017578125 | FAIL |
| 8 | 1250 | 0.142578125 | 0.156982421875 | 0.452392578125 | 0.01690673828125 | 0.0009765625 | 0.01556396484375 | FAIL |

[ФАКТ] Внутренние training eval тех же checkpoints дали real answer 0.186523 (seed 7) и 0.156250 (seed 8), real retrieval 0.524414 и 0.450195.

[ВЫВОД] Independent и внутренние оценки согласуются по качественному выводу: FAIL по real-порогам при чистых controls.

[ВЫВОД] Корректная формулировка результата:

> PHEF при `n_keys=512`, `n_values=64`, `n_segments=262144`, `overwrite_p=0.3`, `key_dim=32`, `dim=128`, `batch_size=32`, `lr=3e-4`, `weight_decay=1e-4`, `aux_value_weight=0.01`, hard retrieval, best-checkpoint по `sel_acc` и 12000 epochs **не проходит** segmented key–value recall на seeds 7 и 8. Независимая оценка каждого best checkpoint на 16384 эпизодах дала real answer 0.156982–0.188660 и real retrieval 0.452393–0.536011; shuffled и masked answer controls остались около chance 1/64, то есть модель не эксплуатирует leakage.

[НЕИЗВЕСТНО] Результат не доказывает недостижимость S=262144 при других `epochs`, lr, schedule, `dim`, `key_dim`, `batch_size`, `aux_value_weight`, других seeds или иной архитектуре.

### 4.4. Сравнение скаляров best checkpoint

[ФАКТ] Прямая сверка сохранённых training JSON:

| Поле | S=131072 seed 4 | S=262144 seed 7 | S=262144 seed 8 |
|---|---:|---:|---:|
| `best_epoch` | 9750 | 1250 | 1250 |
| `best_acc` | 0.990234375 | 0.166015625 | 0.142578125 |
| `escape_epoch` | 500 | 500 | 500 |
| `escape_threshold` | 0.03125 | 0.03125 | 0.03125 |
| `answer_acc` | 0.978515625 | 0.1865234375 | 0.15625 |
| `retrieval_acc` | 0.984375 | 0.5244140625 | 0.4501953125 |
| `memory_gate` | 0.8574084639549255 | 0.098076231777668 | 0.09644930809736252 |
| `recency_scale` | 2.2234997749328613 | 0.37576135993003845 | 0.3684224784374237 |
| `score_scale` | 3.6119940280914307 | 1.6401174068450928 | 1.6366032361984253 |
| `tie_rate` | 0.0625 | 0.0 | 0.0 |
| `control_status` | PASS | FAIL | FAIL |

[ФАКТ] У всех трёх runs `history_len=48` и escape на epoch 500.

[ФАКТ] `sel_acc` на epoch 1250: 0.208984375 (S=131072 seed 4), 0.166015625 (seed 7), 0.142578125 (seed 8).

[ВЫВОД] Ранняя динамика до epoch 1250 у всех трёх похожа. Расхождение возникает позже: при S=131072 рост продолжался до `sel_acc≈0.99` на epoch 9750, при S=262144 максимум остался на epoch 1250 и далее selection-оценки опускались до около-chance.

[ВЫВОД] Различие скаляров best checkpoint между S=131072 и S=262144 воспроизвелось на двух независимых seeds, значения seeds 7 и 8 близки между собой.

[НЕИЗВЕСТНО] Причинное направление. Низкий `memory_gate` может быть причиной слабого retrieval, следствием недостигнутого режима или корреляцией с ранним best epoch. Это [НЕИЗВЕСТНО] до различающего измерения.

### 4.5. Наблюдения по training-логу S=262144

[ФАКТ] В логе seed 7 на epochs 2300–4400 `total` loss принимал значения 32.1083, 89.5723, 111.8441, 255.0226, 303.6765, 431.4607, 559.4807, 623.2484, при `answer` около 4.11–4.22, то есть рост давал именно `retrieval` компонент.

[ФАКТ] В логе seed 7 на epochs 10900–12000 `total` вернулся к значениям около 16.36–16.89.

[ФАКТ] В логе seed 8 на epochs 400–1200 `retrieval_acc` достигал 0.367188, 0.460938, 0.492188, затем к epoch 12000 показанные значения были 0.000000–0.015625.

[НЕИЗВЕСТНО] Причина этой немонотонности. Нельзя называть её divergence, нестабильностью bf16, влиянием clip_grad_norm, gate collapse или иным механизмом без целевого измерения.

---

## 5. Опровергнутые и уточнённые утверждения

### 5.1. Уточнения, сделанные в этой сессии

[ФАКТ] Утверждение «отсутствие checkpoint после probe указывает на нештатное завершение» было ошибочным: при `epochs=25` и `acc_every=250` best-checkpoint блок не может выполниться.

[ФАКТ] Утверждение «лог с единственной строкой `epoch 1` доказывает остановку после epoch 1» было ошибочным: progress печатается только при `epoch == 1 or epoch % 100 == 0`.

[ВЫВОД] Оба утверждения исправлены на основании прочитанного кода. Не воскрешать их.

### 5.2. Не воскрешать без новых замеров (из предыдущих хендоффов)

1. «В S=65536 нет тайбрейкера» — опровергнуто: `recency_scale` обучается.
2. «Head принципиально не может декодировать value embeddings» — опровергнуто fresh-head test.
3. «LayerNorm — причина answer около 0.50» — опровергнуто.
4. «Gate — единственная причина около 0.50» — опровергнуто.
5. «Высокий shuffled score доказывает query leakage» — опровергнуто: fixed points старого shuffle объясняли завышение.
6. «12000 epochs обязательно улучшает large-S» — опровергнуто: поздние epochs деградируют.
7. «Мягкое чтение — адекватный selection signal при large S» — не подтверждено.
8. «Коллапс bf16 неизбежен около S=4096–16384» — не подтверждено.

### 5.3. Что запрещено заявлять о S=262144 сейчас

Все следующие утверждения — **[НЕИЗВЕСТНО]**:

- «S=262144 принципиально недостижим»;
- «нужно увеличить `dim`»;
- «нужно снизить lr»;
- «нужен learning rate schedule»;
- «`aux_value_weight=0.01` недостаточно»;
- «нужно изменить `batch_size`»;
- «проблема в bf16»;
- «проблема в `retrieval_weight`»;
- «gate схлопнулся и это причина»;
- «нужно просто больше epochs» — это гипотеза для §7, а не факт.

---

## 6. Системная стабильность

[ФАКТ] В journal текущей загрузки многократно зафиксированы события Intel iGPU в интервале 11:18–11:30 28 августа:

```text
i915 0000:00:02.0: [drm] Resetting rcs0 for preemption time out
i915 0000:00:02.0: [drm] *ERROR* GT0: rcs0 reset request timed out
i915 0000:00:02.0: [drm] GPU HANG: ecode 12:1:84dffffb, in Renderer [8655]
i915 0000:00:02.0: [drm] Resetting rcs0 for stopped heartbeat on rcs0
i915 0000:00:02.0: [drm] Renderer[8655] context reset due to GPU hang
i915 0000:00:02.0: [drm] Xorg[5202] context reset due to GPU hang
```

[ФАКТ] В отфильтрованном выводе journal не было строк `NVRM Xid`, PCIe AER fatal, MCE, `hardware error`, `kernel panic` или `oom-kill`.

[ФАКТ] После старта probe около 14:00 и full runs 14:16–15:44 и 16:22–17:44 новых отфильтрованных i915-строк в присланных выводах не появилось.

[ФАКТ] Часть проверок journal не выполнилась из-за `sudo: требуется указать пароль`; для этих интервалов kernel-состояние не проверено.

[ФАКТ] Наблюдённые GPU-срезы во время работы: 5159 MiB / 52°C / ~93–95 W при training; 2805–2885 MiB / 44°C / ~61–64 W при independent eval; 15 MiB / 37–47°C / ~21–27 W после завершения.

[НЕИЗВЕСТНО] Peak VRAM для S=262144 не измерялся; приведённые значения — единичные срезы, не пик.

### 6.1. Практическое правило

- Не запускать PHEF одновременно с DIEL или другим CUDA workload.
- Не запускать тяжёлые диагностические `grep -R` во время обучения.
- Закрывать тяжёлые Firefox вкладки, видео и WebGL во время run.
- Не использовать UI Saby во время run; не останавливать Saby-сервисы без отдельного подтверждения.
- Не менять BIOS, GRUB, kernel, i915, Xorg или NVIDIA driver без отдельной диагностики и явного подтверждения.

### 6.2. Если система падает

После перезагрузки, **до запуска нового обучения**:

```bash
sudo journalctl -b -1 -k --no-pager -o short-precise | tail -250
```

Не запускать новый run до анализа этого вывода.

---

## 7. ГЛАВНАЯ ПРОВЕРКА: epochs=24000, seed 7

### 7.1. Гипотеза и что тест различает

Гипотеза H1: при S=262144 нужный режим адресации достижим, но требует больше оптимизационных шагов, чем 12000 epochs.

Гипотеза H2: при S=262144 в этой конфигурации режим не достигается независимо от длительности, и плато после epoch 1250 устойчиво.

[ФАКТ] Основание для постановки: при S=131072 best epochs были 9750, 9750, 10000; при S=262144 оба best epoch оказались 1250.

[ВЫВОД] Тест меняет **только** `--epochs` с 12000 на 24000 при том же seed 7 и всех прочих неизменных параметрах, поэтому его результат интерпретируем однозначно в терминах H1 против H2.

[НЕИЗВЕСТНО] Длительность run. Предыдущий 12000-epoch run S=262144 шёл примерно с 14:16 до 15:44; экстраполировать это на 24000 epochs нельзя.

### 7.2. Проверка свободных имён (read-only)

[ФАКТ] На 28 августа 18:40 все четыре пути были свободны: `results/cap_k512_s262144_aux001_seed7_e24000.json`, `results/cap_k512_s262144_aux001_seed7_e24000.pt`, `results/cap_k512_s262144_aux001_seed7_e24000_eval16384.json`, `/tmp/probe_logs/phef_s262144_aux001_seed7_e24000_train.log`.

Проверить повторно перед запуском:

```bash
cd ~/qubitrino-projects/agi/phef && for f in results/cap_k512_s262144_aux001_seed7_e24000.json results/cap_k512_s262144_aux001_seed7_e24000.pt results/cap_k512_s262144_aux001_seed7_e24000_eval16384.json /tmp/probe_logs/phef_s262144_aux001_seed7_e24000_train.log; do if test -e "$f"; then echo "EXISTS $f"; ls -l "$f"; else echo "MISSING $f"; fi; done
```

### 7.3. Команда запуска (write — требует подтверждения)

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/train_recall.py --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 32 --dim 128 --batch-size 32 --epochs 24000 --lr 3e-4 --weight-decay 1e-4 --retrieval-weight 1.0 --aux-value-weight 0.01 --hard-eval --eval-batch-size 16 --eval-batches 64 --use-best --acc-every 250 --acc-n 512 --seed 7 --out results/cap_k512_s262144_aux001_seed7_e24000.json --best-out results/cap_k512_s262144_aux001_seed7_e24000.pt > /tmp/probe_logs/phef_s262144_aux001_seed7_e24000_train.log 2>&1 &
```

Создаваемые/перезаписываемые артефакты:

- `/tmp/probe_logs/phef_s262144_aux001_seed7_e24000_train.log`
- `results/cap_k512_s262144_aux001_seed7_e24000.json`
- `results/cap_k512_s262144_aux001_seed7_e24000.pt`

### 7.4. Мониторинг (read-only)

```bash
cd ~/qubitrino-projects/agi/phef && echo "=== PROCESS ===" && pgrep -af "train_recall.py.*seed7_e24000" || true && echo "=== LAST EPOCH ===" && grep -E "^epoch " /tmp/probe_logs/phef_s262144_aux001_seed7_e24000_train.log | tail -1 || true && echo "=== LAST SEL_ACC ===" && grep -E "^epoch .* sel_acc " /tmp/probe_logs/phef_s262144_aux001_seed7_e24000_train.log | tail -8 || true && echo "=== MAX SEL_ACC ===" && grep -E "^epoch .* sel_acc " /tmp/probe_logs/phef_s262144_aux001_seed7_e24000_train.log | tail -1 || true && echo "=== LOG TAIL ===" && tail -25 /tmp/probe_logs/phef_s262144_aux001_seed7_e24000_train.log 2>&1 || true && echo "=== GPU ===" && nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader 2>&1
```

### 7.5. Признаки штатного завершения

[ФАКТ] Штатное окончание подтверждается одновременно:

- `pgrep` больше не показывает процесс обучения;
- лог содержит `восстановлен best checkpoint`;
- лог содержит блок `=== CONTROLS ===`;
- лог содержит `control status`;
- лог содержит `сохранено: results/...json`;
- соответствующие JSON и `.pt` существуют.

### 7.6. Independent eval (write — требует подтверждения)

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/eval_recall_checkpoint.py --checkpoint results/cap_k512_s262144_aux001_seed7_e24000.pt --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 32 --dim 128 --eval-batch-size 16 --eval-batches 1024 --seed 7 --out results/cap_k512_s262144_aux001_seed7_e24000_eval16384.json > /tmp/probe_logs/phef_s262144_aux001_seed7_e24000_eval16384.log 2>&1 &
```

PASS criterion без изменения порогов:

```text
real answer > 0.90
real retrieval > 0.90
shuffled answer < 0.03125
masked answer < 0.03125
```

### 7.7. Что проверять в eval JSON

- процесс eval завершён;
- лог содержит `saved: results/...eval16384.json`;
- JSON существует и имеет `hard_eval: true`;
- `config` соответствует точному run, включая `n_segments=262144` и `seed=7`;
- число episodes в `real`, `shuffled`, `masked` равно 16384;
- критерий PASS применён без изменения порогов.

---

## 8. Варианты после главной проверки

### 8.1. Если epochs=24000 даёт independent PASS

[ВЫВОД] Тогда H1 подтверждена для seed 7, и следующие шаги:

1. Репликация на seeds 8 и 9 при `epochs=24000`, полностью идентичная конфигурация, отдельные имена артефактов.
2. Independent eval каждого checkpoint на 16384 эпизодах.
3. Commit четырёх-шести JSON белым списком; `.pt` не коммитить.
4. Отдельно зафиксировать, что при S=262144 требуется больше epochs, чем при S=131072, с указанием фактических best epochs.

[НЕИЗВЕСТНО] Понадобится ли S=524288 ещё больше epochs — переносить нельзя, нужен собственный probe и run.

### 8.2. Если epochs=24000 снова даёт FAIL и best epoch остаётся ранним

[ВЫВОД] Тогда H1 ослаблена для этой конфигурации, и осмысленно проверять по одному фактору за раз. Идеи, каждая — отдельный run с новым именем и отдельным подтверждением:

1. **`dim`**: `--dim 192` или `--dim 256` при прочих неизменных. Мотивация: `dim=128` могло ограничивать ёмкость представления при 262144 сегментах. Ожидаемый эффект — рост peak VRAM, нужен свой probe.
2. **`key_dim`**: `--key-dim 48` или `--key-dim 64`. Мотивация: при 512 ключах и 262144 сегментах различимость адресации зависит от размерности ключа.
3. **`lr`**: `--lr 1e-4` или `--lr 2e-4`. Мотивация: наблюдённая немонотонность retrieval loss и раннее плато. Проверять только один вариант за run.
4. **`batch_size`**: `--batch-size 64`. Мотивация: меньше шума градиента на шаг. Требует probe на OOM.
5. **`aux_value_weight`**: `--aux-value-weight 0.03` или `0.1`. Мотивация: на S=65536 все три веса давали PASS; для S=262144 это не проверялось.
6. **`retrieval_weight`**: `--retrieval-weight 0.5` или `2.0`. Мотивация: в логах рост `total` шёл именно за счёт retrieval компонента.
7. **`acc_every`**: `--acc-every 100`. Мотивация: при 250 селекционные проверки редкие, и best checkpoint мог быть пропущен между ними. Это не изменение обучения, а изменение выборки best.
8. **`escape_factor`**: оставить 2.0; менять только если появится отдельная причина.

[ВЫВОД] Пункт 7 стоит рассмотреть рано и дешево: он не меняет optimization, только частоту фиксации best checkpoint. Если истинный максимум `sel_acc` находился между кратными 250, текущий протокол его не сохранял.

[НЕИЗВЕСТНО] Насколько велик этот эффект. Требуется измерение, а не оценка.

### 8.3. Диагностические тесты без полного обучения

[ВЫВОД] Дешевле полного run и потенциально информативнее:

1. **Fresh-head test на checkpoint S=262144**: обучить линейную head на фиксированных `value_emb` из `cap_k512_s262144_aux001_seed7.pt` и проверить, декодируются ли 64 value embeddings. Ранее такой тест на S=65536 checkpoint отделил проблему head от проблемы retrieval.
2. **Разделение retrieval и answer**: измерить на существующем checkpoint, какова доля эпизодов с корректным retrieval при некорректном answer, и наоборот. Real retrieval 0.452–0.536 при real answer 0.157–0.189 намекает на асимметрию, но точное распределение [НЕИЗВЕСТНО].
3. **Проверка `tie_rate`**: при S=131072 `tie_rate=0.0625`, при S=262144 `tie_rate=0.0`. Понять, что именно измеряет `last_tie_rate` в коде, прежде чем интерпретировать.
4. **Влияние `memory_gate`**: измерить метрики при принудительно увеличенном gate на существующем checkpoint. Это тест, а не исправление; ранее на S=65536 усиление gate не решало проблему.

[ВЫВОД] Пункты 1 и 2 предпочтительны как первые, потому что используют уже существующие checkpoints и не требуют многочасового обучения.

### 8.4. Промежуточный масштаб

[ВЫВОД] Если S=262144 остаётся FAIL, полезно установить, где именно граница: `n_segments=196608` при неизменной остальной конфигурации и 12000 epochs.

[НЕИЗВЕСТНО] Пройдёт ли этот масштаб. Значение важно тем, что отделяет «резкий обрыв между 131072 и 262144» от «плавной деградации».

### 8.5. Что делать с i915

[ВЫВОД] Независимо от научного результата, наиболее безопасный режим для многочасовых run — headless/SSH с минимальной графической нагрузкой на Intel iGPU либо перевод desktop на RTX 4090. Это рекомендация, не выполненное изменение.

---

## 9. Полезные шаблоны команд

### 9.1. Read-only сверка в новом чате

```bash
cd ~/qubitrino-projects/agi && git status --short | head -20 && git log --oneline -8 && git branch -vv && git diff --check && cd phef && for f in phef/model.py phef/world.py experiments/train_recall.py experiments/eval_recall_checkpoint.py; do /home/linux/qubitrino-projects/agi/venv/bin/python -c "import ast; ast.parse(open('$f').read()); print('ok $f')"; done
```

### 9.2. Сверка результатов S=262144

```bash
cd ~/qubitrino-projects/agi/phef && /home/linux/qubitrino-projects/agi/venv/bin/python -c "
import json
for p in ['results/cap_k512_s131072_aux001_seed4.json','results/cap_k512_s262144_aux001_seed7.json','results/cap_k512_s262144_aux001_seed8.json']:
    d=json.load(open(p))
    h=d.get('history') or []
    print('===',p)
    print({k:d.get(k) for k in ['best_epoch','best_acc','escape_epoch','answer_acc','retrieval_acc','memory_gate','recency_scale','score_scale','tie_rate','control_status']})
    print('history_len=',len(h),'max_sel=',max([x['sel_acc'] for x in h], default=None))
"
```

### 9.3. Системная готовность перед run

```bash
echo "=== CUDA PROCESSES ===" && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 && echo "=== PHEF / DIEL ===" && pgrep -af "train_recall.py|eval_recall_checkpoint.py|DIEL|diel" || true && echo "=== KERNEL WARNINGS ===" && sudo journalctl -b -k --no-pager | grep -iE "i915.*(GPU HANG|reset|heartbeat|context reset)|nvrm.*xid|aer.*(error|fatal)|mce|hardware error|kernel panic|oom-kill" | tail -100 || true
```

### 9.4. Git commit результатов

```bash
cd ~/qubitrino-projects/agi && git add ЯВНЫЙ_СПИСОК_ФАЙЛОВ && git status --short && git diff --cached --check && git diff --cached --stat
```

Затем отдельно, после подтверждения:

```bash
cd ~/qubitrino-projects/agi && git commit -m "СООБЩЕНИЕ"
```

```bash
cd ~/qubitrino-projects/agi && git push origin main
```

---

## 10. Рекомендуемые файлы для нового чата

Минимально приложить:

1. Этот handoff.
2. `phef/phef/model.py`.
3. `phef/phef/world.py`.
4. `phef/experiments/train_recall.py`.
5. `phef/experiments/eval_recall_checkpoint.py`.
6. `phef/results/cap_k512_s131072_aux001_seed4.json` и `..._seed4_eval16384.json`.
7. `phef/results/cap_k512_s262144_aux001_seed7.json` и `..._seed7_eval16384.json`.
8. `phef/results/cap_k512_s262144_aux001_seed8.json` и `..._seed8_eval16384.json`.
9. При наличии — JSON нового run `epochs=24000`.

Полезные исторические документы:

- `docs/handoffs/20_PHEF-handoff-S65536-checkpoint-shuffle-control.md`.
- `docs/handoffs/21_PHEF_handoff_S65536_PASS.md`.
- `docs/handoffs/22_PHEF_handoff_S131072_three_seed_PASS.md`.

Не обязательно прикладывать: `train_recall.py.bak`; `/tmp`-логи; old smoke results; `logs/microtest/`; probe JSON.

---

## 11. Краткая стартовая инструкция новому чату

1. Прочитать этот документ целиком.
2. Не делать предположений; маркировать каждое утверждение [ФАКТ]/[ВЫВОД]/[НЕИЗВЕСТНО].
3. Выполнить только read-only сверку из §9.1 и §9.2 и дождаться вывода пользователя.
4. Проверить системную готовность из §9.3.
5. Если главная проверка `epochs=24000` ещё не запускалась — согласовать точную команду §7.3 и получить явное подтверждение.
6. Если она уже завершилась — сверить JSON, применить критерий PASS §7.6 без изменения порогов, затем выбрать вариант из §8.
7. Не коммитить новые результаты до независимой проверки и явного подтверждения пользователя.
8. Помнить: текущий подтверждённый статус — S=131072 PASS на seeds 4–6, S=262144 FAIL на seeds 7–8 при 12000 epochs.
