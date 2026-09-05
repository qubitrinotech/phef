# PHEF — самодостаточный хендофф 24: S=262144 epochs=24000 independent FAIL, план скрининга факторов 1–5 и комбинаций

Дата передачи: 28 августа 2026.
Репозиторий: `~/qubitrino-projects/agi`.
Рабочий каталог PHEF: `~/qubitrino-projects/agi/phef`.
Ветка: `main`.
Последний отправленный commit: `e77f1d4 phef: S=262144 epochs=24000 seed 7 still FAIL (best epoch stays 1250, clean controls)`.
Remote: `gitlab.qubitrinotech.ru:root/agi.git`.

---

## 0. ГЛАВНОЕ ПРАВИЛО — НЕ ГАДАТЬ

НЕ ГАДАТЬ. НЕ ФАНТАЗИРОВАТЬ. НЕ ПОДМЕНЯТЬ НЕИЗВЕСТНОЕ ПРАВДОПОДОБНОЙ ВЕРСИЕЙ.
Это правило важнее скорости, удобства и желания «помочь побыстрее».

### 0.1. Маркировка утверждений

Каждое содержательное утверждение обязательно помечать одной из меток:

- **[ФАКТ]** — прямо подтверждён выводом команды, логом, JSON, checkpoint metadata, git-выводом или иным первичным артефактом, показанным пользователем в этом чате.
- **[ВЫВОД]** — логическое следствие одного или нескольких подтверждённых фактов. Это не новый замер и не должно формулироваться как измеренная величина.
- **[НЕИЗВЕСТНО]** — данных недостаточно. Нельзя заменять пробел догадкой. Вместо этого запросить конкретную минимальную проверку.

Примеры корректности:

- «[ФАКТ] real answer=0.190430» — если это число есть в присланном JSON/логе.
- «[ВЫВОД] Seed 7 не проходит порог, потому что real answer < 0.90» — если эти числа измерены.
- «[НЕИЗВЕСТНО] Peak VRAM при dim=256 неизвестен» — пока нет отдельного измерения именно этого режима.

### 0.2. Граница доступа

Агент не имеет прямого доступа к машине пользователя, shell, Git, GPU, файлам, JSON, checkpoint `.pt`, конфигурациям и `/tmp`-логам.
Все команды выполняет пользователь и присылает полный вывод.
Нельзя говорить «в файле находится…», «GPU использует…», «процесс завершился…» без присланного подтверждения.

### 0.3. Не угадывать содержимое

Не угадывать содержимое: Python-кода; JSON и checkpoint `.pt`; конфигураций; системных журналов; `.gitignore`; логов обучения; статуса процессов; потребления VRAM, времени, throughput или температуры.
Если нужен фрагмент, запросить точную read-only команду и дождаться результата. Предпочитать:

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

Нельзя подтверждать параметры эксперимента по памяти, старому хендоффу или пересказу. Перед выводом о конкретном run сверять фактический `config` в целевом JSON этого run/eval.

### 0.5. Изменения только после согласия

До любого действия, которое меняет состояние, обязательно:

1. Показать точное планируемое действие.
2. Явно перечислить затрагиваемые файлы/сервисы/артефакты.
3. Для кода — показать точный diff до применения.
4. Получить явное подтверждение пользователя.
5. После действия запросить проверку фактического результата.

К действиям с изменением состояния относятся: создание/редактирование/удаление/перемещение файлов; `git add`, `git commit`, `git push`; запуск обучения/eval, создающий JSON/checkpoint/log; остановка или удаление системного сервиса; изменение BIOS, GRUB, kernel parameters, драйверов, Xorg.

Не маскировать команду изменения в длинном блоке и не действовать по неявному «ок».

Практика, подтверждённая в работе: перед каждым write-действием агент просил буквальную фразу подтверждения, например «Подтверждаю запуск exact full-training-команды S=262144 seed 7 epochs=24000». Эту практику сохранить.

### 0.6. Гипотеза, опровергнутая замером

Если гипотеза опровергнута — прямо так и написать. Не воскрешать её без новых данных.

### 0.7. Никакой экстраполяции

Нельзя переносить между конфигурациями: качество; устойчивость по seed; скорость; peak VRAM; время обучения; поведение loss; оптимальный epoch; надёжность controls.

Каждая новая конфигурация требует собственного запуска и независимого eval.

**Отдельно для этого хендоффа:** скрининговый run на 4000 epochs (§7) может только отбирать кандидатов. Заявить PASS по скринингу нельзя. Победитель обязан пройти полный run и independent eval на 16384 эпизодах с неизменными порогами (§8).

### 0.8. Числовые расхождения — не объяснять «на глаз»

Если числа расходятся, причина — [НЕИЗВЕСТНО], пока не проверена. Нельзя объявлять причиной «случайность», «влияние GPU», «утечку», «gate», «LayerNorm», «мало эпох», «перегрузку CPU» без измерения, которое различает объяснения.

### 0.9. Формат ответов и команд

- Отвечать по-русски и кратко.
- Команды давать готовыми для копирования.
- Предпочитать один блок и одну строку запуска для длинных команд.
- Не давать много тяжёлых команд одновременно.
- Для рекурсивного grep всегда ограничивать область и вывод; не сканировать большие каталоги во время обучения.

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
- Длинные команды с `LC_ALL=C` и `PYTHONPATH=.` давать одной строкой, без `\` и переносов.
- Учитывать текущий каталог: в Git-корне пути имеют префикс `phef/`, внутри `~/qubitrino-projects/agi/phef` — `results/...`, `experiments/...`, `phef/...`.
- [ФАКТ] Цепочка через `&&` прерывается на первом ненулевом коде: в мониторинге eval блок `=== GPU ===` не напечатался, потому что `ls` отсутствующего файла вернул ошибку. В мониторинговых блоках, где ожидается отсутствие файла, разделять шаги через `;`, а не `&&`.

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
- Видеовывод desktop: Intel UHD 770 через i915/X11.
- Kernel на момент прошлой системной диагностики: 7.0.0-30-generic Ubuntu.
- NVIDIA driver на момент прошлой системной диагностики: 595.84, CUDA по `nvidia-smi`: 13.2.
- Hostname в приглашении shell: `linux-desktop2`; пользователь `linux`.

[ФАКТ] 28 августа: `git status --short` показывал только untracked записи (`??`) и ни одного изменённого tracked файла; `git diff --check` был пуст.
[ФАКТ] AST-разбор `phef/model.py`, `phef/world.py`, `experiments/train_recall.py`, `experiments/eval_recall_checkpoint.py` прошёл успешно (`ok` по каждому).

---

## 2. Отправленные commits

| Commit | Содержание |
|---|---|
| `8e18019` | CPU-генерация slots в world_gpu |
| `150e9c7` | `batch()` возвращает `(slots, codebook)` |
| `d440829` | pointer-doubling в `_sample_slots` |
| `84dda15` | первый PASS малого масштаба: hard-eval, 1000 epochs |
| `3c712b6` | S=65536 PASS: no-fixed-point shuffle, hard-selection/checkpoint, auxiliary decoding loss, evaluator, seed-0 артефакты |
| `16e4cf0` | S=65536: воспроизводимые JSON для seed 1–2 |
| `d81b3c6` | S=131072: train/eval JSON для seeds 4–6 |
| `d41245c` | S=262144: train/eval JSON для seeds 7–8, отрицательный результат при 12000 epochs |
| `e77f1d4` | S=262144 epochs=24000 seed 7: снова FAIL, best epoch остался 1250 |

[ФАКТ] `git push origin main` для последнего commit завершился успешно: `d41245c..e77f1d4  main -> main`.
[ФАКТ] Commit `e77f1d4` содержал ровно 2 файла, 472 insertions (436 + 36), оба `create mode 100644`:

- `phef/results/cap_k512_s262144_aux001_seed7_e24000.json`
- `phef/results/cap_k512_s262144_aux001_seed7_e24000_eval16384.json`

[ФАКТ] Перед commit `git diff --check` и `git diff --cached --check` дали пустой вывод.
[ФАКТ] Все `.pt` checkpoints, handoff-документы, `train_recall.py.bak`, `logs/microtest/`, probe JSON и исторические результаты остались untracked и намеренно не коммитились.
[ФАКТ] После push: `git log --oneline -3` показал `e77f1d4 (HEAD -> main, origin/main)`, `git branch -vv` — `* main e77f1d4 [origin/main]`.

---

## 3. Проверенный метод, код и CLI

### 3.1. Базовый режим large-S

```
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
[ФАКТ] `--best-out` — «путь к `.pt` лучшего checkpoint по sel_acc; без него checkpoint не сохраняется».
[ФАКТ] `--escape-factor` — «порог выхода с плато в кратности chance (2.0 => 2/n_values)».
[ФАКТ] Default `--epochs` равен 1500.
[ФАКТ] В CLI **нет** флагов для learning rate schedule, warmup, значения `clip_grad_norm`, оптимизатора и dropout.
[ВЫВОД] Любая проверка schedule/warmup/clip требует diff в `train_recall.py` — показать diff до применения и получить согласие (§0.5).

### 3.3. CLI `experiments/eval_recall_checkpoint.py`

[ФАКТ] `--help` подтвердил: обязательные `--checkpoint` и `--out`; опциональные `--n-keys`, `--n-values`, `--n-segments`, `--overwrite-p`, `--key-dim`, `--dim`, `--eval-batch-size`, `--eval-batches`, `--seed`, `--cpu`.
[ВЫВОД] Evaluator принимает `--key-dim` и `--dim`, поэтому eval checkpoints с изменённой архитектурой возможен, но эти значения обязаны совпадать с training-config того checkpoint.

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

[ВЫВОД] При `epochs=25` и `acc_every=250` checkpoint и history не могут быть созданы; отсутствие `.pt` в такой конфигурации ожидаемо и не является признаком сбоя.
[ВЫВОД] Скаляры в report характеризуют восстановленный best checkpoint, а не модель на последней epoch, потому что `load_state_dict(best_state)` (строка 371) выполняется до чтения `model.recency_scale`/`score_scale`/`memory_gate` (строки 405–408).

### 3.5. Критерий PASS в коде

[ФАКТ] `grep` показал в `experiments/train_recall.py`:

```
444: real["answer_acc"] > 0.90
447: and shuffled["answer_acc"] < chance * 2
448: and masked["answer_acc"] < chance * 2
453: report["control_status"] = "PASS" if passed else "FAIL"
```

[ФАКТ] Answer chance равен 1/64 = 0.015625; порог controls равен 2 * chance = 0.03125.
[ФАКТ] В логе run S=262144 напечатано `retrieval chance = 0.000004`, что соответствует 1/262144.
[ВЫВОД] Пороги менять запрещено во всех дальнейших экспериментах.

---

## 4. Научные результаты

### 4.1. S=131072: трёх-seed independent PASS

[ФАКТ] Конфигурация: `n_keys=512, n_values=64, n_segments=131072, overwrite_p=0.3, key_dim=32, dim=128, batch_size=32, epochs=12000, lr=0.0003, hard_eval=true, eval_batch_size=16, use_best=true, acc_every=250, acc_n=512, aux_value_weight=0.01, weight_decay=0.0001`.

| Seed | Best epoch | Checkpoint sel_acc | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Статус |
|---|---|---|---|---|---|---|---|---|
| 4 | 9750 | 0.990234375 | 0.9840087890625 | 0.9893798828125 | 0.01611328125 | 0.00079345703125 | 0.0142822265625 | PASS |
| 5 | 9750 | 0.98046875 | 0.9788818359375 | 0.99102783203125 | 0.015625 | 0.0009765625 | 0.01666259765625 | PASS |
| 6 | 10000 | 0.97265625 | 0.95849609375 | 0.96026611328125 | 0.01544189453125 | 0.0006103515625 | 0.015380859375 | PASS |

[ФАКТ] Checkpoint-файлы seeds 4–6 существуют, каждый 238999 байт, ключи `epoch`, `sel_acc`, `state_dict`, по 16 тензоров в `state_dict`.
[ФАКТ] В training history этих runs sel_acc к epoch 12000 деградировала до 0.029296875 (seed 4), 0.0234375 (seed 5), 0.060546875 (seed 6).
[ВЫВОД] Использование `--use-best` для large-S обязательно; финальная модель на 12000 epochs непригодна.

### 4.2. S=262144, epochs=12000: двух-seed independent FAIL

[ФАКТ] Конфигурация идентична §3.1 при `n_segments=262144` и `epochs=12000`; отличался только seed.

| Seed | Best epoch | Checkpoint sel_acc | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Статус |
|---|---|---|---|---|---|---|---|---|
| 7 | 1250 | 0.166015625 | 0.18865966796875 | 0.5360107421875 | 0.01617431640625 | 0.001220703125 | 0.01593017578125 | FAIL |
| 8 | 1250 | 0.142578125 | 0.156982421875 | 0.452392578125 | 0.01690673828125 | 0.0009765625 | 0.01556396484375 | FAIL |

### 4.3. ГЛАВНЫЙ НОВЫЙ РЕЗУЛЬТАТ: S=262144, epochs=24000, seed 7 — independent FAIL

[ФАКТ] Config из `results/cap_k512_s262144_aux001_seed7_e24000.json`: `n_keys=512, n_values=64, n_segments=262144, overwrite_p=0.3, key_dim=32, hard_eval=True, dim=128, epochs=24000, batch_size=32, eval_batch_size=16, eval_batches=64, lr=0.0003, weight_decay=0.0001, retrieval_weight=1.0, aux_value_weight=0.01, use_best=True, acc_every=250, acc_n=512, escape_factor=2.0, no_memory=False, seed=7, cpu=False`.
[ФАКТ] Run стартовал 28 августа около 19:00, завершился штатно; JSON 7397 байт (21:42), `.pt` 239153 байта (19:09).
[ФАКТ] Лог содержал все признаки штатного завершения: «восстановлен best checkpoint: epoch 1250 sel_acc 0.1680», блок `=== CONTROLS ===`, строку `control status`, строку «сохранено: results/cap_k512_s262144_aux001_seed7_e24000.json».
[ФАКТ] Скаляры JSON: `best_epoch=1250`, `best_acc=0.16796875`, `escape_epoch=500`, `answer_acc=0.19921875`, `retrieval_acc=0.5478515625`, `memory_gate=0.09802315384149551`, `recency_scale=0.3758833706378937`, `score_scale=1.640737533569336`, `tie_rate=0.0`, `control_status=FAIL`.
[ФАКТ] `history_len=96`, `max_sel=0.16796875` — равен `best_acc`.
[ФАКТ] Checkpoint: `keys=['epoch','sel_acc','state_dict']`, `epoch=1250`, `sel_acc=0.16796875`, 16 тензоров.
[ФАКТ] Internal controls в логе: real answer 0.199219, real retrieval 0.547852, shuffled answer 0.020508, masked answer 0.009766, control status FAIL.
[ФАКТ] Independent eval (`results/cap_k512_s262144_aux001_seed7_e24000_eval16384.json`, 885 байт, 22:26): `hard_eval=true`, `checkpoint_epoch=1250`, `checkpoint_sel_acc=0.16796875`, `answer_chance=0.015625`, real `answer_acc=0.1904296875` / `retrieval_acc=0.53778076171875` / `episodes=16384`, shuffled `answer_acc=0.0167236328125` / `retrieval_acc=0.00128173828125` / `episodes=16384`, masked `answer_acc=0.01605224609375` / `retrieval_acc=null` / `episodes=16384`.
[ФАКТ] Config eval: `n_keys=512, n_values=64, n_segments=262144, overwrite_p=0.3, key_dim=32, dim=128, eval_batch_size=16, eval_batches=1024, seed=7, cpu=false`.

[ВЫВОД] Independent статус — FAIL: real answer 0.190430 и real retrieval 0.537781 много ниже 0.90, при этом shuffled 0.016724 и masked 0.016052 ниже порога 0.03125, то есть controls чистые и leakage не эксплуатируется.
[ВЫВОД] Внутренний и independent eval согласуются: 0.199219/0.547852 против 0.190430/0.537781.
[ВЫВОД] **Гипотеза H1 («при S=262144 режим достижим, но нужно больше оптимизационных шагов») опровергнута для seed 7 в этой конфигурации.** Удвоение epochs с 12000 до 24000 не сдвинуло best epoch (1250 в обоих runs) и не улучшило качество: independent real answer 0.188660 при 12000 против 0.190430 при 24000.
[ВЫВОД] Timestamp `.pt` 19:09 при завершении обучения в 21:42 означает, что за последние ~2,5 часа обучения checkpoint не перезаписывался ни разу.
[ВЫВОД] Скаляры best checkpoint воспроизвелись между двумя runs seed 7: `memory_gate` 0.098076 против 0.098023, `recency_scale` 0.375761 против 0.375883, `score_scale` 1.640117 против 1.640738.

[НЕИЗВЕСТНО] Результат не доказывает недостижимость S=262144 при других `key_dim`, `dim`, `lr`, `batch_size`, `retrieval_weight`, `aux_value_weight`, `acc_every`, learning rate schedule, других seeds или иной архитектуре.

### 4.4. Референсные кривые sel_acc на epochs 250–4000

[ФАКТ] Прямое чтение history из JSON:

```
S131072 seed4:
(250, 0.0254) (500, 0.0410) (750, 0.0352) (1000, 0.0605) (1250, 0.2090)
(1500, 0.2363) (1750, 0.2891) (2000, 0.3066) (2250, 0.4199) (2500, 0.4902)
(2750, 0.6367) (3000, 0.7500) (3250, 0.8203) (3500, 0.8418) (3750, 0.7012) (4000, 0.6113)

S262144 seed7 e24000:
(250, 0.0273) (500, 0.0723) (750, 0.0352) (1000, 0.1250) (1250, 0.1680)
(1500, 0.0742) (1750, 0.0156) (2000, 0.0176) (2250, 0.0234) (2500, 0.0195)
(2750, 0.0137) (3000, 0.0098) (3250, 0.0254) (3500, 0.0098) (3750, 0.0215) (4000, 0.0098)
```

[ФАКТ] При S=262144 в окне 1750–24000 sel_acc не превышала примерно 0.025.
[ВЫВОД] Разделение траекторий PASS и FAIL наступает уже к epoch 1500–2000: у S=131072 монотонный рост 0.209 → 0.236 → 0.289 → 0.307, у S=262144 обвал 0.168 → 0.074 → 0.016.
[ВЫВОД] Это основание для скрининга на 4000 epochs (§7): различающий сигнал появляется задолго до 12000.

### 4.5. Сравнение скаляров best checkpoint

[ФАКТ] Прямая сверка сохранённых training JSON:

| Поле | S=131072 seed 4 | S=262144 seed 7 e12000 | S=262144 seed 8 e12000 | S=262144 seed 7 e24000 |
|---|---|---|---|---|
| `best_epoch` | 9750 | 1250 | 1250 | 1250 |
| `best_acc` | 0.990234375 | 0.166015625 | 0.142578125 | 0.16796875 |
| `escape_epoch` | 500 | 500 | 500 | 500 |
| `escape_threshold` | 0.03125 | 0.03125 | 0.03125 | 0.03125 |
| `answer_acc` | 0.978515625 | 0.1865234375 | 0.15625 | 0.19921875 |
| `retrieval_acc` | 0.984375 | 0.5244140625 | 0.4501953125 | 0.5478515625 |
| `memory_gate` | 0.8574084639549255 | 0.098076231777668 | 0.09644930809736252 | 0.09802315384149551 |
| `recency_scale` | 2.2234997749328613 | 0.37576135993003845 | 0.3684224784374237 | 0.3758833706378937 |
| `score_scale` | 3.6119940280914307 | 1.6401174068450928 | 1.6366032361984253 | 1.640737533569336 |
| `tie_rate` | 0.0625 | 0.0 | 0.0 | 0.0 |
| `control_status` | PASS | FAIL | FAIL | FAIL |

[ФАКТ] `history_len` равен 48 у runs с 12000 epochs и 96 у run с 24000 epochs.
[НЕИЗВЕСТНО] Причинное направление. Низкий `memory_gate` может быть причиной слабого retrieval, следствием недостигнутого режима или корреляцией с ранним best epoch. Это [НЕИЗВЕСТНО] до различающего измерения.

### 4.6. Наблюдения по training-логу S=262144 epochs=24000

[ФАКТ] Окно epochs 2300–4000: total loss колебался 71.71–367.34 при answer стабильно 4.11–4.21, то есть весь рост давал retrieval-компонент (67.51–363.14).
[ФАКТ] retrieval_acc в этом окне: 0.000000–0.015625.
[ФАКТ] Окно epochs 9600–11300: total стабилен около 16.25–17.66, answer 4.09–4.33, retrieval 12.12–13.42, retrieval_acc 0.000000–0.007812.
[ФАКТ] Окно epochs 23300–24000: total 16.01–16.67, answer 4.09–4.47, retrieval 11.90–12.30.
[ВЫВОД] Взрыв retrieval loss локализован именно в retrieval-члене; answer-член на протяжении всего run остаётся около `ln(64) ≈ 4.16`, то есть на уровне неинформативного предсказания.
[НЕИЗВЕСТНО] Причина немонотонности. Нельзя называть её divergence, нестабильностью bf16, влиянием `clip_grad_norm`, gate collapse или иным механизмом без целевого измерения.

### 4.7. Расхождение history между двумя runs seed 7

[ФАКТ] Поэпохное сравнение history 12000-run и 24000-run seed 7:

```
250  0.02734375  0.02734375  SAME
500  0.072265625 0.072265625 SAME
750  0.03515625  0.03515625  SAME
1000 0.125       0.125       SAME
1250 0.166015625 0.16796875  DIFF
1500 0.072265625 0.07421875  DIFF
1750 0.015625    0.015625    SAME
2000 0.013671875 0.017578125 DIFF
2250 0.01953125  0.0234375   DIFF
2500 0.0234375   0.01953125  DIFF
2750 0.015625    0.013671875 DIFF
3000 0.0078125   0.009765625 DIFF
len 48 96
```

[ВЫВОД] Первые четыре selection-проверки полностью детерминированы; расхождение начинается с epoch 1250.
[ВЫВОД] Для интерпретации главного результата расхождение несущественно: 0.1660 и 0.1680 оба на порядок ниже порога, best epoch в обоих runs равен 1250.
[НЕИЗВЕСТНО] Причина расхождения. Не называть её bf16, non-deterministic CUDA kernels, состоянием RNG или чем-либо ещё без различающего измерения. Величина расхождения соответствует 1–2 эпизодам из 512 на проверку, но это наблюдение, а не объяснение.

### 4.8. Измеренная скорость

[ФАКТ] Замеры по логу run S=262144 epochs=24000 (dim=128, key_dim=32, batch_size=32):

- 19:02 → epoch 300
- 19:29 → epoch 4000
- 20:20 → epoch 11300
- завершение около 21:42 → epoch 24000

[ВЫВОД] Между 19:02 и 19:29 пройдено 3700 epochs за ~27 мин (~137 epochs/мин); между 19:29 и 20:20 — 7300 epochs за ~51 мин (~143 epochs/мин).
[ВЫВОД] При этой скорости скрининг на 4000 epochs занимает примерно 28–30 минут, полный run 12000 epochs — примерно 1 ч 25 мин.
[НЕИЗВЕСТНО] Скорость при изменённых `dim`, `key_dim`, `batch_size`. Переносить нельзя (§0.7), измерять по двум точкам лога каждого run.

---

## 5. Опровергнутые и уточнённые утверждения

### 5.1. Опровергнуто в этой сессии

- «При S=262144 нужен только больший бюджет epochs» — **опровергнуто** run epochs=24000 seed 7: best epoch остался 1250, independent real answer 0.190430. Не воскрешать без новых данных.

### 5.2. Уточнения предыдущей сессии

[ФАКТ] Утверждение «отсутствие checkpoint после probe указывает на нештатное завершение» было ошибочным: при `epochs=25` и `acc_every=250` best-checkpoint блок не может выполниться.
[ФАКТ] Утверждение «лог с единственной строкой epoch 1 доказывает остановку после epoch 1» было ошибочным: progress печатается только при `epoch == 1 or epoch % 100 == 0`.

### 5.3. Не воскрешать без новых замеров

- «В S=65536 нет тайбрейкера» — опровергнуто: `recency_scale` обучается.
- «Head принципиально не может декодировать value embeddings» — опровергнуто fresh-head test.
- «LayerNorm — причина answer около 0.50» — опровергнуто.
- «Gate — единственная причина около 0.50» — опровергнуто.
- «Высокий shuffled score доказывает query leakage» — опровергнуто: fixed points старого shuffle объясняли завышение.
- «12000 epochs обязательно улучшает large-S» — опровергнуто: поздние epochs деградируют.
- «Мягкое чтение — адекватный selection signal при large S» — не подтверждено.
- «Коллапс bf16 неизбежен около S=4096–16384» — не подтверждено.

### 5.4. Что запрещено заявлять о S=262144 сейчас

Все следующие утверждения — [НЕИЗВЕСТНО]:

- «S=262144 принципиально недостижим»;
- «нужно увеличить `dim`»;
- «нужно увеличить `key_dim`»;
- «нужно снизить `lr`»;
- «нужен learning rate schedule»;
- «`aux_value_weight=0.01` недостаточно»;
- «нужно изменить `batch_size`»;
- «проблема в bf16»;
- «проблема в `retrieval_weight`»;
- «gate схлопнулся и это причина»;
- «`clip_grad_norm=1.0` слишком мягкий или слишком жёсткий».

Каждое из них — гипотеза, подлежащая проверке по плану §7, а не установленная причина.

---

## 6. Системная стабильность

[ФАКТ] В kernel-журнале текущей загрузки все совпадения — i915 (Intel iGPU) в интервале 11:18:30–11:30:23 28 августа: `Resetting rcs0 for preemption time out`, `*ERROR* GT0: rcs0 reset request timed out`, `GPU HANG: ecode 12:1:84dffffb` / `85dffffd` / `8697fefe`, `context reset` для `Renderer[8655]` и `Xorg[5202]`.
[ФАКТ] Ни одной строки `NVRM Xid`, PCIe AER error/fatal, MCE, hardware error, kernel panic или oom-kill.
[ФАКТ] После 11:30:23 новых совпадений не появлялось, в том числе за время probe, обоих 12000-epoch runs и run epochs=24000.
[ФАКТ] `journalctl -b -k` без sudo сработал и дал полный вывод; `sudo journalctl` требовал пароль.
[ФАКТ] Наблюдённые GPU-срезы: training S=262144 — 45–55 °C, 92–100 W, 5159 MiB; independent eval — 41 °C, 61 W, 2805 MiB; простой — 36 °C, 20 W, 15 MiB.
[ФАКТ] `nvidia-smi --query-compute-apps` во время run показывал единственный процесс PHEF, 5136 MiB; `pgrep -af "DIEL|diel"` был пуст.
[НЕИЗВЕСТНО] Peak VRAM ни для одного режима не измерялся; приведённые значения — единичные срезы.

### 6.1. Практическое правило

- Не запускать PHEF одновременно с DIEL или другим CUDA workload.
- Не запускать тяжёлые диагностические `grep -R` во время обучения.
- Закрывать тяжёлые Firefox вкладки, видео и WebGL во время run.
- Не использовать UI Saby во время run; не останавливать Saby-сервисы без отдельного подтверждения.
- Не менять BIOS, GRUB, kernel, i915, Xorg или NVIDIA driver без отдельной диагностики и явного подтверждения.

### 6.2. Если система падает

После перезагрузки, до запуска нового обучения:

```bash
sudo journalctl -b -1 -k --no-pager -o short-precise | tail -250
```

Не запускать новый run до анализа этого вывода.

---

## 7. ПЛАН: скрининг факторов 1–5 на 4000 epochs

### 7.1. Логика плана

[ВЫВОД] Основание для скрининга: по §4.4 траектории PASS и FAIL расходятся к epoch 1500–2000, поэтому 4000 epochs достаточно, чтобы отбраковать бесперспективный вариант, и это в ~4,5 раза дешевле полного run.
[ВЫВОД] Основание для порядка факторов: по §4.6 взрыв loss локализован в retrieval-члене при неинформативном answer-члене, поэтому приоритет у факторов, влияющих на адресацию и на устойчивость retrieval-оптимизации.

**Правило одного фактора.** В каждом скрининге меняется ровно один параметр относительно §3.1 при `n_segments=262144`, `seed=7`, `epochs=4000`. Никаких других отличий.

### 7.2. Критерий скрининга — зафиксирован ДО запусков

Вариант признаётся **перспективным**, если выполнены оба условия:

1. sel_acc хотя бы в одной проверке окна epochs 1500–4000 превысила **0.20**;
2. к epoch 4000 sel_acc **не** опустилась ниже 0.05.

Вариант признаётся **отбракованным**, если максимум sel_acc в окне 1500–4000 не превысил 0.10.

Промежуточный случай (максимум между 0.10 и 0.20) — **неопределённый**: решение о продлении принимать отдельно, сравнивая с референсной кривой §4.4, и не задним числом переопределяя критерий.

[ВАЖНО] Скрининг не даёт права заявлять PASS или FAIL по протоколу. Он только отбирает кандидатов на полный run §8.

### 7.3. Referenceные значения для сравнения

Из §4.4, окно 1500–4000:

- S=131072 seed 4 (PASS-траектория): 0.2363, 0.2891, 0.3066, 0.4199, 0.4902, 0.6367, 0.7500, 0.8203, 0.8418, 0.7012, 0.6113.
- S=262144 seed 7 (FAIL-траектория): 0.0742, 0.0156, 0.0176, 0.0234, 0.0195, 0.0137, 0.0098, 0.0254, 0.0098, 0.0215, 0.0098.

### 7.4. Скрининг 1 — `--key-dim 64`

Мотивация: [ФАКТ] retrieval-softmax различает 262144 позиции; [ФАКТ] при key_dim=32 режим работал на S=131072 и сломался на S=262144. [НЕИЗВЕСТНО] достаточна ли размерность ключа — это и проверяется.

Проверка свободных имён (read-only):

```bash
cd ~/qubitrino-projects/agi/phef && for f in results/scr_s262144_seed7_kd64_e4000.json results/scr_s262144_seed7_kd64_e4000.pt /tmp/probe_logs/phef_scr_s262144_seed7_kd64_e4000.log; do if test -e "$f"; then echo "EXISTS $f"; ls -l "$f"; else echo "MISSING $f"; fi; done
```

Команда запуска (write — требует подтверждения):

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/train_recall.py --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 64 --dim 128 --batch-size 32 --epochs 4000 --lr 3e-4 --weight-decay 1e-4 --retrieval-weight 1.0 --aux-value-weight 0.01 --hard-eval --eval-batch-size 16 --eval-batches 64 --use-best --acc-every 250 --acc-n 512 --seed 7 --out results/scr_s262144_seed7_kd64_e4000.json --best-out results/scr_s262144_seed7_kd64_e4000.pt > /tmp/probe_logs/phef_scr_s262144_seed7_kd64_e4000.log 2>&1 &
```

Промежуточный вариант при неопределённом результате: `--key-dim 48` с именами `..._kd48_e4000`.

### 7.5. Скрининг 2 — `--lr 1e-4`

Мотивация: [ФАКТ] взрыв retrieval loss с ~12 до 363–560 при стабильном answer (§4.6).

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/train_recall.py --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 32 --dim 128 --batch-size 32 --epochs 4000 --lr 1e-4 --weight-decay 1e-4 --retrieval-weight 1.0 --aux-value-weight 0.01 --hard-eval --eval-batch-size 16 --eval-batches 64 --use-best --acc-every 250 --acc-n 512 --seed 7 --out results/scr_s262144_seed7_lr1e4_e4000.json --best-out results/scr_s262144_seed7_lr1e4_e4000.pt > /tmp/probe_logs/phef_scr_s262144_seed7_lr1e4_e4000.log 2>&1 &
```

[ВЫВОД] Оговорка при интерпретации: снижение lr втрое замедляет и полезное обучение, поэтому при неопределённом результате осмысленно продлить именно этот скрининг до 8000–12000 epochs отдельным run, а не отбраковывать сразу. Вариант `--lr 2e-4` — резерв с именами `..._lr2e4_e4000`.

### 7.6. Скрининг 3 — `--retrieval-weight 0.5`

Мотивация: [ФАКТ] рост total шёл именно через взвешенный retrieval-член.

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/train_recall.py --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 32 --dim 128 --batch-size 32 --epochs 4000 --lr 3e-4 --weight-decay 1e-4 --retrieval-weight 0.5 --aux-value-weight 0.01 --hard-eval --eval-batch-size 16 --eval-batches 64 --use-best --acc-every 250 --acc-n 512 --seed 7 --out results/scr_s262144_seed7_rw05_e4000.json --best-out results/scr_s262144_seed7_rw05_e4000.pt > /tmp/probe_logs/phef_scr_s262144_seed7_rw05_e4000.log 2>&1 &
```

Резерв: `--retrieval-weight 2.0`, имена `..._rw20_e4000`.

### 7.7. Скрининг 4 — `--dim 256` (сначала probe на VRAM)

[НЕИЗВЕСТНО] Peak VRAM и скорость при dim=256. Обязателен technical probe до полного скрининга.

Probe (write — требует подтверждения):

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/train_recall.py --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 32 --dim 256 --batch-size 32 --epochs 25 --lr 3e-4 --weight-decay 1e-4 --retrieval-weight 1.0 --aux-value-weight 0.01 --hard-eval --eval-batch-size 16 --eval-batches 64 --use-best --acc-every 250 --acc-n 512 --seed 7 --out results/probe_s262144_seed7_dim256.json > /tmp/probe_logs/phef_probe_s262144_seed7_dim256.log 2>&1 &
```

[ВЫВОД] При `epochs=25` и `acc_every=250` checkpoint не создастся и history будет пустой — это ожидаемо (§3.4) и не является сбоем. Probe проверяет только отсутствие OOM и traceback.

Во время probe снять VRAM (read-only, повторить несколько раз):

```bash
nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader 2>&1
```

Скрининг после успешного probe:

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/train_recall.py --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 32 --dim 256 --batch-size 32 --epochs 4000 --lr 3e-4 --weight-decay 1e-4 --retrieval-weight 1.0 --aux-value-weight 0.01 --hard-eval --eval-batch-size 16 --eval-batches 64 --use-best --acc-every 250 --acc-n 512 --seed 7 --out results/scr_s262144_seed7_dim256_e4000.json --best-out results/scr_s262144_seed7_dim256_e4000.pt > /tmp/probe_logs/phef_scr_s262144_seed7_dim256_e4000.log 2>&1 &
```

Промежуточный вариант: `--dim 192`, имена `..._dim192_e4000`.

### 7.8. Скрининг 5 — `--batch-size 64` (сначала probe на OOM)

Probe:

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/train_recall.py --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 32 --dim 128 --batch-size 64 --epochs 25 --lr 3e-4 --weight-decay 1e-4 --retrieval-weight 1.0 --aux-value-weight 0.01 --hard-eval --eval-batch-size 16 --eval-batches 64 --use-best --acc-every 250 --acc-n 512 --seed 7 --out results/probe_s262144_seed7_bs64.json > /tmp/probe_logs/phef_probe_s262144_seed7_bs64.log 2>&1 &
```

Скрининг после успешного probe:

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/train_recall.py --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 32 --dim 128 --batch-size 64 --epochs 4000 --lr 3e-4 --weight-decay 1e-4 --retrieval-weight 1.0 --aux-value-weight 0.01 --hard-eval --eval-batch-size 16 --eval-batches 64 --use-best --acc-every 250 --acc-n 512 --seed 7 --out results/scr_s262144_seed7_bs64_e4000.json --best-out results/scr_s262144_seed7_bs64_e4000.pt > /tmp/probe_logs/phef_scr_s262144_seed7_bs64_e4000.log 2>&1 &
```

[ВЫВОД] Оговорка: при `batch_size=64` одна epoch обрабатывает вдвое больше эпизодов, поэтому 4000 epochs здесь — не тот же бюджет данных, что при 32. Это ограничивает сравнимость и должно быть указано при интерпретации; при неопределённом результате корректнее сравнивать по числу обработанных эпизодов, а не epochs.

### 7.9. Мониторинг любого скрининга (read-only)

Подставить `TAG` (например `kd64`, `lr1e4`, `rw05`, `dim256`, `bs64`):

```bash
cd ~/qubitrino-projects/agi/phef && echo "=== PROCESS ===" && pgrep -af "train_recall.py.*TAG_e4000" || true; echo "=== LAST EPOCH ===" && grep -E "^epoch " /tmp/probe_logs/phef_scr_s262144_seed7_TAG_e4000.log | tail -1; echo "=== SEL_ACC ===" && grep -E "^epoch .* sel_acc " /tmp/probe_logs/phef_scr_s262144_seed7_TAG_e4000.log | tail -16; echo "=== LOG TAIL ===" && tail -20 /tmp/probe_logs/phef_scr_s262144_seed7_TAG_e4000.log 2>&1; echo "=== GPU ===" && nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader 2>&1
```

### 7.10. Признаки штатного завершения

[ФАКТ] Штатное окончание подтверждается одновременно:

- `pgrep` больше не показывает процесс обучения;
- лог содержит `восстановлен best checkpoint`;
- лог содержит блок `=== CONTROLS ===`;
- лог содержит `control status`;
- лог содержит `сохранено: results/...json`;
- соответствующие JSON и `.pt` существуют.

[ВАЖНО] `control status` в скрининге почти наверняка будет FAIL просто потому, что 4000 epochs — не полный протокол. Читать нужно траекторию sel_acc, а не `control_status`.

### 7.11. Сверка скрининга с референсом (read-only)

```bash
cd ~/qubitrino-projects/agi/phef && /home/linux/qubitrino-projects/agi/venv/bin/python -c "
import json,sys
p=sys.argv[1]
d=json.load(open(p)); h=d.get('history') or []
print('CONFIG', d.get('config'))
print({k:d.get(k) for k in ['best_epoch','best_acc','escape_epoch','answer_acc','retrieval_acc','memory_gate','recency_scale','score_scale','tie_rate','control_status']})
w=[x for x in h if 1500<=x['epoch']<=4000]
print('curve', [(x['epoch'],round(x['sel_acc'],4)) for x in h])
print('max_1500_4000=', max([x['sel_acc'] for x in w], default=None), 'last=', h[-1]['sel_acc'] if h else None)
print('REF S131072 max_1500_4000 = 0.8418 ; REF S262144 max_1500_4000 = 0.0742')
" results/scr_s262144_seed7_TAG_e4000.json
```

### 7.12. Порядок выполнения

1. Скрининг 1 `--key-dim 64`.
2. Скрининг 2 `--lr 1e-4`.
3. Скрининг 3 `--retrieval-weight 0.5`.
4. Probe + скрининг 4 `--dim 256`.
5. Probe + скрининг 5 `--batch-size 64`.

Запускать строго по одному: одновременный запуск двух runs нарушает §6.1 и делает измерения скорости и VRAM неинтерпретируемыми.

---

## 8. Что делать с перспективным кандидатом

[ВЫВОД] Если скрининг признан перспективным по §7.2, порядок такой:

1. Полный run с тем же единственным изменением, `epochs=12000`, `seed=7`, новые имена артефактов вида `cap_k512_s262144_TAG_seed7.json` / `.pt`. Обоснование выбора 12000, а не 24000: [ФАКТ] при S=131072 best epochs были 9750, 9750, 10000; [ФАКТ] при S=262144 удвоение до 24000 ничего не дало.
2. Independent eval best checkpoint на 16384 эпизодах. **Обязательно** передать `--key-dim` и `--dim`, совпадающие с training-config этого checkpoint.
3. Применить критерий PASS без изменения порогов:

```
real answer > 0.90
real retrieval > 0.90
shuffled answer < 0.03125
masked answer < 0.03125
```

4. При PASS — репликация на seeds 8 и 9 при полностью идентичной конфигурации и отдельных именах, затем independent eval каждого.
5. Commit только JSON белым списком; `.pt` не коммитить.

Шаблон independent eval (подставить TAG и, при изменении архитектуры, актуальные `--key-dim`/`--dim`):

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/eval_recall_checkpoint.py --checkpoint results/cap_k512_s262144_TAG_seed7.pt --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 32 --dim 128 --eval-batch-size 16 --eval-batches 1024 --seed 7 --out results/cap_k512_s262144_TAG_seed7_eval16384.json > /tmp/probe_logs/phef_s262144_TAG_seed7_eval16384.log 2>&1 &
```

Что проверять в eval JSON: процесс завершён; лог содержит `saved: results/...eval16384.json`; JSON существует; `hard_eval: true`; config соответствует точному run, включая `n_segments=262144`, `seed=7`, `key_dim` и `dim` того checkpoint; число `episodes` в `real`, `shuffled`, `masked` равно 16384; критерий PASS применён без изменения порогов.

[НЕИЗВЕСТНО] Понадобится ли S=524288 иных параметров — переносить нельзя, нужен собственный probe и run.

---

## 9. Комбинации двух параметров

**Правило допуска.** Комбинация запускается только после того, как оба входящих в неё одиночных фактора измерены по §7. Согласовано с пользователем 28 августа: сначала одиночные факторы 1–5, затем комбинации.

[ВЫВОД] Интерпретация комбинации ограничена: если комбинация перспективна, а оба одиночных фактора нет, приписать эффект одному из них нельзя — это будет [НЕИЗВЕСТНО] до дополнительного разложения.

### 9.1. Приоритет комбинаций

| № | Комбинация | Мотивация из измерений | Имя артефактов |
|---|---|---|---|
| C1 | `--key-dim 64` + `--lr 1e-4` | ёмкость адресации плюс устойчивость retrieval-оптимизации — два независимых по механизму фактора | `scr_s262144_seed7_kd64_lr1e4_e4000` |
| C2 | `--key-dim 64` + `--retrieval-weight 0.5` | ёмкость адресации плюс уменьшение вклада взорвавшегося члена | `scr_s262144_seed7_kd64_rw05_e4000` |
| C3 | `--key-dim 64` + `--dim 256` | одновременное расширение ключевого и скрытого пространства; требует probe на VRAM | `scr_s262144_seed7_kd64_dim256_e4000` |
| C4 | `--lr 1e-4` + `--retrieval-weight 0.5` | два фактора устойчивости оптимизации без изменения архитектуры | `scr_s262144_seed7_lr1e4_rw05_e4000` |
| C5 | `--key-dim 64` + `--batch-size 64` | ёмкость адресации плюс снижение шума градиента; требует probe на OOM | `scr_s262144_seed7_kd64_bs64_e4000` |

[ВЫВОД] C1 первая, потому что объединяет по одному фактору из каждого механистически различного класса, выделенного измерением §4.6 (ёмкость адресации против устойчивости retrieval-члена).

### 9.2. Шаблон команды комбинации

Пример для C1 (write — требует подтверждения):

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/train_recall.py --n-keys 512 --n-values 64 --n-segments 262144 --overwrite-p 0.3 --key-dim 64 --dim 128 --batch-size 32 --epochs 4000 --lr 1e-4 --weight-decay 1e-4 --retrieval-weight 1.0 --aux-value-weight 0.01 --hard-eval --eval-batch-size 16 --eval-batches 64 --use-best --acc-every 250 --acc-n 512 --seed 7 --out results/scr_s262144_seed7_kd64_lr1e4_e4000.json --best-out results/scr_s262144_seed7_kd64_lr1e4_e4000.pt > /tmp/probe_logs/phef_scr_s262144_seed7_kd64_lr1e4_e4000.log 2>&1 &
```

Для C2–C5 менять только соответствующие флаги и все три имени артефактов; ничего иного не трогать.

[ВЫВОД] Для комбинаций с `--lr 1e-4` бюджет 4000 epochs может оказаться недостаточным по той же причине, что в §7.5. Если C1 или C4 попадут в неопределённую зону, корректный следующий шаг — продление именно этой конфигурации отдельным run на 12000 epochs, а не отбраковка.

### 9.3. Если все одиночные факторы и комбинации C1–C5 отбракованы

[ВЫВОД] Тогда исчерпаны CLI-факторы, и осмысленно переходить к изменениям кода, каждое — отдельным diff с согласованием (§0.5):

1. Learning rate schedule: warmup плюс cosine decay. Мотивация: [ФАКТ] раннее плато и последующий взрыв retrieval loss.
2. Значение `clip_grad_norm`: сейчас жёстко 1.0 в строке `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
3. Отключение autocast bf16 для retrieval-члена или запуск с `enabled=False`. [НЕИЗВЕСТНО] влияние; §5.3 запрещает называть bf16 причиной заранее.
4. `--acc-every 100` — не изменение обучения, а изменение частоты фиксации best checkpoint. Дешёвый и законный пункт, его можно вставить в любой момент.

### 9.4. Диагностика на существующих checkpoints

[ВЫВОД] Не ведёт к PASS напрямую, но дёшево и может переориентировать план:

1. Разложение retrieval против answer: измерить на `cap_k512_s262144_aux001_seed7_e24000.pt` долю эпизодов с корректным retrieval при некорректном answer и наоборот. [ФАКТ] real retrieval 0.5378 при real answer 0.1904 — асимметрия есть, её структура [НЕИЗВЕСТНО].
2. Fresh-head test на том же checkpoint: обучить линейную head на фиксированных `value_emb` и проверить декодируемость 64 value embeddings. Ранее такой тест на S=65536 отделил проблему head от проблемы retrieval.
3. Смысл `last_tie_rate` в коде: [ФАКТ] при S=131072 tie_rate=0.0625, при S=262144 tie_rate=0.0. Прочитать код до любой интерпретации.

### 9.5. Промежуточный масштаб

[ВЫВОД] Независимо от исхода скринингов полезно установить границу: `n_segments=196608` при неизменной остальной конфигурации §3.1 и 12000 epochs.
[НЕИЗВЕСТНО] Пройдёт ли этот масштаб. Значение в том, что он отделяет резкий обрыв между 131072 и 262144 от плавной деградации.

### 9.6. i915

[ВЫВОД] Для многочасовых run наиболее безопасный режим — headless/SSH с минимальной графической нагрузкой на Intel iGPU либо перевод desktop на RTX 4090. Это рекомендация, не выполненное изменение.

---

## 10. Полезные шаблоны команд

### 10.1. Read-only сверка в новом чате

```bash
cd ~/qubitrino-projects/agi && git status --short | head -20 && git log --oneline -8 && git branch -vv && git diff --check && cd phef && for f in phef/model.py phef/world.py experiments/train_recall.py experiments/eval_recall_checkpoint.py; do /home/linux/qubitrino-projects/agi/venv/bin/python -c "import ast; ast.parse(open('$f').read()); print('ok $f')"; done
```

### 10.2. Сверка ключевых результатов

```bash
cd ~/qubitrino-projects/agi/phef && /home/linux/qubitrino-projects/agi/venv/bin/python -c "
import json
for p in ['results/cap_k512_s131072_aux001_seed4.json','results/cap_k512_s262144_aux001_seed7.json','results/cap_k512_s262144_aux001_seed8.json','results/cap_k512_s262144_aux001_seed7_e24000.json']:
    d=json.load(open(p)); h=d.get('history') or []
    print('===',p)
    print({k:d.get(k) for k in ['best_epoch','best_acc','escape_epoch','answer_acc','retrieval_acc','memory_gate','recency_scale','score_scale','tie_rate','control_status']})
    print('history_len=',len(h),'max_sel=',max([x['sel_acc'] for x in h], default=None))
"
```

### 10.3. Системная готовность перед run

```bash
echo "=== CUDA PROCESSES ===" && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1; echo "=== PHEF / DIEL ===" && pgrep -af "train_recall.py|eval_recall_checkpoint.py|DIEL|diel" || true; echo "=== KERNEL WARNINGS ===" && journalctl -b -k --no-pager 2>&1 | grep -iE "i915.*(GPU HANG|reset|heartbeat|context reset)|nvrm.*xid|aer.*(error|fatal)|mce|hardware error|kernel panic|oom-kill" | tail -60 || true
```

### 10.4. Git commit результатов

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

```bash
cd ~/qubitrino-projects/agi && git log --oneline -3 && git status --short | head -5 && git branch -vv
```

---

## 11. Рекомендуемые файлы для нового чата

Минимально приложить:

- Этот handoff.
- `phef/phef/model.py`.
- `phef/phef/world.py`.
- `phef/experiments/train_recall.py`.
- `phef/experiments/eval_recall_checkpoint.py`.
- `phef/results/cap_k512_s131072_aux001_seed4.json` и `..._seed4_eval16384.json`.
- `phef/results/cap_k512_s262144_aux001_seed7_e24000.json` и `..._e24000_eval16384.json`.
- `phef/results/cap_k512_s262144_aux001_seed7.json` и `..._seed8.json` при необходимости сравнения.
- JSON новых скринингов по мере появления.

Полезные исторические документы:

- `docs/handoffs/20_PHEF-handoff-S65536-checkpoint-shuffle-control.md`
- `docs/handoffs/21_PHEF_handoff_S65536_PASS.md`
- `docs/handoffs/22_PHEF_handoff_S131072_three_seed_PASS.md`
- `docs/handoffs/23_PHEF_handoff_S262144_two_seed_FAIL_plan_e24000.md`

Не обязательно прикладывать: `train_recall.py.bak`; `/tmp`-логи; old smoke results; `logs/microtest/`; probe JSON.

---

## 12. Краткая стартовая инструкция новому чату

1. Прочитать этот документ целиком.
2. Не делать предположений; маркировать каждое утверждение [ФАКТ]/[ВЫВОД]/[НЕИЗВЕСТНО].
3. Выполнить только read-only сверку §10.1 и §10.2 и дождаться вывода пользователя.
4. Проверить системную готовность §10.3.
5. Проверить свободные имена артефактов очередного скрининга.
6. Согласовать точную команду скрининга §7 и получить явное подтверждение буквальной фразой перед запуском.
7. Во время run использовать только read-only мониторинг §7.9; соблюдать §6.1.
8. После завершения — сверка §7.11, применение критерия §7.2 без переопределения задним числом.
9. Перспективного кандидата вести по §8; комбинации — только по §9 и только после одиночных факторов.
10. Не коммитить новые результаты до независимой проверки и явного подтверждения пользователя.

**Текущий подтверждённый статус:** S=65536 PASS; S=131072 PASS на seeds 4–6; S=262144 FAIL на seeds 7–8 при 12000 epochs и FAIL на seed 7 при 24000 epochs. Гипотеза «нужно больше epochs» опровергнута. Следующий шаг — скрининг 1 `--key-dim 64` по §7.4.
