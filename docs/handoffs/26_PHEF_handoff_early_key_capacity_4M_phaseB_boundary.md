# PHEF — самодостаточный хендофф 26: Early KD32 key-capacity ≥4M, Phase-B boundary 2^17…2^18

Дата передачи: 2026-09-04.
Репозиторий: `~/qubitrino-projects/agi`.
Рабочий каталог PHEF: `~/qubitrino-projects/agi/phef`.
Ветка: `main`.
Текущий HEAD на момент handoff: `ab81050`.
Remote: `gitlab.qubitrinotech.ru:root/agi.git`.

---

## 0. Главное правило — не гадать

Не гадать. Не фантазировать. Не подменять неизвестное правдоподобной версией.

Каждое содержательное утверждение маркировать:

- **[ФАКТ]** — прямо подтверждено командным выводом, JSON, checkpoint metadata, логом или Git-выводом
- **[ВЫВОД]** — логическое следствие подтвержденных фактов
- **[ГИПОТЕЗА]** — проверяемое объяснение, не установленный факт
- **[НЕИЗВЕСТНО]** — данных недостаточно; запросить минимальную различающую проверку

### 0.1. Граница доступа

Ассистент не имеет прямого доступа к локальной машине, shell, Git, GPU, JSON, checkpoint, исходникам или логам. Все команды запускает пользователь и присылает полный вывод.

Нельзя утверждать, что «процесс жив», «GPU свободна», «JSON создан», «в коде есть флаг» или «run завершился», если это не подтверждено присланным первичным артефактом.

### 0.2. Config — только из целевого JSON

Параметры конкретного run/eval подтверждать только полем `config` соответствующего JSON, а не пересказом, именем файла или старым handoff.

### 0.3. Действия, меняющие состояние

Перед каждым write-действием обязательно:

1. Показать точную команду или diff.
2. Назвать все затрагиваемые файлы.
3. Получить явное подтверждение пользователя.
4. После запуска запросить read-only проверку фактического результата.

К write-действиям относятся: запуск train/eval, создающий JSON/log/checkpoint; создание/изменение файла; `git add`, commit, push; удаление/остановка процессов; системные изменения.

### 0.4. Никакой экстраполяции

Нельзя переносить без отдельного измерения между конфигурациями:

- качество
- устойчивость по seed
- capacity
- peak VRAM
- throughput и время
- поведение loss
- влияние `rw`, lr, `S`, `key_dim`, epochs
- надёжность controls

### 0.5. Критерий PASS

Для full independent eval:

```text
real answer >= 0.90
real retrieval >= 0.90
shuffled answer < 0.03125
masked answer < 0.03125
hard_eval = true
episodes = 16384 в real, shuffled и masked
```

Answer chance:

```text
1 / n_values = 1 / 64 = 0.015625
```

Controls threshold:

```text
2 * chance = 0.03125
```

Distance probes на 1024 эпизодах используются для profiling/capacity screening и не заменяют full independent PASS.

---

## 1. Окружение и обязательные правила запуска

- Git root: `~/qubitrino-projects/agi`
- PHEF: `~/qubitrino-projects/agi/phef`
- Python: `/home/linux/qubitrino-projects/agi/venv/bin/python`
- Ветка: `main`
- Remote: `gitlab.qubitrinotech.ru:root/agi.git`
- GPU: RTX 4090, общий объём VRAM ранее наблюдался около 24 GiB
- На одной GPU не запускать параллельные CUDA jobs
- Для CUDA Python всегда:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- Для PHEF Python всегда:

```bash
env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python
```

- Длинные tasks запускать только через `setsid nohup ... & disown`
- Не запускать train/eval через pipe с `tail`
- Логи и checkpoints не добавлять в Git без отдельного решения
- `/tmp` не является местом для воспроизводимых артефактов; JSON сохранять в `results/`

### 1.1. Shell-подводные камни

- В длинных командах использовать `cd ...;` и не рассчитывать, что background-цепочка меняет каталог текущей оболочки
- Если ожидается отсутствие файла, использовать `;`, а не длинную цепочку `&&`
- `git diff --cached --stat` возвращает ненулевой exit code при пустом index
- `tail -f` показывает только то, что пишет приложение. Текущий evaluator печатает старт и финал, но не batch progress
- Для живости долгого eval использовать `ps` и `nvidia-smi`, не перезапускать job только из-за молчащего log

---

## 2. Состояние Git и commits

[ФАКТ] На последней показанной проверке:

```text
ab81050 (HEAD -> main, origin/main)
```

[ФАКТ] Commit `ab81050` зафиксировал forward curriculum:

```text
Phase A: S=65536, rw=1.0, обучение с нуля, seed 7
Phase B: init Phase A -> S=524288, rw=0.125, lr=1e-4
```

[ФАКТ] В истории также находятся:

| Commit | Содержание |
|---|---|
| `0a0d6d8` | S=262144, `rw=0.25`, seed 9 PASS; вместе с предыдущими seed 7–8 образует 3-seed PASS |
| `d095b29` | S=524288, `rw=0.125`, seed 7 independent PASS |
| `7d91902` | S=1048576 screening `rw=0.0625`, FAIL |
| `852a578` | S=1048576 screening `rw=0.125`, FAIL |
| `02a33a7` | Добавлен `--init-checkpoint` |
| `3c1eb5c` | Zero-shot S=2^19 checkpoint на S=2^20, independent PASS |
| `a6bfe48` | Exploratory zero-shot на S=2^21, 2048 эпизодов |
| `3f68585` | `probe_distance_profile.py`, profiling дистанции |
| `7627ee8` | Двухфазное расписание retrieval weight |
| `423f0c5` | Schedule reproduction seed 4, partial reverse-transfer repair |
| `ab81050` | Forward curriculum Phase A → Phase B |

[НЕИЗВЕСТНО] Текущий список tracked/untracked файлов после новых eval не проверен в этом handoff. Перед `git add` обязательно выполнить `git status --short`, затем выбрать явный whitelist.

---

## 3. Метод и модели

### 3.1. Общая постановка

```text
n_values = 64
overwrite_p = 0.3
dim = 128
aux_value_weight = 0.01
hard_eval = true
weight_decay = 1e-4
use_best = true
acc_every = 250
acc_n = 512
```

### 3.2. Early / Phase-A checkpoint

Файл:

```text
results/early_s65536_seed7_rw10_e6000_best.pt
```

[ФАКТ] Training JSON:

```text
results/early_s65536_seed7_rw10_e6000.json
```

[ФАКТ] Config:

```text
n_segments = 65536
n_keys = 512
key_dim = 32
dim = 128
batch_size = 32
epochs = 6000
lr = 3e-4
retrieval_weight = 1.0
aux_value_weight = 0.01
seed = 7
```

[ФАКТ] Best checkpoint: epoch 4500, `sel_acc=1.0`.

[ФАКТ] Internal training report:

```text
answer_acc = 1.0
retrieval_acc = 1.0
shuffled_answer_acc = 0.017578125
masked_answer_acc = 0.013671875
control_status = PASS
```

### 3.3. Phase-B checkpoint

Файл:

```text
results/phaseB_from_early_at_s524288_rw0125_e6000_best.pt
```

[ФАКТ] Training JSON:

```text
results/phaseB_from_early_at_s524288_rw0125_e6000.json
```

[ФАКТ] Config:

```text
init_checkpoint = results/early_s65536_seed7_rw10_e6000_best.pt
n_segments = 524288
n_keys = 512
key_dim = 32
dim = 128
batch_size = 32
epochs = 6000
lr = 1e-4
retrieval_weight = 0.125
aux_value_weight = 0.01
seed = 7
```

[ФАКТ] Best checkpoint: epoch 750, `sel_acc=1.0`.

[ФАКТ] Internal training report:

```text
answer_acc = 0.9990234375
retrieval_acc = 0.9990234375
shuffled_answer_acc = 0.015625
masked_answer_acc = 0.01171875
control_status = PASS
```

[ФАКТ] History Phase B:

```text
epoch 250: sel_acc 0.99609375
epoch 500: sel_acc 0.998046875
epoch 750: sel_acc 1.0
epoch 2500: sel_acc 0.990234375
epoch 4000: sel_acc 0.958984375
epoch 4500: sel_acc 0.609375
epoch 6000: sel_acc 0.02734375
```

[ВЫВОД] `--use-best` является обязательной частью протокола large-S: финальная модель на epoch 6000 непригодна, но best checkpoint на epoch 750 сохраняет высокое базовое качество.

---

## 4. Главные результаты

### 4.1. Early KD32: full independent capacity

Все full eval ниже используют checkpoint:

```text
results/early_s65536_seed7_rw10_e6000_best.pt
```

[ФАКТ] Full independent results:

| `n_keys` | `S` | Batch × batches | Episodes | Real answer | Real retrieval | Shuffled answer | Masked answer | Статус |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 524288 | 65536 | 16 × 1024 | 16384 | 0.979187 | 0.979004 | 0.015991 | 0.016357 | PASS |
| 1048576 | 65536 | 8 × 2048 | 16384 | 0.977966 | 0.977905 | 0.015076 | 0.014832 | PASS |
| 4194304 | 65536 | 2 × 8192 | 16384 | 0.978455 | 0.978271 | 0.014038 | 0.016296 | PASS |

[ВЫВОД] При `key_dim=32`, `dim=128`, Early checkpoint имеет подтверждённую clean key-capacity минимум 4194304 = `2^22` различных ключей. Верхний предел не найден.

[ВЫВОД] Между 524288, 1048576 и 4194304 ключами нет значимой наблюдаемой деградации: real answer остаётся около 0.978–0.979.

[НЕИЗВЕСТНО] Устойчивость результата по train seeds не проверена: все приведённые Early full eval используют checkpoint training seed 7.

### 4.2. Early KD32: distance profiles

[ФАКТ] Distance probes на 1024 episodes при `S=65536`:

| `n_keys` | Overall answer | Дальний квантиль answer | Максимальная показанная дистанция |
|---:|---:|---:|---:|
| 4096 | 0.999023 | 0.996094 | 36844 |
| 32768 | 0.988281 | 0.988281 | 63098 |
| 65536 | 0.982422 | 0.960938 | 65101 |
| 131072 | 0.981445 | 0.972656 | 64916 |
| 262144 | 0.985352 | 0.984375 | 64645 |
| 524288 | 0.977539 | 0.972656 | 64876 |

[ВЫВОД] В доступной выборке Early KD32 checkpoint удерживает высокую accuracy и в дальних квантилях. Это поддерживает содержательную addressability, а не объяснение «только недавней записью».

[НЕИЗВЕСТНО] Эти probes не заменяют independent controls и не являются доказательством PASS сами по себе.

### 4.3. KD64: отрицательный результат

[ФАКТ] KD64 checkpoint:

```text
results/early_s65536_seed7_kd64_rw10_e6000_best.pt
```

[ФАКТ] Config совпадает с Early Phase A, кроме `key_dim=64`.

[ФАКТ] Internal training report проходит controls:

```text
best_epoch = 4500
best_acc = 1.0
answer_acc = 0.99609375
retrieval_acc = 0.99609375
shuffled_answer_acc = 0.0224609375
masked_answer_acc = 0.0068359375
```

[ФАКТ] Его 1024-episode distance probes:

| `n_keys` | Overall answer | Дальний квантиль answer |
|---:|---:|---:|
| 4096 | 0.961914 | 0.953307 |
| 32768 | 0.862305 | 0.804688 |
| 65536 | 0.855469 | 0.753906 |
| 131072 | 0.847656 | 0.757812 |

[ВЫВОД] В проверенной одной конфигурации и одном seed `key_dim=64` существенно хуже KD32 по capacity probes.

[НЕИЗВЕСТНО] Нельзя обобщать это до «KD64 вреден»: не проверены другие `dim`, lr, epochs, weights, schedule и seeds.

### 4.4. Phase B: full independent capacity boundary

Все full eval ниже используют checkpoint:

```text
results/phaseB_from_early_at_s524288_rw0125_e6000_best.pt
```

[ФАКТ] Full independent results:

| `n_keys` | `S` | Batch × batches | Episodes | Real answer | Real retrieval | Shuffled answer | Masked answer | Статус |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 65536 | 524288 | 8 × 2048 | 16384 | 0.941345 | 0.940613 | 0.017212 | 0.015137 | PASS |
| 131072 | 524288 | 8 × 2048 | 16384 | 0.910400 | 0.909302 | 0.016357 | 0.015442 | PASS |
| 262144 | 524288 | 8 × 2048 | 16384 | 0.878418 | 0.876099 | 0.016541 | 0.015625 | FAIL |
| 1048576 | 524288 | 8 × 2048 | 16384 | 0.847961 | 0.845520 | 0.015320 | 0.015320 | FAIL |

[ВЫВОД] Official PASS boundary Phase B находится строго в интервале:

```text
131072 < K_PASS_PhaseB < 262144
```

[ВЫВОД] Long-S adaptation Phase B сохранила clean hard-retrieval, но уменьшила practical key-capacity по сравнению с Early checkpoint, который PASS до минимум `2^22`.

### 4.5. Phase B: distance profile

[ФАКТ] Phase-B probes, 1024 episodes:

| `n_keys` | Overall answer | Overall slot | Дальний квантиль answer | Максимальная показанная дистанция |
|---:|---:|---:|---:|---:|
| 4096 | 0.993164 | 0.993164 | 0.992188 | 30397 |
| 32768 | 0.971680 | 0.970703 | 0.949219 | 199950 |
| 65536 | 0.940430 | 0.938477 | 0.898438 | 479060 |

[ВЫВОД] При `n_keys=65536` и `S=524288` есть измеримый спад в наиболее дальнем квантиле: 0.898438 против 0.957031 в ближнем. Но полная independent оценка всё же даёт PASS 0.941345 / 0.940613.

---

## 5. Что установлено и что нет

### 5.1. Установлено

1. [ФАКТ] На `S=262144`, `rw=0.25` есть 3-seed PASS.
2. [ФАКТ] На `S=524288`, `rw=0.125` обучение с нуля дало independent PASS на seed 7.
3. [ФАКТ] Zero-shot от `S=524288` checkpoint на `S=1048576` проходил full eval; exploratory `S=2097152` eval на 2048 эпизодах также был высоким.
4. [ФАКТ] Forward curriculum Early `S=65536, rw=1.0` → Phase B `S=524288, rw=0.125, lr=1e-4` технически и функционально работает.
5. [ФАКТ] Early KD32 checkpoint проходит clean full independent PASS при `n_keys=2^22`.
6. [ФАКТ] Phase B проходит clean full independent PASS при `n_keys=2^17`, но FAIL при `n_keys=2^18`.
7. [ФАКТ] Batch 16 не помещался для Early eval `n_keys=2^20`; batch 8 работал. Для 4M batch 2 работал.
8. [ФАКТ] 4M full eval работал при batch 2 и 8192 batches, но был очень долгим: JSON создан после примерно 17 часов от запуска.

### 5.2. Не установлено

1. [НЕИЗВЕСТНО] Почему Phase B теряет capacity: причиной может быть `S`, low `rw`, lr, длительность второй фазы или взаимодействие факторов.
2. [НЕИЗВЕСТНО] Точный capacity ceiling Early KD32 выше 4M.
3. [НЕИЗВЕСТНО] Устойчивость Early 4M result по train seeds.
4. [НЕИЗВЕСТНО] Механизм плохого KD64 run.
5. [НЕИЗВЕСТНО] Можно ли сохранить Early capacity на `S=524288` другой длительностью Phase B, меньшим lr, другим `rw` или ранней остановкой.
6. [НЕИЗВЕСТНО] Phase-B capacity crossing внутри интервала 131072–262144: точное значение ещё не измерено.

### 5.3. Не воскрешать без нового измерения

- «S=262144 принципиально не обучается» — опровергнуто `rw=0.25`.
- «Нужно просто больше epochs при исходном rw=1.0» — опровергнуто для старого S=262144 seed 7 режима: 24000 epochs не улучшили best result.
- «Ёмкость определяется только длиной S» — опровергнуто сравнением fixed checkpoint на разных `S`.
- «KD32 ограничена десятками тысяч ключей» — опровергнуто full PASS до 4M.
- «KD64 автоматически увеличивает ёмкость» — не подтверждено; в текущем run наблюдается обратное.
- «Большой retrieval loss полностью объясняет позднюю деградацию Phase B» — недостаточно: Phase-B `sel_acc` начинает снижаться до позднего роста retrieval term.
- «4M capacity — максимум» — не доказано; это только lower bound.
- «Отсутствие промежуточных строк evaluator означает зависание» — опровергнуто многими штатно завершившимися eval.

---

## 6. Технические артефакты и ограничения

### 6.1. Новые full-eval JSON, кандидаты для commit

Проверить `git status` и при необходимости добавить исключительно явным whitelist:

```text
phef/results/early_s65536_seed7_rw10_keys524288_eval16384.json
phef/results/early_s65536_seed7_rw10_keys1048576_b8_eval16384.json
phef/results/early_s65536_seed7_rw10_keys4194304_b2_eval16384.json
phef/results/phaseB_from_early_s524288_keys65536_b8_eval16384.json
phef/results/phaseB_from_early_s524288_keys131072_b8_eval16384.json
phef/results/phaseB_from_early_s524288_keys262144_b8_eval16384_retry1.json
phef/results/phaseB_from_early_s524288_keys1048576_b8_eval16384.json
```

### 6.2. Не добавлять автоматически

Не коммитить без отдельного решения:

```text
*.pt
phef/logs/*
phef/experiments/train_recall.py.bak
docs/handoffs/* старые или черновые
results/*probe*.json
неполный log первой попытки Phase-B keys262144
неполный/неудачный batch=16 eval Early keys1048576
technical 4M b2 probe JSON
probeshim checkpoint
```

### 6.3. Оценочные ресурсы

[ФАКТ] Проверенные форматы:

| Режим | Память / результат |
|---|---|
| `n_keys=1048576`, `S=65536`, batch 16 | Не поместился; около 23060 MiB VRAM, allocator OOM |
| `n_keys=1048576`, `S=65536`, batch 8 | Успешно; наблюдалось около 13820 MiB VRAM |
| `n_keys=4194304`, `S=65536`, batch 2 | Успешно; наблюдалось около 10730 MiB VRAM |
| 4M full eval, batch 2 × 8192 | Выполнен; очень долгий, около 17 часов до JSON |

[ВЫВОД] До изменения evaluator не стоит планировать много дальнейших 4M+ full eval: отсутствие progress и checkpointed accumulators делает длительные запуски неудобными и уязвимыми к reboot.

---

## 7. Следующие задачи по приоритету

### 7.1. Сначала оформить результаты

1. Read-only проверить Git status и diff.
2. Проверить, какие full-eval JSON уже tracked.
3. Сделать commit только с явным whitelist JSON из §6.1.
4. Отдельно создать и закоммитить этот handoff после того, как он будет приведён в соответствие с фактическим Git status.

### 7.2. Следующее различающее обучение

Основной открытый вопрос:

```text
Что именно в Phase B сокращает capacity с >=2^22 до интервала 2^17…2^18?
```

Не менять одновременно несколько факторов.

Первый наиболее чистый кандидат:

```text
взять тот же Early checkpoint
оставить S=524288
оставить rw=0.125
оставить lr=1e-4
сократить Phase-B training duration
сохранять best checkpoint
измерять capacity на n_keys=131072 и n_keys=262144
```

[ГИПОТЕЗА] Если capacity ухудшается с накоплением обновлений второй фазы, более короткая Phase B сохранит больше capacity при том же long-S objective.

[НЕИЗВЕСТНО] Нельзя считать эту гипотезу установленной до нового run.

### 7.3. Альтернативные направления

- Full independent eval Early KD32 при `n_keys=2^23` только после отдельного memory probe; ожидается ещё более долгий run
- Reproducibility: повторить Early Phase A на новом train seed и оценить сначала на 1M или 4M ключей
- Точно локализовать Phase-B crossing внутри 131072–262144, если это нужно как численный benchmark
- Разделить влияние `S`, `rw`, lr и длительности через одну изменяемую переменную за run
- Улучшить evaluator: progress, периодические partial aggregate/checkpoint, resume; сначала read-only inspection, затем точный diff и явное согласие
- Исправить поле `expected_dist_uniform` в `probe_distance_profile.py`; текущее поле неверно и должно игнорироваться

---

## 8. Шаблоны команд

### 8.1. Read-only Git inventory

```bash
cd ~/qubitrino-projects/agi; git status --short; git diff --check; git log --oneline -8
```

### 8.2. Проверка JSON

```bash
cd ~/qubitrino-projects/agi/phef; /home/linux/qubitrino-projects/agi/venv/bin/python -c "import json; print(json.dumps(json.load(open('results/NAME.json')), ensure_ascii=False, indent=2, sort_keys=True))"
```

### 8.3. Full independent eval

```bash
cd ~/qubitrino-projects/agi/phef; mkdir -p logs; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; setsid nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/eval_recall_checkpoint.py --checkpoint results/CHECKPOINT.pt --n-keys K --n-values 64 --n-segments S --overwrite-p 0.3 --key-dim 32 --dim 128 --eval-batch-size B --eval-batches N --seed 7 --out results/TAG.json > logs/TAG.log 2>&1 < /dev/null & disown
```

Всегда выполнять:

```text
B × N = 16384
```

Перед запуском фиксировать criterion и получать отдельное явное подтверждение.

### 8.4. Живость долгого eval

```bash
cd ~/qubitrino-projects/agi/phef; date; pgrep -af 'eval_recall_checkpoint.py.*TAG' || true; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; tail -n 20 logs/TAG.log; ls -lh results/TAG.json 2>&1
```

Интерпретация:

- растут `times` в `ps`, есть `pcpu`, процесс есть в `nvidia-smi` — job вычисляется
- JSON существует — job завершён
- процесс исчез и JSON отсутствует — сначала читать log, не повторять автоматически
- повторные allocator OOM и стабильно занятая VRAM — не объявлять FAIL модели, сначала корректно остановить процесс после подтверждения

---

## 9. Краткая формулировка состояния

[ФАКТ] Early KD32 model, обученная с `S=65536`, `rw=1.0`, seed 7, проходит clean full independent hard eval на 16384 эпизодах при `n_keys=4194304`: real answer 0.978455, real retrieval 0.978271, shuffled answer 0.014038, masked answer 0.016296.

[ФАКТ] После long-S adaptation того же representation в Phase B (`S=524288`, `rw=0.125`, `lr=1e-4`) clean full PASS сохраняется при `n_keys=131072` — 0.910400/0.909302 — но уже FAIL при `n_keys=262144` — 0.878418/0.876099.

[ВЫВОД] Главный научный результат: PHEF может сформировать high-capacity key representation с минимум 4.19 млн ключей при `key_dim=32`; ограничение текущего long-S pipeline — не базовая representational capacity, а её сохранение во время адаптации на длинной последовательности.
