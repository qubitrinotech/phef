# PHEF — самодостаточный хендофф: S=131072, трёх-seed PASS, i915-риск и план S=262144

Дата передачи: 28 августа 2026.

Репозиторий: `~/qubitrino-projects/agi`.
Рабочий каталог PHEF: `~/qubitrino-projects/agi/phef`.
Ветка: `main`.
Последний отправленный commit: `d81b3c6 phef: add reproducible S=131072 recall results for seeds 4-6`.
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

- «[ФАКТ] `real answer=0.984009`» — если это число есть в присланном JSON/логе.
- «[ВЫВОД] Seed 4 проходит заранее заданный порог, потому что `real > 0.90`, controls `< 0.03125`» — если эти числа измерены.
- «[НЕИЗВЕСТНО] Peak VRAM при S=262144 неизвестен» — пока нет отдельного измерения именно этого режима.

### 0.2. Граница доступа

- Агент **не имеет прямого доступа** к машине пользователя, shell, Git, GPU, файлам, JSON, checkpoint `.pt`, конфигурациям и `/tmp`-логам.
- Все команды выполняет пользователь и присылает полный вывод.
- Нельзя говорить «в файле находится…», «GPU использует…», «процесс завершился…» без присланного подтверждения.

### 0.3. Не угадывать содержимое

Не угадывать содержимое:

- Python-кода;
- JSON и checkpoint `.pt`;
- конфигураций;
- системных журналов;
- `.gitignore`;
- логов обучения;
- статуса процессов;
- потребления VRAM, времени, throughput или температуры.

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

К действиям с изменением состояния относятся:

- создание/редактирование/удаление/перемещение файлов;
- `git add`, `git commit`, `git push`;
- запуск обучения/eval, создающий JSON/checkpoint/log;
- остановка, отключение, маскирование или удаление системного сервиса;
- изменение BIOS, GRUB, kernel parameters, драйверов, Xorg и т. п.

Не маскировать команду изменения в длинном блоке и не действовать по неявному «ок» там, где нужно точное подтверждение.

### 0.6. Гипотеза, опровергнутая замером

Если гипотеза опровергнута — прямо так и написать. Не воскрешать её без новых данных.

Особенно запрещено возвращаться к уже опровергнутым объяснениям «на глаз».

### 0.7. Никакой экстраполяции

Нельзя переносить между конфигурациями:

- качество;
- устойчивость по seed;
- скорость;
- peak VRAM;
- время обучения;
- поведение loss;
- оптимальный epoch;
- надёжность controls.

Каждая новая конфигурация, например S=262144 вместо S=131072, требует собственного probe, обучения и независимого eval.

### 0.8. Числовые расхождения — не объяснять «на глаз»

Если числа расходятся, причина — **[НЕИЗВЕСТНО]**, пока не проверена. Нужен минимальный конкретный тест.

Нельзя объявлять причиной:

- «случайность»;
- «влияние GPU»;
- «утечку»;
- «gate»;
- «LayerNorm»;
- «мало эпох»;
- «перегрузку CPU»;

без измерения, которое различает объяснения.

### 0.9. Формат ответов и команд

- Отвечать по-русски и кратко.
- Команды давать готовыми для копирования.
- Предпочитать один блок и одну строку запуска для длинных команд.
- Не давать много тяжёлых команд одновременно.
- Для рекурсивного `grep` всегда ограничивать область и вывод; не сканировать большие каталоги во время обучения.
- Если прежняя команда дала «простыню», остановить `Ctrl+C` и предложить короткую замену.

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

Долгие run запускать только через `nohup`, с логом и в фоне:

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u ... > /tmp/probe_logs/NAME.log 2>&1 &
```

Нельзя запускать обучение через pipe с `tail`.

Перед созданием лога в `/tmp/probe_logs` всегда включать `mkdir -p /tmp/probe_logs` в ту же команду. `/tmp` — непостоянное хранилище, поэтому обязательные воспроизводимые результаты сохраняются в `phef/results/` и коммитятся как JSON.

### 0.11. Heredoc, пути и shell

- Не склеивать heredoc создания файла и проверочные команды в один блок: ранее это повреждало окончание скрипта при вставке.
- Длинные команды с `LC_ALL=C` и `PYTHONPATH=.` давать **одной строкой**, без `\` и переносов.
- Учитывать текущий каталог: в Git-корне пути имеют префикс `phef/`, внутри `~/qubitrino-projects/agi/phef` — `results/...`, `experiments/...`, `phef/...`.
- Текстовые списки файлов и commit message не являются shell-командами. Для shell нужно давать именно `git add ...`, `git commit -m "..."` и т. п.

### 0.12. Правила Git

Перед commit обязательны:

```bash
git status --short
```

```bash
git diff --check
```

```bash
git diff --cached --check
```

```bash
git diff --cached --stat
```

```bash
git diff --cached
```

Также проверять синтаксис изменённых `.py`:

```bash
/home/linux/qubitrino-projects/agi/venv/bin/python -c "import ast; ast.parse(open('FILE.py').read()); print('ok FILE.py')"
```

Не добавлять без отдельного согласования:

- `.pt` checkpoints;
- `.bak`;
- временные скрипты;
- `/tmp`-логи;
- `logs/microtest/`;
- исторические/черновые результаты;
- handoff-файлы, не обновлённые под новый результат.

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
- Видеовывод desktop: Intel UHD 770 через `i915`/X11; NVIDIA отображается как `NVIDIA-G0` offload provider.
- Kernel на момент системной диагностики: `7.0.0-30-generic` Ubuntu.
- NVIDIA driver на момент системной диагностики: `595.84`, CUDA version shown by `nvidia-smi`: `13.2`.

---

## 2. Отправленные commits

| Commit | Содержание |
|---|---|
| `8e18019` | CPU-генерация slots в `world_gpu` |
| `150e9c7` | `batch()` возвращает `(slots, codebook)` вместо развёрнутых `event_keys` |
| `d440829` | pointer-doubling в `_sample_slots` вместо цикла по сегментам |
| `84dda15` | первый PASS малого масштаба: hard-eval, 1000 epochs |
| `3c712b6` | S=65536 PASS: no-fixed-point shuffle, hard-selection/checkpoint, auxiliary decoding loss, evaluator, seed-0 артефакты |
| `16e4cf0` | S=65536: добавлены воспроизводимые JSON для seed 1–2; seed-0 JSON уже были tracked в `3c712b6` |
| `d81b3c6` | S=131072: добавлены train/eval JSON для seeds 4–6 |

[ФАКТ] `git push origin main` для последнего commit завершился успешно: `16e4cf0..d81b3c6 main -> main`.

---

## 3. Проверенный метод и важный код

### 3.1. Целевой PHEF-режим

Проверенные успешные large-S run используют:

```text
n_keys=512
n_values=64
overwrite_p=0.3
key_dim=32
dim=128
batch_size=32
epochs=12000
lr=3e-4
hard_eval=True
eval_batch_size=16
use_best=True
acc_every=250
acc_n=512
retrieval_weight=1.0
weight_decay=1e-4
```

Для S=131072 также проверено:

```text
aux_value_weight=0.01
```

[ФАКТ] Независимый evaluator использует hard evaluation и `eval_batch_size=16`, `eval_batches=1024`, то есть 16384 эпизода.

### 3.2. `phef/phef/model.py`

[ФАКТ] `PHEF.forward` принимает:

```python
slots, codebook, values, timestamps, query_keys,
shuffle=False, mask_memory=False, hard=False,
return_retrieval=False
```

[ФАКТ] `hard=True` выбирает `scores.argmax(dim=1)` и value embedding выбранного сегмента.

[ФАКТ] Мягкое чтение материализует `memory_values` формы `(bs, n_segments, dim)` и при большом evaluation batch может OOM.

[ВЫВОД] Для large-S текущий целевой путь — hard retrieval и `eval_batch_size=16`. Не заявлять ничего о soft path без отдельного измерения.

### 3.3. Shuffle control

[ФАКТ] Ранее `shuffle=True` использовал `torch.randperm(batch_size)`, который допускает fixed points.

[ФАКТ] Теперь применяется случайный ненулевой циклический shift:

```python
shift = torch.randint(1, batch_size, (), device=values.device)
permutation = torch.roll(
    torch.arange(batch_size, device=values.device),
    shifts=int(shift.item()),
)
```

[ФАКТ] После исправления old checkpoint с прежним завышенным shuffled result дал shuffled answer около chance.

[ВЫВОД] Для всех текущих reported controls используется no-fixed-point shuffle. Не сравнивать с историческими score, полученными до исправления, как будто controls эквивалентны.

### 3.4. Auxiliary value loss

[ФАКТ] В `experiments/train_recall.py` есть параметр:

```text
--aux-value-weight
```

[ФАКТ] Default `--aux-value-weight` равен `0.0`; базовое поведение без флага сохранено.

[ФАКТ] При обучении auxiliary loss вычисляется как:

```python
aux_value_loss = F.cross_entropy(
    model.head(model.norm(model.value_emb.weight)),
    torch.arange(args.n_values, device=device),
)
```

и добавляется с весом `args.aux_value_weight`.

[ФАКТ] На S=65536 узкая абляция seed 3 показала independent PASS для весов `0.01`, `0.03` и `0.1`.

[ВЫВОД] Для S=131072 выбран `aux_value_weight=0.01` как минимальный **из проверенных успешных** весов, не как доказанный глобальный минимум.

### 3.5. Best checkpoint

[ФАКТ] `--best-out PATH.pt` сохраняет checkpoint с ключами `epoch`, `sel_acc`, `state_dict` при новом максимуме selection accuracy.

[ФАКТ] При `--use-best` скрипт восстанавливает лучший checkpoint перед финальной внутренней оценкой.

[ВЫВОД] На S=65536 и S=131072 финальная модель на epoch 12000 может деградировать, поэтому корректный проверенный протокол включает `--use-best` и независимый eval именно best checkpoint.

### 3.6. Evaluator

[ФАКТ] `experiments/eval_recall_checkpoint.py` загружает `.pt`, всегда включает hard evaluation, делает real/shuffled/masked controls и сохраняет JSON.

[ФАКТ] Его нужно запускать с `PYTHONPATH=.`.

[ВЫВОД] Не менять evaluator без read-only проверки актуального кода, точного diff и подтверждения пользователя.

---

## 4. Научные результаты

### 4.1. S=65536: первоначальный PASS seed 0

[ФАКТ] Seed 0 использовал:

```text
n_segments=65536
aux_value_weight=0.1
seed=0
```

[ФАКТ] Best checkpoint был сохранён на epoch 3500 с `sel_acc=1.0`.

[ФАКТ] Независимый eval на 16384 эпизодах:

| Метрика | Значение |
|---|---:|
| Real answer | 0.999329 |
| Real retrieval | 0.999329 |
| Shuffled answer | 0.014587 |
| Shuffled retrieval | 0.000977 |
| Masked answer | 0.013306 |
| Answer chance | 0.015625 |

### 4.2. S=65536: воспроизводимость seeds 0–2

[ФАКТ] Для той же конфигурации S=65536, `aux_value_weight=0.1`, independent eval каждого checkpoint на 16384 эпизодах дал:

| Seed | Best epoch | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Статус |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 3500 | 0.999329 | 0.999329 | 0.014587 | 0.000977 | 0.013306 | PASS |
| 1 | 4000 | 0.999573 | 0.999573 | 0.014099 | 0.000854 | 0.015747 | PASS |
| 2 | 3000 | 0.998047 | 0.997986 | 0.016235 | 0.000854 | 0.015137 | PASS |

[ВЫВОД] В точно проверенной конфигурации S=65536 PASS воспроизводится на seeds 0–2.

### 4.3. S=65536: абляция auxiliary loss

[ФАКТ] На seed 3 при всех прочих неизменных параметрах independent eval на 16384 эпизодах дал:

| `aux_value_weight` | Best epoch | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Статус |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.01 | 4750 | 0.999695 | 0.999695 | 0.015686 | 0.001038 | 0.016113 | PASS |
| 0.03 | 4000 | 0.999573 | 0.999573 | 0.014893 | 0.001038 | 0.015503 | PASS |
| 0.10 | 4000 | 0.999207 | 0.999146 | 0.015442 | 0.000977 | 0.015137 | PASS |

[ВЫВОД] В рамках узкой абляции одного seed `aux_value_weight=0.01` достаточен для PASS на S=65536.

[НЕИЗВЕСТНО] Не доказано, что 0.01 — глобально минимальный достаточный вес: ниже 0.01 не проверялось, и для 0.01 в этой абляции есть только seed 3.

### 4.4. S=131072: probe

[ФАКТ] Probe S=131072 запускался с seed 4, `aux_value_weight=0.01`, `batch_size=32`, hard-eval, 25 epochs.

[ФАКТ] Он завершился без CUDA OOM; GPU после завершения была свободна.

[ФАКТ] На 25 epochs `control_status=FAIL`, с near-chance real metric; best epoch 25, `sel_acc=0.013672`.

[ВЫВОД] Этот probe подтверждает только технический запуск без OOM для этой короткой конфигурации. Он не был тестом качества и не опровергает достижимость PASS после 12000 epochs.

[НЕИЗВЕСТНО] Peak VRAM и throughput S=131072 не были корректно измерены: `nvidia-smi` после завершения не показывает пик, а в логе не было elapsed time.

### 4.5. S=131072: трёх-seed independent PASS

[ФАКТ] Для следующей конфигурации:

```text
n_keys=512
n_values=64
n_segments=131072
overwrite_p=0.3
key_dim=32
dim=128
batch_size=32
epochs=12000
lr=3e-4
hard_eval=True
eval_batch_size=16
use_best=True
acc_every=250
acc_n=512
aux_value_weight=0.01
```

[ФАКТ] Каждый best checkpoint независимо оценён на 16384 эпизодах (`eval_batch_size=16`, `eval_batches=1024`):

| Seed | Best epoch | Checkpoint sel_acc | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Статус |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 9750 | 0.990234 | 0.984009 | 0.989380 | 0.016113 | 0.000793 | 0.014282 | PASS |
| 5 | 9750 | 0.980469 | 0.978882 | 0.991028 | 0.015625 | 0.000977 | 0.016663 | PASS |
| 6 | 10000 | 0.972656 | 0.958496 | 0.960266 | 0.015442 | 0.000610 | 0.015381 | PASS |

[ФАКТ] Answer chance равен `1/64 = 0.015625`; критерий PASS для controls равен `2 * chance = 0.03125`.

[ВЫВОД] В точно проверенном режиме S=131072, `aux_value_weight=0.01`, hard retrieval и seeds 4–6 воспроизводимо проходят критерий: real answer/retrieval > 0.90, shuffled/masked answer < 0.03125.

[ВЫВОД] Корректная формулировка:

> PHEF при `n_keys=512`, `n_values=64`, `n_segments=131072`, `overwrite_p=0.3`, `key_dim=32`, `dim=128`, hard retrieval и `aux_value_weight=0.01` воспроизводимо проходит segmented key–value recall на трёх seeds. Независимая оценка каждого best checkpoint на 16384 эпизодах дала real answer 0.958496–0.984009 и real retrieval 0.960266–0.991028; shuffled и masked answer controls находятся около chance 1/64.

[НЕИЗВЕСТНО] Результат не доказывает PASS для S>131072, soft-read, других `overwrite_p`, иных model sizes, других n_keys/n_values, или всех будущих seeds.

---

## 5. Установленные причины и опровергнутые гипотезы

### 5.1. Старый checkpoint с answer около 0.50

[ФАКТ] Старый checkpoint `cap_k512_s65536_best3.pt` имел retrieval около 0.999878, но answer около 0.504333.

[ФАКТ] Его штатная head корректно декодировала лишь 35 из 64 value embeddings; свежая линейная head на фиксированных `value_emb` достигала 100% уже к 10 шагу.

[ФАКТ] Усиление gate не устраняло проблему; `head(memory)` и `head(LayerNorm(memory))` были около 0.558.

[ВЫВОД] Bottleneck был не в retrieval, LayerNorm, gate или разделимости embeddings, а в недостаточном прямом обучающем сигнале штатной head к декодированию value embeddings.

### 5.2. Не воскрешать без новых замеров

1. «В S=65536 нет тайбрейкера» — опровергнуто: `recency_scale` обучается, адресация доходила примерно до 1.0.
2. «Head принципиально не может декодировать value embeddings» — опровергнуто fresh-head test.
3. «LayerNorm — причина answer около 0.50» — опровергнуто близкими `head(memory)` и `head(LayerNorm(memory))`.
4. «Gate — единственная причина около 0.50» — опровергнуто: его усиление не решило проблему.
5. «Высокий shuffled score доказывает query leakage» — опровергнуто как необходимое объяснение: fixed points старого shuffle полностью объясняли завышение; после no-fixed-point shuffle score стал около chance.
6. «12000 epochs обязательно улучшает large-S» — опровергнуто: поздние epochs могут деградировать; нужен best checkpoint.
7. «Мягкое чтение — адекватный selection signal при S=65536» — не подтверждено; текущий large-S режим требует hard selection.
8. «Коллапс bf16 неизбежен около S=4096–16384» — не подтверждено.
9. «Cross-entropy неизбежно разгоняет scale при lr=3e-4» — не подтверждено прежними измерениями.

---

## 6. Системная стабильность: обязательный контекст

### 6.1. Что наблюдалось

[ФАКТ] Во время длительных экспериментов Ubuntu дважды показывала экран с `KERNEL / Please reboot / Fatal exception`.

[ФАКТ] В системном журнале нет подтверждённых NVIDIA Xid, PCIe AER fatal, MCE или OOM-killer событий, которые прямо связывали бы падение с RTX 4090, VRAM, CPU или системной RAM.

[ФАКТ] В журнале неоднократно зафиксированы события Intel iGPU:

```text
i915 ... GPU HANG
i915 ... rcs0 reset request timed out
i915 ... stopped heartbeat on rcs0
i915 ... Xorg[...] context reset due to GPU hang
```

[ФАКТ] i915 error state показывал `GPU HANG ... in Renderer [8655]`, platform `ALDERLAKE_S`, PCI ID `0xa780`; PID 8655 уже завершился до идентификации.

[ФАКТ] `/dev/dri/card1` и `/dev/dri/renderD128` использовали Xorg, GNOME Shell, Firefox и несколько процессов Saby. Saby имеет CEF renderer processes и ранее многократно выдавал `general protection fault` в `libstdc++.so.6.0.33`.

[ФАКТ] PHEF использует CUDA RTX 4090. В `nvidia-smi` RTX 4090 — compute GPU; desktop X11 использует Intel `i915` как primary framebuffer. Во время seed 5 RTX 4090 показывала около 3799 MiB allocated, температуру 49°C и power около 76.89 W, без NVIDIA Xid в присланных строках.

[ВЫВОД] По представленным данным i915/Xorg — наиболее подтверждённая текущая системная проблема. Нельзя утверждать, что PHEF стал непосредственной причиной kernel crash; PHEF мог лишь совпасть по времени с нагрузкой.

### 6.2. Ollama: устранённый фактор

[ФАКТ] До удаления `ollama.service` бесконечно пытался запускать отсутствующий `/usr/local/bin/ollama` каждые 3 секунды (`Restart=always`, `RestartSec=3`, `status=203/EXEC`). В одной предыдущей загрузке счётчик restart превысил 29000.

[ФАКТ] Пользователь подтвердил и выполнил удаление остатков Ollama. После этого:

- unit `ollama.service` не найден;
- `/usr/local/bin/ollama` отсутствует;
- `/usr/local/lib/ollama` удалён;
- user/group `ollama` отсутствуют;
- `/home/linux/ollama-models-backup.txt` сохранён.

[ВЫВОД] Restart-loop Ollama устранён. Это уменьшило фоновый шум и нагрузку, но не устранило i915 GPU hangs.

### 6.3. Практическое правило до исправления i915

- Не запускать PHEF одновременно с DIEL или другим CUDA workload.
- Не запускать тяжёлые диагностические `grep -R` во время обучения.
- По возможности закрывать/минимизировать тяжёлые Firefox вкладки, видео и WebGL во время run.
- Не использовать UI Saby во время run; не останавливать Saby-сервисы без отдельной проверки и явного подтверждения, так как они могут быть рабочими.
- Не менять BIOS, GRUB, kernel, i915, Xorg или NVIDIA driver без отдельной диагностики, точного плана, возможности восстановления/SSH и явного подтверждения.

[ВЫВОД] Для дальнейших многочасовых S>131072 наиболее безопасны: headless/SSH запуск с минимальной графической нагрузкой на Intel iGPU либо перевод desktop/монитора на RTX 4090. Это рекомендация, а не уже выполненное изменение.

### 6.4. Если система снова падает

После перезагрузки, **до запуска нового обучения**, выполнить и прислать короткий вывод:

```bash
sudo journalctl -b -1 -k --no-pager -o short-precise | tail -250
```

Не запускать новый run до анализа этого вывода.

---

## 7. Первое действие в новом чате: read-only сверка

Перед любым новым обучением, code change или commit выполнить только read-only проверку и прислать полный вывод:

```bash
cd ~/qubitrino-projects/agi && git status --short && git log --oneline -8 && git branch -vv && git diff --check && git diff -- phef/phef/model.py phef/experiments/train_recall.py phef/experiments/eval_recall_checkpoint.py && cd phef && for f in phef/model.py phef/world.py experiments/train_recall.py experiments/eval_recall_checkpoint.py; do /home/linux/qubitrino-projects/agi/venv/bin/python -c "import ast; ast.parse(open('$f').read()); print('ok $f')"; done && /home/linux/qubitrino-projects/agi/venv/bin/python - <<'PY'
import json, os, torch
paths = [
    "results/cap_k512_s131072_aux001_seed4.json",
    "results/cap_k512_s131072_aux001_seed4_eval16384.json",
    "results/cap_k512_s131072_aux001_seed5.json",
    "results/cap_k512_s131072_aux001_seed5_eval16384.json",
    "results/cap_k512_s131072_aux001_seed6.json",
    "results/cap_k512_s131072_aux001_seed6_eval16384.json",
]
for p in paths:
    print("\n===", p, "===")
    print("exists=", os.path.exists(p), "bytes=", os.path.getsize(p) if os.path.exists(p) else None)
    if os.path.exists(p):
        with open(p) as f:
            print(json.dumps(json.load(f), indent=2, sort_keys=True))
for p in [
    "results/cap_k512_s131072_aux001_seed4.pt",
    "results/cap_k512_s131072_aux001_seed5.pt",
    "results/cap_k512_s131072_aux001_seed6.pt",
]:
    print("\n===", p, "===")
    print("exists=", os.path.exists(p), "bytes=", os.path.getsize(p) if os.path.exists(p) else None)
    if os.path.exists(p):
        c = torch.load(p, map_location="cpu", weights_only=True)
        print("keys=", sorted(c))
        print("epoch=", c.get("epoch"), "sel_acc=", c.get("sel_acc"), "state_dict_tensors=", len(c.get("state_dict", {})))
PY
```

[ФАКТ] На момент прошлых проверок в рабочем дереве имелось много untracked старых results, handoff-файлов, `train_recall.py.bak` и `logs/microtest/`. Они намеренно не включались в commits.

[ВЫВОД] Не удалять и не добавлять эти файлы без отдельного решения пользователя.

---

## 8. Следующий научный шаг: S=262144

### 8.1. Научная цель

[ВЫВОД] Следующий разумный масштаб — `n_segments=262144` при неизменных:

```text
n_keys=512
n_values=64
overwrite_p=0.3
key_dim=32
dim=128
batch_size=32
lr=3e-4
hard_eval=True
eval_batch_size=16
use_best=True
acc_every=250
acc_n=512
aux_value_weight=0.01
```

[НЕИЗВЕСТНО] PASS на S=131072 не доказывает достижимость, скорость, peak VRAM, best epoch или устойчивость на S=262144.

### 8.2. Предлагаемый строгий протокол

#### Шаг A. Системная готовность

Перед запуском убедиться read-only командой:

```bash
echo "=== CUDA PROCESSES ===" && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 && echo "=== PHEF / DIEL ===" && pgrep -af "train_recall.py|eval_recall_checkpoint.py|DIEL|diel" || true && echo "=== KERNEL WARNINGS ===" && sudo journalctl -b -k --no-pager | grep -iE "i915.*(GPU HANG|reset|heartbeat|context reset)|nvrm.*xid|aer.*(error|fatal)|mce|hardware error|kernel panic|oom-kill" | tail -100 || true
```

[ВЫВОД] Не запускать S=262144 одновременно с DIEL или другим CUDA workload. При новых i915 hang/error сначала решить вопрос устойчивости среды.

#### Шаг B. Short technical probe

После явного подтверждения создать отдельный probe с новым seed и отдельными именами JSON/checkpoint/log.

Предварительный предлагаемый probe, если read-only сверка подтвердит актуальность интерфейса:

```text
n_segments=262144
batch_size=32
epochs=25
aux_value_weight=0.01
hard_eval=True
eval_batch_size=16
```

[ВЫВОД] Цель probe: проверить, что режим запускается, не OOM и корректно сохраняет артефакты. Не интерпретировать 25-epoch quality как тест PASS.

[НЕИЗВЕСТНО] `batch_size=32` может оказаться неприемлемым на S=262144. Не менять его заранее без фактического OOM/успеха probe.

#### Шаг C. Full training одного seed

Только после штатного probe — один full run 12000 epochs, `--use-best`, отдельный best checkpoint.

[ВЫВОД] Не называть это PASS до независимого evaluator на 16384 эпизодах.

#### Шаг D. Independent eval

Если checkpoint создан, запустить:

```text
eval_batch_size=16
eval_batches=1024
hard evaluation
real / shuffled / masked controls
```

PASS criterion:

```text
real answer > 0.90
real retrieval > 0.90
shuffled answer < 0.03125
masked answer < 0.03125
```

#### Шаг E. Reproduction

Только после independent PASS seed A запустить ещё два independent seed B/C по полностью идентичной конфигурации, затем independent eval каждого checkpoint.

### 8.3. Запрет на преждевременные заявления

До выполнения шагов B–E нельзя говорить:

- «S=262144 работает»;
- «S=262144 не работает»;
- «нужно уменьшить batch size»;
- «нужно увеличить epochs»;
- «0.01 недостаточно»;
- «OOM гарантирован/невозможен»;
- «скорость будет в N раз ниже».

Все эти утверждения — **[НЕИЗВЕСТНО]** до измерения.

---

## 9. Полезные шаблоны команд

### 9.1. Мониторинг long run

Подставить точное имя run:

```bash
cd ~/qubitrino-projects/agi/phef && echo "=== PROCESS ===" && pgrep -af "train_recall.py.*RUN_NAME" || true && echo "=== LAST EPOCH ===" && grep -E "^epoch " /tmp/probe_logs/RUN_NAME_train.log | tail -1 || true && echo "=== LOG TAIL ===" && tail -30 /tmp/probe_logs/RUN_NAME_train.log && echo "=== GPU ===" && nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader && echo "=== KERNEL ERRORS ===" && sudo journalctl -b -k --no-pager | grep -iE "i915.*(GPU HANG|reset|heartbeat|context reset)|nvrm.*xid|aer.*(error|fatal)|mce|hardware error|kernel panic|oom-kill" | tail -50 || true
```

### 9.2. Признаки штатного завершения

[ФАКТ] Штатное окончание training run подтверждается одновременно:

- `pgrep` больше не показывает процесс обучения;
- лог содержит `восстановлен best checkpoint`;
- лог содержит блок `=== CONTROLS ===`;
- лог содержит `control status`;
- лог содержит `сохранено: results/...json`;
- соответствующие JSON и `.pt` существуют.

### 9.3. После независимого eval

Проверять:

- процесс eval завершён;
- лог содержит `saved: results/...eval16384.json`;
- JSON существует и имеет `hard_eval: true`;
- config JSON соответствует точному run;
- число episodes в `real`, `shuffled`, `masked` равно 16384;
- применён критерий PASS без изменения порогов.

---

## 10. Рекомендуемые файлы для нового чата

Минимально приложить:

1. Этот handoff.
2. `phef/phef/model.py`.
3. `phef/phef/world.py`.
4. `phef/experiments/train_recall.py`.
5. `phef/experiments/eval_recall_checkpoint.py`.
6. `phef/results/cap_k512_s131072_aux001_seed4.json`.
7. `phef/results/cap_k512_s131072_aux001_seed4_eval16384.json`.
8. `phef/results/cap_k512_s131072_aux001_seed5.json`.
9. `phef/results/cap_k512_s131072_aux001_seed5_eval16384.json`.
10. `phef/results/cap_k512_s131072_aux001_seed6.json`.
11. `phef/results/cap_k512_s131072_aux001_seed6_eval16384.json`.
12. При необходимости диагностики/воспроизведения: соответствующие `.pt` checkpoints, но не коммитить их автоматически.

Полезные исторические документы:

- `docs/handoffs/18_PHEF_HANDOFF.md`.
- `docs/handoffs/19_PHEF — хендофф ёмкость памяти и устойчивость обучения.md`.
- `docs/handoffs/20_PHEF-handoff-S65536-checkpoint-shuffle-control.md`.
- `docs/handoffs/21_PHEF_handoff_S65536_PASS.md`.

Не обязательно прикладывать:

- `train_recall.py.bak`;
- `/tmp`-логи;
- old smoke results;
- временные диагностические скрипты;
- `logs/microtest/`.

---

## 11. Краткая стартовая инструкция новому чату

1. Прочитать этот документ.
2. Не делать предположений.
3. Выполнить только read-only сверку из §7.
4. Сверить актуальные файлы, Git status и JSON S=131072.
5. Перед любым новым артефактом согласовать точные параметры, имена результатов/checkpoint/log и критерий PASS.
6. Если среда стабильна — начать с отдельного technical probe S=262144.
7. Не коммитить новые результаты до отдельной независимой проверки и явного подтверждения пользователя.
