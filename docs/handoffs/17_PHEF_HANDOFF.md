# PHEF — хендофф №17

Дата: 24 августа 2026, 20:36 MSK
Проект: `~/qubitrino-projects/agi/phef`
Git-корень: `~/qubitrino-projects/agi` (в `git diff` пути начинаются с `phef/...`)
Машина: `linux-desktop2`, RTX 4090 24564 MiB, драйвер 595.84
Единственный допустимый Python: `/home/linux/qubitrino-projects/agi/venv/bin/python`

Статус: **все три открытых технических вопроса хендоффа №16 закрыты фактическими замерами. Прогон 131072 на 2500 эпох завершён успешно: 40000 эпох на этом масштабе не нужны. Замерено, что 94.5% времени эпохи уходит на генерацию данных в `batch()`. Пик памяти `analyze` линеен по `--eval-batch-size` и не зависит от `--eval-batches`. Эквивалентность правки `(slots, codebook)` подтверждена на CPU до машинного эпсилон. Шесть коммитов запушены, `origin/main` = `720c600`. GPU свободна.**

---

## 0. ГЛАВНОЕ ПРАВИЛО

# НЕ ГАДАТЬ. НЕ ФАНТАЗИРОВАТЬ. НЕ НАДУМЫВАТЬ.

Это правило важнее скорости ответа и удобства.

1. **Не догадываться о содержимом кода.** Нужен участок файла — дать пользователю команду `sed`/`grep`, дождаться вывода. Не строить логику на предположении о функции.
2. **Не догадываться о формате файлов.** Перед использованием каждого JSON заново прочитать `json.load` и показать фактические ключи. Перед использованием каждого `.pt` заново выполнить `torch.load` и показать фактические ключи.
3. **Не экстраполировать память и скорость между конфигурациями.** Только замер конкретного прогона.
4. **Не объяснять расхождения рассуждением.** Если числа не сходятся — писать `[НЕИЗВЕСТНО]` и предложить конкретную проверку.
5. **Не помнить поведение фреймворка.** Если это важно — проверить микротестом.
6. **Всегда сверять config целевого JSON, а не текст хендоффа.**
7. **Помечать каждое утверждение:** `[ФАКТ]` — прочитано из вывода команды в этой сессии; `[ВЫВОД]` — расчёт/следствие, не замер; `[НЕИЗВЕСТНО]` — данных нет.
8. **Агент не имеет доступа к машине.** Все команды выполняет пользователь. Ничего не запускать без явного подтверждения и запрашивать вывод.
9. **Запуски только через** `nohup ... > файл.log 2>&1 &`, **никогда не через** `| tail`. Перед Python обязательно:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
10. **Интерпретатор строго** `/home/linux/qubitrino-projects/agi/venv/bin/python`.
11. `tail` запрещён в пайпе с запускаемым обучением. Для чтения уже записанного лога использовать `grep`/`sed`.
12. Не создавать, не коммитить, не удалять, не перемещать артефакты без отдельного подтверждения.
13. При создании каталогов для `--out`/лога всегда включать `mkdir -p` в тот же блок команд.
14. **Новое:** не использовать `sed -n "$(( $(wc -l < файл) - 19 )),\$p"` — в этой сессии конструкция упала с `sed: неверный ключ — «1»`. Для чтения хвоста лога использовать `sed -n 'N,$p'` с явным числом или `grep -n`.

---

## 1. Структура путей — исправление хендоффа №16

[ФАКТ] `wc -l phef/model.py` из cwd `~/qubitrino-projects/agi/phef` → `122`. `phef/phef/world_gpu.py` из того же cwd → «Нет такого файла или каталога».

[ВЫВОД] Вложенности `phef/phef/` относительно каталога `phef` не существует. Ссылки хендоффа №16 вида `phef/phef/model.py` даны от git-корня. Из cwd `~/qubitrino-projects/agi/phef` это `phef/model.py` и `phef/world_gpu.py`.

[ФАКТ] Корень `~/qubitrino-projects/agi`: `docs/`, `.git/`, `.gitignore`, `phef/`, `pyproject.toml`, `README.md`, `venv/`.

[ФАКТ] `~/qubitrino-projects/agi/phef`: `phef/`, `experiments/`, `logs/`, `output/`, `results/`, `collect_summary.py`, `requirements.txt`, 14 файлов `run_*.{sh,log}`.

[ФАКТ] `VIRTUAL_ENV=/home/linux/qubitrino-projects/agi/phef/.venv`, при этом каталога не существует. Осколок оболочки, открытой до удаления `.venv` в прошлой сессии. Идущим прогонам не мешает — они стартуют с абсолютным путём. В новых терминалах venv не активировать, всегда писать абсолютный путь к интерпретатору.

---

## 2. Ловушка `--ckpt-every`: срезы не писались

[ФАКТ] Фактический код `experiments/analyze_memory.py`, строки 592 и 634–636:
```python
if args.use_best and epoch % args.acc_every == 0:
    ...
    if (
        args.ckpt_every > 0
        and (epoch // args.acc_every) % args.ckpt_every == 0
    ):
```

[ФАКТ] `help` аргумента (строка 394): `chekpoint kazhdye N zamerov acc-every; 0 - vyklyuchit`, `default=10`.

[ВЫВОД] `--ckpt-every` считает **замеры**, не эпохи. При `acc_every=250` и `ckpt_every=250` условие истинно на эпохе 62500. Прогон на 2500 эпох не записал бы ни одного среза.

[ФАКТ] Первый прогон (PID 1779054, старт 23 августа 22:35:59) шёл с `--ckpt-every 250`. На epoch 250 в логе появилась строка `epoch 250 sel_acc 0.8931`, а каталог `results/traj131072/` остался пуст — подтверждение вывода замером.

[ФАКТ] Прогон убит `kill -TERM 1779054` в 00:14, GPU освободилась до 15 MiB. Перезапущен как PID 1849356 в 00:14:12 с `--ckpt-every 1`. На epoch 250 появилась строка `ckpt @ 250: last + best (best epoch 250)` и три файла.

[ФАКТ] `save_ckpt` (строки 334–350) пишет `payload` с ключами `state_dict`, `config`, `best_acc`, `best_epoch`, `escape_epoch`, `selection`, `epoch`, через `tmp` + `os.replace` — атомарно. Оптимизатора и RNG нет, resume невозможен.

[ФАКТ] Пути срезов: `base = os.path.splitext(args.out)[0]`, далее `base + "_last.pt"`, `base + "_best.pt"`, `dump_history(base + "_history.json", history)`.

[ВЫВОД] Правило на будущее: для среза каждые `acc_every` эпох ставить `--ckpt-every 1`. Проверять до запуска, что `acc_every * ckpt_every` меньше `epochs`.

---

## 3. Главный результат: 40000 эпох на 131072 не нужны

[ФАКТ] Прогон PID 1849356: 64 × 4, `n_segments=131072`, `init_weights` = `results/capacity_scan_dim128_curr32768/dim128_seg32768_s0_e40k_curr.pt`, `epochs 2500`, `acc-every 250`, `acc-n 2048`, `ckpt-every 1`, `select-metric soft`, `seed 0`, `eval-batch-size 64`, `eval-batches 80`. Старт 00:14:12, обучение до 10:18:14, финал 10:22:23.

[ФАКТ] Полная траектория из `results/traj131072/seg131072_e2500_ckpt1_history.json` (`list` из 10 элементов):

| epoch | sel_acc | loss | score_scale | recency_scale | memory_gate |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.89307 | 0.21904 | 5.58038 | 99.39031 | 16.62303 |
| 500 | 0.90723 | 0.44840 | 5.59761 | 99.62740 | 16.56470 |
| 750 | 0.89404 | 0.33377 | 5.57577 | 99.86523 | 16.52856 |
| 1000 | 0.89453 | 0.37648 | 5.58301 | 100.10176 | 16.51676 |
| 1250 | 0.89795 | 0.31155 | 5.57793 | 100.33955 | 16.50228 |
| 1500 | 0.89502 | 0.34557 | 5.57316 | 100.57940 | 16.49442 |
| 1750 | 0.90674 | 0.34831 | 5.58024 | 100.81160 | 16.51278 |
| 2000 | 0.8979 | 0.4206 | — | — | — |
| 2250 | 0.9126 | — | — | — | — |
| 2500 | 0.9067 | 0.3157 | — | — | — |

Точки 2000–2500 приведены по логу; полные поля этих трёх точек в `_history.json` есть, но в этой сессии распечатаны не были — `[НЕИЗВЕСТНО]` их `score_scale`/`recency_scale`/`memory_gate`.

[ФАКТ] Финальный JSON `results/traj131072/seg131072_e2500_ckpt1.json`, 35 корневых полей:

| Поле | Значение |
|---|---:|
| `escape_epoch` | 250 |
| `answer_acc` | 0.999609375 |
| `answer_acc_soft` | 0.9060546875 |
| `retrieval_acc` | 0.999609375 |
| `retrieval_value_acc` | 0.999609375 |
| `oracle_bound` | 1.0 |
| `tie_rate` | 0.0 |
| `tie_rate_mean` | 0.0 |
| `hard_soft_gap` | 0.0935546875 |
| `entropy_norm` | 0.1652896970510483 |
| `best_acc` | 0.91259765625 |
| `best_epoch` | 2250 |
| `best_loss` | 0.27979637309908867 |
| `score_scale` | 5.582423210144043 |
| `recency_scale` | 101.28172302246094 |
| `memory_gate` | 16.54851722717285 |
| `selection` | `val_accuracy` |
| `select_metric` | `soft` |
| `acc_recent` | 0.999609375 |
| `acc_remote` | 0.999609375 |
| `drop_recent_remote` | `None` |
| `weights_path` | `results/traj131072/seg131072_e2500_ckpt1.pt` |

[ФАКТ] Полный список 35 ключей (тот же набор, что у `timing_probe`): `acc_by_distance`, `acc_recent`, `acc_remote`, `answer_acc`, `answer_acc_soft`, `best_acc`, `best_epoch`, `best_loss`, `chance`, `chance_loss`, `config`, `count_by_distance`, `drop_recent_remote`, `entropy_norm`, `escape_criterion`, `escape_epoch`, `escape_threshold`, `hard_soft_gap`, `history`, `memory_gate`, `oracle_bound`, `oracle_bounds`, `recency_scale`, `retrieval_acc`, `retrieval_value_acc`, `score_scale`, `select_metric`, `selection`, `slot_acc_by_distance`, `slot_acc_recent`, `slot_acc_remote`, `slot_drop_recent_remote`, `tie_rate`, `tie_rate_mean`, `weights_path`.

### Сравнение трёх масштабов обучения на 131072

| Поле | 25 эпох | 250 эпох | 2500 эпох |
|---|---:|---:|---:|
| `answer_acc` | 1.0 | 1.0 | 0.999609375 |
| `answer_acc_soft` | 0.89668 | 0.90371 | 0.90605 |
| `hard_soft_gap` | 0.10332 | 0.09629 | 0.09355 |
| `best_acc` | 0.90625 | 0.89307 | 0.91260 |
| `best_epoch` | 10 | 250 | 2250 |
| `recency_scale` | 99.15889 | 99.39031 | 101.28172 |
| `entropy_norm` | 0.16702 | 0.16683 | 0.16529 |
| `memory_gate` | 16.63581 | 16.62303 | 16.54852 |

[ВЫВОД] `sel_acc` колеблется в коридоре 0.89307…0.91260, размах 0.0195, без направленного роста и без монотонного участка длиннее двух точек. Стандартная ошибка биномиальной оценки при p≈0.9, n=2048 равна ≈0.0066 — весь разброс укладывается в ±1.5 сигмы.

[ВЫВОД] `best_acc 0.91260` на epoch 2250 превышает 0.90723 на epoch 500 всего на 0.0054, то есть меньше одной сигмы. Как свидетельство обучения — слабое.

[ВЫВОД] `answer_acc` впервые **не** 1.0: 0.999609375 = 1 − 2/5120, два промаха из 5120 эпизодов. Более долгое обучение дало результат хуже, чем 25 эпох.

[ВЫВОД] `hard_soft_gap` сжимается монотонно, но затухающе: 0.10332 → 0.09629 → 0.09355. Ответ на вопрос №16: да, сжимается, скорость падает примерно в 10 раз на порядок эпох.

[ВЫВОД] `recency_scale` дрейфует **линейно вверх** на всех точках: шаг ≈0.237 за 250 эпох, то есть ≈0.00095 на эпоху. Экстраполяция на 40000 эпох дала бы ≈137 против эталонных 99.149 у 32768. `[НЕИЗВЕСТНО]` сохраняется ли линейность и вредит ли это точности — при дрейфе на 1.9 единицы `sel_acc` не изменился.

[ВЫВОД] `score_scale` стоит: размах 0.0244 на 5.58, то есть 0.4%. `memory_gate` монотонно убывает шесть точек, затем отскакивает.

[ФАКТ] Независимая проверка: прогон на **5 эпох** от весов `seg131072_e2500_ckpt1.pt` дал `hard_soft_gap = +0.093359` против 0.093555 у финала 2500 эпох.

[ВЫВОД] Разрыв определяется геометрией весов, а не длительностью обучения — воспроизводится за 5 эпох до четвёртого знака.

### Тревожная находка: margin в геометрии модели

[ФАКТ] `oracle_bounds` финального JSON:
```text
enc_m0.0  = 0.996875    m0.0  = 1.0    mg_m0.0  = 0.273828
enc_m0.02 = 0.979883    m0.02 = 1.0    mg_m0.02 = 0.273828
enc_m0.05 = 0.881445    m0.05 = 1.0    mg_m0.05 = 0.273633
enc_m0.1  = 0.436523    m0.1  = 1.0    mg_m0.1  = 0.273242
raw_top1  = 1.0         mg_top1 = 0.999609    score_top1 = 0.999609
```

[ФАКТ] `oracle_bound` (тело прочитано, строки 20–61) считает долю эпизодов, где `target_similarity > best_distractor + margin`, при `same_slot = slots.eq(query_slots.unsqueeze(1))` и `distractor_similarity = similarity.masked_fill(same_slot, -2.0)`.

[ВЫВОД] В геометрии модели (`dot / sqrt(dim)`, без нормировки) целевой ключ строго отделён от лучшего дистрактора лишь в **27.4%** эпизодов, при top-1 попадании 99.96%. Разница `mg_m0.0` и `mg_m0.1` равна 0.0006 — там, где margin есть, он сразу большой; в остальных 72.6% его нет.

[ВЫВОД] Сырые unit-ключи разделимы идеально (`m0.* = 1.0`, `raw_top1 = 1.0`), в нормированной геометрии энкодера деградация плавная (`enc_m0.1 = 0.4365`). Значит проблема в масштабе выходов энкодера, а не в самих ключах.

[НЕИЗВЕСТНО] Причина. Возможная связь: `recency_scale = 101.28` против `score_scale = 5.58` — recency-член в 18 раз крупнее semantic-члена, решение может приниматься по времени, а не по смыслу. Проверять отдельным экспериментом, не рассуждением.

### Дефект метрики по дистанции

[ФАКТ] `acc by distance` финального JSON: bucket 0 → `answer 0.9996`, `slot_hit 0.9996`, `n = 5120`; buckets 1–7 → `nan`, `n = 0`. `drop_recent_remote = None`.

[ВЫВОД] На 131072 сегментах вся масса запросов попадает в одно ведро, разбивка неинформативна. Отдельный дефект, требует пересмотра границ бакетов для больших `n_segments`.

---

## 4. Задача A: узкое место замерено, правка обоснована

### Замер `batch(64)` на 131072

[ФАКТ] Скрипт `/tmp/probe_batch.py`, лог `logs/probe_batch/batch64_seg131072.log`. Три вызова `BranchingWorldGPU.batch(64)` при `n_keys=512`, `n_values=64`, `n_segments=131072`, `key_dim=32`, `seed=0`, `device='cuda'`:
```text
call 0: 3497.2 ms  peak 1260.0 MiB  alloc_after 1216.0 MiB
call 1: 3475.7 ms  peak 1260.0 MiB  alloc_after 1216.0 MiB
call 2: 3382.9 ms  peak 1260.0 MiB  alloc_after 1216.0 MiB
shapes [(64,131072,32), (64,131072), (64,131072), (64,32), (64,), (64,)]
dtypes ['torch.float32','torch.int64','torch.int64','torch.float32','torch.int64','torch.int64']
```

[ВЫВОД] Установившееся время ≈ **3.42 с** на вызов. Прогрева нет — первый вызов лишь на 3% дольше.

[ВЫВОД] При `accum_steps=4` четыре вызова на эпоху дают **13.7 с из замеренных 14.50 с/эпоху = 94.5%**. На forward + backward + шаг AdamW остаётся ≈0.8 с. Диагноз хендоффа №16 подтверждён количественно.

[ВЫВОД] Раскладка 1216 MiB: `event_keys` 1024 MiB (64×131072×32×4), `values` 64 MiB, `timestamps` 64 MiB, `slots` 64 MiB (все три int64, 64×131072×8). Пик 1260 MiB превышает остаток на 44 MiB — временные `match`, `rev`.

[ВЫВОД] Побочные находки для лёгкой оптимизации: `timestamps` создан как `torch.arange(s).expand(bs, s).contiguous()` — 64 копии одного вектора, экономия 63 MiB доступна без изменения семантики. `values` в int64 при `n_values=64` влезает в int8, экономия 56 MiB, но потребует каста в long перед `value_emb`.

### Фактический код `batch()`

[ФАКТ] `phef/world_gpu.py`, строки 85–140:
```python
def batch(self, bs):
    s = self.n_segments
    dev = self.device
    slots = self._sample_slots(bs)
    rows = torch.arange(bs, device=dev)
    values = self._randint(self.n_values, (bs, s))
    timestamps = torch.arange(s, device=dev).expand(bs, s).contiguous()
    pick = self._randint(s, (bs,))
    query_slots = slots[rows, pick]
    match = slots == query_slots.unsqueeze(1)
    rev = torch.flip(match, dims=[1]).float()
    answer_positions = s - 1 - rev.argmax(dim=1)
    answers = values[rows, answer_positions]
    self.last_slots = slots
    self.last_query_slots = query_slots
    self.last_answer_positions = answer_positions
    if self.symbolic:
        return (slots, values, timestamps, query_slots, answers, answer_positions)
    codebook = torch.randn((bs, self.n_keys, self.key_dim), generator=self.gen, device=dev)
    codebook = codebook / codebook.norm(dim=-1, keepdim=True)
    event_keys = torch.gather(codebook, 1, slots.unsqueeze(-1).expand(bs, s, self.key_dim))
    query_keys = codebook[rows, query_slots]
    return (event_keys, values, timestamps, query_keys, answers, answer_positions)
```

[ФАКТ] Нормировка `codebook` идёт по последней оси **до** `gather`.
[ФАКТ] `slots`, `query_slots`, `answer_positions` уже сохраняются в `self.last_*` и читаются оттуда в анализе (строки 184–185 `analyze_memory.py`), а не из кортежа.

### Фактический код `model.py` (122 строки)

[ФАКТ] Один класс `PHEF(nn.Module)`, `__init__` строка 8, `forward` строка 45.
[ФАКТ] `key_encoder = nn.Sequential(nn.Linear(key_dim, dim), nn.GELU(), nn.Linear(dim, dim))`, аналогично `query_encoder`.
[ФАКТ] Параметры: `recency_scale`, `score_scale`, `memory_gate`, плюс `value_emb = nn.Embedding(n_values, dim)`, `norm = nn.LayerNorm(dim)`, `head = nn.Linear(dim, n_values)`.
[ФАКТ] `event_keys` встречается трижды: параметр `forward` (47), `event_keys = event_keys[permutation]` внутри `if shuffle:` (74), `memory_keys = self.key_encoder(event_keys)` (78).
[ФАКТ] Слов `codebook` и `slots` в `model.py` нет.
[ФАКТ] `permutation = torch.randperm(batch_size)` — перестановка по **батчевой** оси, применяется к `event_keys`, `values`, `timestamps`.
[ФАКТ] Далее: `semantic = torch.einsum("bd,bsd->bs", query_state, memory_keys) / math.sqrt(self.dim)`; `recency = timestamps.float() / max(1, self.n_segments - 1)`; `scores = self.score_scale * semantic + self.recency_scale * recency`; `memory_values = self.value_emb.weight.to(query_state.dtype)[values]`; в soft-ветке `weights = torch.softmax(scores, dim=1)` и `memory_state = torch.einsum("bs,bsd->bd", weights, memory_values)`.

### Микротест эквивалентности (CPU)

[ФАКТ] Скрипт `/tmp/eq_test.py`, `taskset -c 0`, `bs=4, n_keys=8, key_dim=32, s=64, dim=128, n_values=16`, `torch.manual_seed(0)`. Лог `logs/microtest/eq_test.log` (untracked, не коммитился):
```text
enc gather   max_abs 4.470348358154297e-08
semantic     max_abs 2.2351741790771484e-08
semantic     rel     2.3921629122014565e-07
normalize    max_abs 7.450580596923828e-08
raw_sim      max_abs 1.1920928955078125e-07
```

[ВЫВОД] Все отклонения на уровне машинного эпсилон float32 (ε ≈ 1.19e-07). Максимальное — ровно один эпсилон. Тождества подтверждены:
\[ \text{key\_encoder}(\text{gather}(C, s)) = \text{gather}(\text{key\_encoder}(C), s) \]
\[ \text{normalize}(\text{key\_encoder}(\text{gather}(C,s)), \dim=-1) = \text{gather}(\text{normalize}(\text{key\_encoder}(C), \dim=-1), s) \]
\[ \text{einsum}("bd,bsd\to bs", q, \text{gather}(C,s)) = \text{gather}(\text{einsum}("bd,bkd\to bk", q, C), s) \]

### Все шесть потребителей `event_keys`

| Место | Текущее выражение | Замена |
|---|---|---|
| `phef/model.py:78–84` | `key_encoder(event_keys)`, затем `einsum bd,bsd->bs` | `gather(einsum("bd,bkd->bk", q, key_encoder(C)), 1, slots)` |
| `analyze_memory.py:193` | `oracle_bound(event_keys, query_keys, ...)` | то же на `C`, `scale=1.0` |
| `analyze_memory.py:201` | `normalize(key_encoder(event_keys), dim=-1)` | `normalize(key_encoder(C), dim=-1)` + gather на уровне similarity |
| `analyze_memory.py:218` | `raw_events = key_encoder(event_keys)` | `key_encoder(C)` |
| `analyze_memory.py:243` | `einsum("bd,bsd->bs", query_keys, event_keys)` | `gather(einsum("bd,bkd->bk", query_keys, C), 1, slots)` |
| `world_gpu.py:125` | материализация через `gather` | не создавать, вернуть `(slots, codebook)` |

[ВЫВОД] `oracle_bound` внутри работает только с `similarity` формы `bs × s` — ему не нужен тензор `bs × s × d`. Достаточно хелпера, считающего `sim_keys` формы `bs × n_keys` и разворачивающего через `gather`.

[ВЫВОД] `shuffle` совместим: перестановка по батчевой оси, применять к `slots[permutation]` и `codebook[permutation]`.

[ВЫВОД] Экономия по формам при `bs=64`, `s=131072`, `n_keys=512`, `key_dim=32`, `dim=128`:
- `event_keys` 1024 MiB → `slots` 64 MiB (int64) + `codebook` 4 MiB
- `memory_keys` = `key_encoder(event_keys)` 4096 MiB → `key_encoder(codebook)` 16 MiB
- развёрнутой остаётся только `similarity` формы `bs × s` — 32 MiB

[ВЫВОД] Избыточность `event_keys` относительно `codebook` равна `n_segments / n_keys = 131072 / 512 = 256`, не ~500 как говорилось ранее. `n_keys=512` подтверждён фактическим `/proc/PID/cmdline`.

[НЕИЗВЕСТНО] Какая доля 3.42 с приходится на `gather`, а какая на `_sample_slots`, `torch.flip`, `argmax` по 131072, `randint`. Без разбивки конкретное ускорение не обещается. Нужен профиль по операциям.

[НЕИЗВЕСТНО] `memory_values = value_emb.weight[values]` формы `64 × 131072 × 128` — ещё 4096 MiB, gather-трюком не убирается (`values` случайны на каждый сегмент). Но `n_values=64`, `value_emb.weight` формы `64 × 128`, поэтому применим приём: `einsum("bs,bsd->bd", weights, emb[values])` = `einsum("bv,vd->bd", scatter_add(weights по values), emb.weight)`. Более тонкая правка, не проверена.

[НЕИЗВЕСТНО] Влияние эпсилон-расхождения на обучение за 2500 эпох — накопление через градиенты может сдвинуть `sel_acc` в четвёртом знаке. Проверяется сравнительным прогоном.

[НЕИЗВЕСТНО] Фактический выигрыш по времени и памяти на GPU при `s=131072`. **Это главный незакрытый замер.**

---

## 5. Пик памяти `analyze`: рычаг найден

[ФАКТ] `grep` по `max_memory_allocated|reset_peak|memory_reserved` в `analyze_memory.py` — пусто. Скрипт память не измеряет. Замеры делались внешним сэмплером `nvidia-smi` с шагом 2 с.

[ФАКТ] `--eval-batches` имеет `default=20`; прогоны шли с 80.

[ФАКТ] Два прогона по 5 эпох от весов `results/traj131072/seg131072_e2500_ckpt1.pt`, распределения точек сэмплера:

| eval_batch_size | eval_batches | обучение, MiB | пик analyze, MiB | запас | warning | длительность analyze |
|---:|---:|---:|---:|---:|---|---:|
| 64 | 40 | 14637 | **24041** | 523 MiB (2.1%) | воспроизвёлся | ≈106 с |
| 16 | 40 | 14637 | **14641** | 9923 MiB (40%) | отсутствует | ≈270 с |

[ФАКТ] Ранее при `eval-batch-size 64`, `eval-batches 80` пик был 24018 MiB.

[ФАКТ] Текст warning: `[W824 20:03:13.674068983 CUDACachingAllocator.cpp:508] expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 2162688, total: 25252724736).`

[ВЫВОД] **`--eval-batches` на пик не влияет**: 24018 при 80 против 24041 при 40 — разница 0.1%, шум аллокатора. Батчи идут последовательно, память переиспользуется.

[ВЫВОД] **Пик линеен по `--eval-batch-size`**. Снижение с 64 до 16 убрало 9400 MiB. Расчёт подтверждает: при `bs=64` тензоры `raw_events` и `encoded_events` формы `bs × n_segments × dim` дают по 4096 MiB, `event_keys` 1024 MiB — итого 9216 MiB против замеренной добавки 9404 MiB.

[ВЫВОД] Рекомендация для 131072 и выше: `--eval-batch-size 16`. Плата — время `analyze` вырастает в 2.5 раза.

[ФАКТ] Расхождение метрик при малой выборке, все три прогона на одних весах:

| Прогон | eval_batch_size | eval_batches | эпизодов | answer_acc | hard_soft_gap |
|---|---:|---:|---:|---:|---:|
| финал 2500 | 64 | 80 | 5120 | 0.999609 | 0.093555 |
| probe eb40 | 64 | 40 | 2560 | 0.999219 | 0.093359 |
| probe ebs16 | 16 | 40 | 640 | 1.000000 | 0.100000 |

[ВЫВОД] `answer_acc = 1.0` при 640 эпизодах — два промаха не попали в выборку. `hard_soft_gap = 0.1000` ровно — эффект дискретизации, 64/640. Для сохранения 5120 эпизодов при `eval-batch-size 16` нужно `--eval-batches 320`; память не вырастет, время `analyze` [ВЫВОД] порядка 36 минут.

[ФАКТ] `sel_acc` на epoch 5 равен 0.8809 при `eval-batch-size 64` и 0.8984 при 16, на одних весах и сиде.
[ВЫВОД] `quick_acc` использует `eval_batch_size`, поэтому при том же `acc_n` набираются разные эпизоды. Не недетерминизм.

---

## 6. Скорость и детерминизм

[ФАКТ] Замеры с/эпоху на 131072 при 64 × 4:
- хендофф №16, по 5-эпохным интервалам: 17.6
- первый прогон, epoch 250 по mtime лога: 14.87
- перезапуск, epoch 1750 по mtime среза: 14.44
- перезапуск, полные 2500 эпох за 36242 с: **14.50**

[ВЫВОД] Оценка 17.6 из хендоффа №16 завышена на 21%. Для 40000 эпох при 14.50 с/эпоху: ≈580000 с ≈ **6.7 суток**.

[ФАКТ] `sel_acc` на epoch 250 совпал у двух независимых прогонов с `seed 0` и у probe хендоффа №16: 0.8931 / 0.8931 / 0.89307. Детерминизм на уровне четвёртого знака подтверждён.

[ФАКТ] Фаза `analyze` финального прогона: с 10:18:14 до 10:22:23 = 4 мин 9 с при `eval-batch-size 64`, `eval-batches 80`.

---

## 7. Git: шесть коммитов, всё запушено

[ФАКТ] `origin/main` = `720c600`. Коммиты сессии от старого к новому:
```text
6bc2163 agi: move docs to docs/ and docs/handoffs/, add README
8e1150a phef: timing probe 131072, замер скорости 25 эпох
3af1f0f phef: лог остановленного прогона 131072 x4 на 40000 эпох
830e346 phef: траектория 131072 на 2500 эпох - 40000 эпох не нужны
127ff2c phef: замер batch(64) на 131072 - 94.5% времени эпохи
720c600 phef: пик памяти analyze линеен по eval-batch-size, не по eval-batches
```

[ФАКТ] `6bc2163`: 22 файла, 488 insertions, 1220 deletions. 17 переименований распознаны как `rename (100%)`, два `create mode` (`docs/HOLOGRAPHIC_MEMORY_ABLATION.md`, `docs/handoffs/16_PHEF_HANDOFF.md`), два `delete mode` (`analyze_memory.py.pre_v4`, `.pre_v5`).

[ФАКТ] Особенность: до `git add README.md` индекс содержал `R README.md -> docs/HOLOGRAPHIC_MEMORY_ABLATION.md`. После добавления нового README git разложил операцию на `M README.md` + `A docs/HOLOGRAPHIC_MEMORY_ABLATION.md`. Содержимое проверено: `git show :docs/HOLOGRAPHIC_MEMORY_ABLATION.md` начинается с `# Holographic Hysteresis Memory — ablation report` (8836 B), `git show :README.md` — с `# PHEF — predictive hysteresis memory` (8335 B).

[ФАКТ] README правился скриптом `/tmp/fix_readme.py` с проверкой «каждый паттерн встречается ровно один раз», бэкап в `/tmp/README.backup.md`. Четыре замены: блок структуры под `docs/`, уточнение семантики `--ckpt-every`, строка таблицы `131072 траектория`, абзац с замером 14.9 с/эпоху.

[ФАКТ] `git status --short` на конец сессии: единственный untracked — `phef/logs/microtest/`.

[ВЫВОД] Незакрытые мелочи README: строка 19 говорит `хендоффы сессий, 1..16` — после добавления №17 нужно `1..17`. Строка 129 таблицы лестницы помечает `131072 траектория` как «в работе», хотя прогон завершён с `best_acc 0.91260 @ 2250`. Абзац про 14.9 с/эпоху стоит уточнить до 14.50 по полному прогону.

---

## 8. Открытые вопросы

[НЕИЗВЕСТНО] **Главный.** Фактический выигрыш по времени и памяти от правки `(slots, codebook)` на GPU при `n_segments=131072`. Эквивалентность подтверждена на CPU, экономия посчитана по формам, замера нет. Делать скриптом в `/tmp`, без правки файлов проекта.

[НЕИЗВЕСТНО] Разбивка 3.42 с вызова `batch(64)` по операциям: `gather`, `_sample_slots`, `flip`, `argmax`, `randint`.

[НЕИЗВЕСТНО] Почему `mg_m0.0 = 0.2738` при `mg_top1 = 0.9996`. Гипотеза о доминировании recency-члена (`recency_scale 101.28` против `score_scale 5.58`) не проверена.

[НЕИЗВЕСТНО] Что даёт монотонный дрейф `recency_scale` на горизонте 40000 эпох и вредит ли он.

[НЕИЗВЕСТНО] Применим ли приём `scatter_add` к `memory_values` формы `bs × n_segments × dim` (4096 MiB при bs=64).

[НЕИЗВЕСТНО] Почему jump 8192 → 32768 дал `recency_scale=70.48` при `answer_acc=1.0`, тогда как эталон через 16384 дал 99.15. Открыт с прошлых сессий.

[НЕИЗВЕСТНО] Границы бакетов `acc_by_distance` для больших `n_segments` — на 131072 вся масса в bucket 0.

[ВЫВОД] Масштаб 262144: при линейном росте `event_keys` вызов `batch()` даст ≈6.8 с, эпоха ≈27 с, а `event_keys` 2048 MiB. Без правки `(slots, codebook)` это ещё вдвое медленнее. С правкой обучение перестаёт быть memory-bound по генерации данных. `analyze` безопасен при `--eval-batch-size 16`.

---

## 9. Порядок действий нового агента

1. Прочитать этот handoff целиком, включая §0.
2. Запросить `git status --short`, `git log --oneline -3`, `nvidia-smi`.
3. Начать с GPU-замера двух путей вычисления `semantic` при `n_segments=131072` — скриптом в `/tmp`, время и `torch.cuda.max_memory_allocated()` для полного пути через `gather` и дешёвого через `codebook`. Файлы проекта не править.
4. Перед любым запуском заново читать target JSON/`.pt` фактической командой.
5. Ничего не запускать без явного подтверждения пользователя.
6. Полный 131072 на 40000 эпох **не запускать** — замерено, что роста нет.
7. Любой прогон на 131072 и выше — с `--eval-batch-size 16`, компенсируя `--eval-batches` вчетверо.
8. Проверять `acc_every * ckpt_every < epochs`, иначе срезов не будет.
9. Перед коммитом всегда `git diff --check`, `git diff --cached --stat`, точный diff, явное исключение untracked-чужого.
10. В любую команду запуска включать `mkdir -p` для каталогов `--out` и лога в той же команде.
11. Мелкий долг: README строка 19 (`1..16` → `1..17`), строка 129 (снять «в работе», поставить `best_acc 0.91260`), уточнить 14.9 → 14.50 с/эпоху.
