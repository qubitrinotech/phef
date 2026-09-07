# PHEF — самодостаточный handoff: Memory Flight Recorder, recency budget, механическая диагностика

**Дата передачи:** 2026-09-05, 16:41 MSK.
**Репозиторий:** `~/qubitrino-projects/agi`
**Рабочий каталог PHEF:** `~/qubitrino-projects/agi/phef`
**Ветка:** `main`
**Подтверждённый remote:** `git@gitlab.qubitrinotech.ru:root/agi.git`
**Последний подтверждённый published HEAD:** `5304ea9`
**Последний подтверждённый remote tip:** `origin/main = 5304ea9`

---

## 0. Основные правила работы

Эти правила имеют приоритет над содержательной частью handoff и непроверенными утверждениями из прошлой переписки. Handoff — датированная запись, а не доказательство текущего состояния.

### 0.1. Источник истины

1. Источник истины — только свежий первичный артефакт, присланный пользователем: полный вывод команды, содержимое файла, JSON, лог, checkpoint metadata, Git-вывод или трассировка.
2. Не утверждать по памяти, имени файла, аналогии, старому handoff или собственному предыдущему ответу, что существует файл, checkpoint, JSON, процесс, флаг, конфигурация, commit, remote, GPU job или результат.
3. Ассистент не имеет прямого доступа к локальной машине, shell, Git, GPU, файлам, исходникам и логам. Все команды запускает пользователь. Нельзя писать «я проверил», «я запустил», «GPU свободна» или «run завершился» без вставленного пользователем подтверждения.
4. ТЗ описывает намерение; handoff и журнал — исторический контекст. Любой изменяемый факт перед использованием сверяется командой.
5. Необъяснённое расхождение чисел, конфигураций или артефактов помечается **[НЕИЗВЕСТНО]**. До объяснения на нём не строить следующий эксперимент и не фиксировать его как результат.

### 0.2. Маркировка и порядок ответа

Каждое содержательное утверждение маркировать:

- **[ФАКТ]** — прямо подтверждён первичным артефактом
- **[ВЫВОД]** — логически следует из подтверждённых фактов
- **[ГИПОТЕЗА]** — проверяемое объяснение, не установленный факт
- **[НЕИЗВЕСТНО]** — данных недостаточно; нужен минимальный различающий read-only шаг

Порядок ответа: **сверка фактов → вывод → решение → команда**.

Один ответ — один проверяемый шаг. До команды явно указать:

- что именно она проверяет или меняет;
- какой результат ожидается;
- какой признак успеха должен появиться в выводе.

Если данных не хватает, сначала дать минимальную read-only команду. Не предлагать решение на непроверенной предпосылке. Не использовать формулировки «скорее всего», «обычно», «должно работать» для фактических вопросов о проекте.

### 0.3. Изменения только с явным согласием

Любое write-действие возможно только после явного подтверждения пользователя. К таким действиям относятся:

- запуск train/eval, создающий JSON, log или checkpoint;
- создание, изменение, перемещение или удаление файлов;
- `git add`, `git restore`, `git commit`, `git push`, `git reset`;
- остановка процессов;
- системные изменения.

Перед подтверждением обязательно показать:

1. точную команду или полный diff;
2. все затрагиваемые файлы и output paths;
3. ожидаемые создаваемые артефакты;
4. риск/необратимость действия.

После write-действия запросить read-only проверку фактического результата. Не считать действие успешным только потому, что команда была предложена или запущена.

### 0.4. Git discipline

1. Никогда не использовать `git add .`, `git add -A`, массовый commit или массовое удаление без отдельной инвентаризации и явного согласия.
2. Перед commit показывать exact staged file list, `git diff --cached --stat`, `git diff --cached --check` и содержательный staged diff/summary.
3. Использовать явный whitelist путей.
4. `.pt` weights/checkpoints, `logs/`, drafts, backups и неизвестные файлы не добавлять автоматически.
5. Один commit — одна законченная мысль.
6. `git push` — отдельное подтверждаемое действие после проверки remote URL, branch/upstream, точного диапазона `origin/main..HEAD` и состава commits.
7. После push проверять синхронность `HEAD` и `origin/main` через `git status -sb`, `git rev-parse --short HEAD`, `git rev-parse --short origin/main`.

### 0.5. Экспериментальная дисциплина

1. Конфигурацию конкретного train/eval подтверждать полем `config` соответствующего целевого JSON, а не именем файла, старым handoff или пересказом.
2. Не переносить без отдельного измерения между разными checkpoint, epoch, seed, `S`, `n_keys`, `rw`, lr, batch size, eval batch size, числа episodes или другого protocol: качество; capacity; устойчивость по seed; peak VRAM; throughput/время; поведение loss; надёжность controls.
3. Для full independent PHEF eval operational PASS определяется только так:

```text
hard_eval = true
real.answer_acc >= 0.90
real.retrieval_acc >= 0.90
shuffled.answer_acc < 0.03125
masked.answer_acc < 0.03125
episodes = 16384 в real, shuffled и masked
```

При `n_values=64`:

```text
answer chance = 1 / 64 = 0.015625
control threshold = 2 × chance = 0.03125
```

4. Distance probes на 1024 episodes — screening/profiling. Они не заменяют full independent PASS.
5. Если checkpoint snapshots отсутствуют между двумя measured epochs, нельзя заявлять более точный historical epoch boundary; новый rerun с более частым сохранением — отдельный эксперимент, а не восстановление старой trajectory.
6. **MFR (§5) — механическая диагностика, а не PASS-критерий.** Результаты MFR объясняют механизм, но не заменяют full independent hard-eval и не могут использоваться для заявления PASS/FAIL.

### 0.6. GPU и длинные jobs

1. Перед CUDA job read-only проверить активные train/eval processes, GPU compute processes и отсутствие conflict output paths.
2. На единственной GPU не запускать параллельные CUDA jobs.
3. Для CUDA Python использовать:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

4. Для PHEF использовать единственный подтверждённый interpreter:

```text
/home/linux/qubitrino-projects/agi/venv/bin/python
```

5. Долгие tasks запускать только после подтверждения через `setsid nohup ... < /dev/null & disown`.
6. Не запускать evaluator через pipe с `tail`.
7. Молчащий log сам по себе не означает зависание: evaluator может печатать только start и final lines. Проверять состояние по process, `nvidia-smi`, log и output JSON; не перезапускать job до разбора этих фактов.

### 0.7. Ошибки и переход в новый чат

Если прошлый вывод или правило оказалось ошибочным:

1. остановить текущий шаг;
2. признать ошибку одной короткой строкой;
3. указать, какой факт опровергнут;
4. предложить корректирующую read-only проверку или правку правила;
5. не исправлять последствия молча.

Перед переходом в новый чат handoff должен содержать: эти правила, свежий Git/branch/remote status, подтверждённые результаты с protocol, открытые вопросы, опровергнутые гипотезы и первый read-only шаг.

---

## 1. Первый шаг нового чата

**[ФАКТ]** Последняя подтверждённая синхронизация на 2026-09-05 16:29 MSK:

```text
HEAD        = 5304ea9
origin/main = 5304ea9
main...origin/main
```

**[НЕИЗВЕСТНО]** Это состояние могло измениться после 2026-09-05 16:41 MSK. Новый чат обязан начать с read-only проверки ниже.

```bash
cd ~/qubitrino-projects/agi; \
printf '%s\n' '===== git ====='; \
git status -sb | head -n 8; \
git log -3 --oneline --decorate; \
git rev-parse --short HEAD; \
git rev-parse --short origin/main; \
printf '%s\n' '===== MFR tool ====='; \
ls -lh phef/experiments/probe_memory_flight_recorder.py; \
env LC_ALL=C PYTHONPATH=phef /home/linux/qubitrino-projects/agi/venv/bin/python \
  -c 'print("interpreter ok")'; \
printf '%s\n' '===== MFR artifacts ====='; \
ls -lh phef/results/mfr_*.json 2>&1; \
printf '%s\n' '===== active CUDA jobs ====='; \
pgrep -af 'train_recall.py|eval_recall_checkpoint.py|probe_memory_flight_recorder.py' || true; \
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

**Признак успеха:** Git state показан фактически; MFR-скрипт существует; список MFR JSON либо подтверждён, либо их отсутствие явно; GPU state показан без догадок.

---

## 2. Подтверждённое Git-состояние

**[ФАКТ]** На 2026-09-05 16:29 MSK local `main` и `origin/main` совпадали на:

```text
5304ea9 (HEAD -> main, origin/main)
PHEF: add handoffs and experiment result metadata
```

**[ФАКТ]** Предшествующие published commits:

```text
0a822af  PHEF: record Phase-B replay 131072-key boundary evals
3de40c7  phef: add opt-in epoch snapshot checkpoints
38936d7  phef: add standalone long-S capacity evals
```

### 2.1. Untracked local artifacts

**[ФАКТ]** На последней проверке untracked оставались: `docs/handoffs/18_PHEF_HANDOFF_draft.md`; `docs/handoffs/2026-08-23_19-24_16_PHEF_HANDOFF.md`; `docs/handoffs/2026-09-05_12-34_PHEF_handoff_phaseB_replay_boundary.md`; пустой файл `main`; `phef/experiments/train_recall.py.bak`; весь `phef/logs/**`; весь `phef/results/**/*.pt`.

**[ФАКТ]** Новые untracked артефакты этой сессии:

```text
phef/experiments/probe_memory_flight_recorder.py
phef/results/mfr_smoke_snapshot_epoch000002.json          (v1, устарел)
phef/results/mfr_v2_smoke_snapshot_epoch000002.json       (v2, валидный)
```

**[ВЫВОД]** Не выполнять broad staging, cleanup или delete без отдельной инвентаризации и явного подтверждения.

---

## 3. Механическая модель задачи

Этот раздел — главный содержательный вклад сессии. Он выведен из фактического исходного кода, а не из предположений.

### 3.1. Определение задачи

**[ФАКТ]** `phef/world.py`, класс `BranchingWorld`:

```python
match = slots == query_slots[:, None]
answer_positions = self.n_segments - 1 - np.argmax(match[:, ::-1], axis=1)
answers = values[rows, answer_positions]
```

**[ФАКТ]** Задача: найти **последнее** вхождение запрошенного ключа в эпизоде длиной `n_segments`. `codebook` пересэмплируется каждый батч и L2-нормируется, поэтому lookup-таблица структурно невозможна. `values` случайны и независимы. `timestamps` — просто `arange(n_segments)`.

**[ФАКТ]** `_sample_slots` реализует overwrite-цепочки через pointer doubling; `reuse[:, 0] = False`; замерено ускорение 551.6 → 23.1 мс при bs=8, n_segments=131072.

### 3.2. Формула адресации

**[ФАКТ]** `phef/model.py`, `PHEF.forward`:

```python
query_state = self.query_encoder(query_keys)
memory_keys = self.key_encoder(codebook)
semantic = torch.gather(
    torch.einsum("bd,bkd->bk", query_state, memory_keys) / math.sqrt(self.dim),
    1, slots)
recency = timestamps.float() / max(1, self.n_segments - 1)
scores = self.score_scale * semantic
scores = scores + self.recency_scale * recency
```

То есть:

\[
s_j = \alpha \cdot \frac{\langle q,\; k_{\mathrm{slot}_j}\rangle}{\sqrt{d}} + \beta \cdot \frac{t_j}{S-1},
\]

где \(\alpha =\) `score_scale`, \(\beta =\) `recency_scale`, \(d =\) `dim`, \(S =\) `n_segments`, \(t_j = j\).

**[ФАКТ]** При `hard=True` берётся строгий `argmax`, `last_tie_rate` считается по порогу `1e-6`. Softmax-режим используется только при обучении. Далее `gate = sigmoid(memory_gate)`, `hidden = norm(query_state + gate * memory_state)`, `logits = head(hidden)`.

**[ФАКТ]** `phef/memory.py` содержит класс `EventMemory` с `rerank`-механизмом, но `PHEF.forward` его **не использует**. Это неиспользуемый в текущем пути код.

### 3.3. Центральное следствие

**[ВЫВОД]** Все вхождения одного и того же slot имеют **одинаковый** semantic-score, потому что `semantic` есть `gather` по `slots` из матрицы, индексируемой ключом, а не позицией. Следовательно, различить дубликаты одного ключа может **только** recency-слагаемое.

**[ВЫВОД]** Правильное решение факторизуется на две независимые подзадачи:

1. **семантическое различение** — отделить правильный ключ от остальных `n_keys - 1` ключей;
2. **recency-выбор** — среди `dup_count` вхождений правильного ключа выбрать самое позднее.

**[ВЫВОД]** Ошибки строго делятся на два **несовместимых** класса:

```text
same_key_recency_error   : slots[pred] == query_slot, но pred != answer_position
wrong_key_semantic_error : slots[pred] != query_slot
```

Третьего варианта не существует. Это делает диагноз однозначным.

### 3.4. Условие корректности и recency budget

**[ВЫВОД]** Правильная позиция \(P\) побеждает конкурента \(j\), если

\[
\alpha\,(\mathrm{sem}_P - \mathrm{sem}_j) > \beta \cdot \frac{t_j - t_P}{S-1}.
\]

Для конкурентов с **другим** ключом определяем семантический зазор

\[
\Delta_{\mathrm{sem}} = \min_{j:\ \mathrm{slot}_j \neq \mathrm{query\_slot}} (\mathrm{sem}_P - \mathrm{sem}_j).
\]

Отсюда **recency budget** — изящное решение этой сессии:

\[
B = \frac{\alpha\,\Delta_{\mathrm{sem}}\,(S-1)}{|\beta|}
\quad[\text{timestamp-шаги}],
\qquad
B_{\mathrm{norm}} = \frac{B}{S-1} = \frac{\alpha\,\Delta_{\mathrm{sem}}}{|\beta|}
\quad[\text{безразмерная}].
\]

**Интерпретация:** \(B\) — максимальное число timestamp-шагов, на которое recency способна перебить семантику. \(B_{\mathrm{norm}}\) сравнима между разными \(S\). Если \(B_{\mathrm{norm}} > 1\), recency не может перебить семантику ни на какой дистанции внутри эпизода, и адресация опирается на содержание ключа. Сжатие \(B_{\mathrm{norm}}\) ниже 1 указывает на переход к recency-эвристике, которая не масштабируется с ростом числа ключей.

**[ВЫВОД]** Порог семантической безопасности на всём эпизоде:

\[
\Delta_{\mathrm{sem}} > \frac{|\beta|}{\alpha}.
\]

**[ВЫВОД]** Важная поправка (найдена в smoke v1): \(\Delta_{\mathrm{sem}}\) может быть **отрицательным**. Это означает семантическую инверсию — правильный ключ проигрывает чужому ДО учёта recency, и никакая recency его не спасёт. Такие эпизоды не являются «бюджетом» и должны учитываться отдельно как `semantic_inversion`. Смешивание их с положительными значениями делает средний бюджет бессмысленным.

### 3.5. Конкурирующие требования

**[ВЫВОД]** Модель обязана одновременно:

- держать \(\beta\) достаточно большим, чтобы решать recency-подзадачу (иначе дубликаты не различаются);
- держать \(\alpha\Delta_{\mathrm{sem}}\) достаточно большим, чтобы recency не перебивала семантику на больших дистанциях.

Это **конкурирующие** требования, и их баланс есть измеримая величина \(B_{\mathrm{norm}}\).

---

## 4. Подтверждённые скаляры адресации

**[ФАКТ]** Прямо считано из `state_dict` трёх checkpoints:

| Checkpoint | `score_scale` α | `recency_scale` β | `memory_gate` raw | α/\|β\| | Порог Δsem |
|---|---:|---:|---:|---:|---:|
| `early_s65536_seed7_rw10_e6000_best.pt` (epoch 4500, sel_acc 1.0) | 2.0147361755371094 | 1.3600133657455444 | 0.7847882509231567 | 1.4814 | 0.6750 |
| `smoke_snapshot_epochs/epoch_000002.pt` (epoch 2, sel_acc 0.375) | 0.998076856136322 | 0.001816276926547289 | −0.0012274461332708597 | 549.5180 | 0.0018198 |

**[ФАКТ]** У файлов формата `best_out` (`*_best.pt`) поле `config` содержит `null` для всех ключей: старый формат сохраняет только `epoch`, `sel_acc` и `state_dict`. Config там отсутствует, а не потерян. Snapshot-формат (`epoch_NNNNNN.pt`) содержит полную `config`.

**[ФАКТ]** Config smoke checkpoint: `n_keys=8`, `n_values=8`, `n_segments=16`, `overwrite_p=0.3`, `key_dim=4`, `dim=16`, `seed=123`, `retrieval_weight=1.0`, `lr=0.001`.

**[ВЫВОД]** Early KD32 вырастил \(\beta\) до величины, сопоставимой с \(\alpha\) — отношение 1.48 против 549.5 у необученного checkpoint, разница в 371 раз. Это не деградация, а необходимость: без большого \(\beta\) recency-подзадача не решается.

**[НЕИЗВЕСТНО]** Скаляры \(\alpha\), \(\beta\), gate для replay checkpoints epoch 750, 1500, 2500 ещё **не считаны**. Это первое, что нужно измерить.

---

## 5. Memory Flight Recorder (MFR)

### 5.1. Статус

**[ФАКТ]** Создан и валидирован файл:

```text
phef/experiments/probe_memory_flight_recorder.py
```

`format_version: 2`. Компиляция `COMPILE OK`. Smoke-прогон выполнен успешно на CPU.

**[ФАКТ]** Гарантии: работает в `torch.inference_mode()`; импортирует только `PHEF` и `BranchingWorld`; не вызывает `backward` или `optimizer`; не сохраняет `.pt`; не модифицирует ни один существующий модуль; **не** изменяет `score_scale`/`recency_scale` через `fill_` (в отличие от `probe_gap.py` и `probe_recency.py`), поэтому не может оставить модель в модифицированном состоянии; отказывается перезаписывать существующий `--out`.

**[ВЫВОД]** MFR не может изменить ни один опубликованный результат, включая Early 4M PASS и replay boundary.

### 5.2. Измеряемые величины

| Метрика | Определение | Что различает |
|---|---|---|
| `margin` | \(m = s_P - \max_{j\neq P} s_j\) | Запас устойчивости адреса |
| `same_key_recency_error` | `slots[pred] == query_slot`, `pred != P` | Recency проиграла внутри правильного ключа |
| `wrong_key_semantic_error` | `slots[pred] != query_slot` | Семантика не отделила ключ |
| `delta_sem` | \(\min_j(\mathrm{sem}_P - \mathrm{sem}_j)\) по чужим ключам | Фактический семантический зазор |
| `semantic_inversion_rate` | Доля \(\Delta_{\mathrm{sem}} \le 0\) | Провал семантического различения |
| `semantic_lead_rate` | \(1 - \) inversion rate | Доля эпизодов, где семантика ведёт |
| `recency_budget` | \(B\) в timestamp-шагах, только \(\Delta_{\mathrm{sem}}>0\) | Абсолютный бюджет |
| `recency_budget_norm` | \(B_{\mathrm{norm}} = \alpha\Delta_{\mathrm{sem}}/\lvert\beta\rvert\) | Сравнимая величина; порог 1.0 |
| `dup_count` | Число вхождений `query_slot` | Насколько задача требует recency |
| `tail_distance` | \(S - 1 - P\) | Профиль по дальности ответа |
| `true_rank` | Ранг правильной позиции в scores | Ошибка «почти выбрал» vs collapse |
| `entropy` | \(H(\mathrm{softmax}(s))\) | Диффузность адресации |
| `tie_at_max_rate` | Доля эпизодов с ties на максимуме | Вырожденность argmax |

Все агрегаты потоковые (`Running`, `Quantiles`), per-episode массивы не хранятся. Дополнительно строится `tail_buckets` — логарифмический профиль по `tail_distance`.

### 5.3. Валидация: подтверждённый smoke-результат

**[ФАКТ]** Прогон на `smoke_snapshot_epochs/epoch_000002.pt`, CPU, `--n-keys 8 --n-values 8 --n-segments 16 --key-dim 4 --dim 16 --batch-size 8 --batches 4 --tail-buckets 4 --seed 123`, 32 episodes per condition:

```text
scalars:
  score_scale                      = 0.998076856136322
  recency_scale                     = 0.001816276926547289
  alpha_over_abs_beta               = 549.5179956030432
  delta_sem_threshold_full_episode  = 0.0018197766187849698
  memory_gate_sigmoid               = 0.4996930956840515
  last_tie_rate                     = 0.0

real:
  answer_acc               = 0.25
  retrieval_acc            = 0.15625
  semantic_lead_rate       = 0.15625
  semantic_inversion_rate  = 0.84375
  wrong_key_semantic_rate  = 0.84375
  same_key_recency_rate    = 0.0
  tie_at_max_rate          = 0.0
  delta_sem            n=32  mean=-0.016921580  min=-0.059877910  max=0.017195711
  delta_sem_positive   n=5   mean= 0.008749053  min= 0.002133942  max=0.017195711
  recency_budget_norm  n=5   mean= 4.80776207   min= 1.172639489  max=9.449353218
  margin               n=32  mean=-0.018147191  min=-0.059278410  max=0.000484344

shuffled:
  answer_acc               = 0.25
  retrieval_acc            = 0.09375
  semantic_lead_rate       = 0.0
  semantic_inversion_rate  = 1.0
  wrong_key_semantic_rate  = 0.875
  same_key_recency_rate    = 0.03125
  delta_sem            n=32  mean=-0.017533440  min=-0.045980692  max=0.0
  delta_sem_positive   n=0
  recency_budget_norm  n=0

masked:
  answer_acc               = 0.1875
  retrieval_acc            = None
```

### 5.4. Почему этот результат доказывает корректность инструмента

**[ФАКТ]** Точное численное тождество в `real`:

```text
semantic_lead_rate      = retrieval_acc            = 0.15625
semantic_inversion_rate = wrong_key_semantic_rate  = 0.84375
same_key_recency_error  = 0
```

**[ВЫВОД]** Это не совпадение, а следствие механизма. При \(\alpha/\lvert\beta\rvert = 549.5\) порог равен \(0.00182\); минимальный положительный зазор равен \(0.00213\), то есть уже выше порога. Значит recency физически не может перебить семантику ни в одном эпизоде, и retrieval корректен ровно там, где семантика ведёт. `same_key_recency_error = 0` подтверждает это независимо, а `recency_budget_norm.min = 1.1726 > 1` — количественно.

**[ВЫВОД]** MFR воспроизвёл предсказанное аналитически разложение численно точно. Инструмент измеряет именно тот механизм, для которого создан.

**[ФАКТ]** Контроль `shuffled` дал `semantic_lead_rate = 0.0`, `delta_sem.max = 0.0`, `delta_sem_positive.n = 0`. При перемешивании памяти между элементами батча вхождений правильного ключа в чужом эпизоде нет. Это ожидаемое поведение контроля, а не дефект.

**[ФАКТ]** Энтропия в v1-прогоне: `mean = 2.7725037`, при \(\ln 16 = 2.7725887\). Распределение scores почти равномерное, что соответствует необученному checkpoint.

### 5.5. Исправленный дефект v1 → v2

**[ФАКТ]** В v1 `recency_budget` считался по всем \(\Delta_{\mathrm{sem}}\), включая отрицательные, и дал `mean = -139.48`, `min = -493.56`.

**[ВЫВОД]** Опровергнуто: величина \(B\) применима только при \(\Delta_{\mathrm{sem}} > 0\). Отрицательный «бюджет» не имеет физического смысла.

**[ФАКТ]** Исправление в v2: `semantic_inversion` как отдельный счётчик; `recency_budget` только по положительным зазорам; добавлены `recency_budget_norm`, `delta_sem_positive`, блок `semantic_health`; гистограмма `delta_sem` расширена в отрицательную область; `alpha_over_abs_beta` и `delta_sem_threshold_full_episode` выведены в `scalars`.

---

## 6. Подтверждённые результаты предыдущих сессий

### 6.1. Early KD32 capacity

**[ФАКТ]** Checkpoint `results/early_s65536_seed7_rw10_e6000_best.pt`, epoch 4500, sel_acc 1.0. Training config: `n_segments=65536`, `n_keys=512`, `key_dim=32`, `dim=128`, `batch_size=32`, `epochs=6000`, `lr=3e-4`, `retrieval_weight=1.0`, `aux_value_weight=0.01`, `seed=7`.

**[ФАКТ]** Full independent hard eval, 16384 episodes:

| n_keys | S | Batch × batches | Real answer | Real retrieval | Shuffled answer | Masked answer | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 524288 | 65536 | 16 × 1024 | 0.979187 | 0.979004 | 0.015991 | 0.016357 | PASS |
| 1048576 | 65536 | 8 × 2048 | 0.977966 | 0.977905 | 0.015076 | 0.014832 | PASS |
| 4194304 | 65536 | 2 × 8192 | 0.978455 | 0.978271 | 0.014038 | 0.016296 | PASS |

**[ВЫВОД]** Подтверждённая нижняя граница key-capacity: \(2^{22} = 4194304\). Верхняя граница не установлена. Отсутствие тренда к порогу 0.90 на трёх точках оправдывает проверку \(2^{23}\).

### 6.2. Historical Phase-B best checkpoint

**[ФАКТ]** `results/phaseB_from_early_at_s524288_rw0125_e6000_best.pt`. Config: `init_checkpoint=early_s65536_seed7_rw10_e6000_best.pt`, `n_segments=524288`, `n_keys=512`, `key_dim=32`, `dim=128`, `batch_size=32`, `epochs=6000`, `lr=1e-4`, `retrieval_weight=0.125`, `aux_value_weight=0.01`, `seed=7`, best epoch 750, best sel_acc 1.0.

| n_keys | S | Batch × batches | Real answer | Real retrieval | Shuffled answer | Masked answer | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 65536 | 524288 | 8 × 2048 | 0.941345 | 0.940613 | 0.017212 | 0.015137 | PASS |
| 131072 | 524288 | 8 × 2048 | 0.910400 | 0.909302 | 0.016357 | 0.015442 | PASS |
| 262144 | 524288 | 8 × 2048 | 0.878418 | 0.876099 | 0.016541 | 0.015625 | FAIL |
| 1048576 | 524288 | 8 × 2048 | 0.847961 | 0.845520 | 0.015320 | 0.015320 | FAIL |

**[ВЫВОД]** Operational key-capacity boundary лежит между 131072 и 262144.

### 6.3. Replay epoch boundary

**[ФАКТ]** Trajectory `results/phaseB_replay_s524288_rw0125_lr1e4_seed7/`. Saved snapshots: `epoch_000750.pt`, `epoch_001500.pt`, `epoch_002500.pt`, `best.pt`. Между 750 и 1500 checkpoints отсутствуют.

**[ФАКТ]** Fixed protocol: `n_keys=131072`, `n_values=64`, `n_segments=524288`, `overwrite_p=0.3`, `key_dim=32`, `dim=128`, `eval_batch_size=8`, `eval_batches=2048`, `seed=7`, `hard_eval=true`, 16384 episodes per condition.

| Epoch | sel_acc | Real answer | Real retrieval | Shuffled answer | Masked answer | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 750 | 1.0 | 0.91668701171875 | 0.91558837890625 | 0.01611328125 | 0.01544189453125 | PASS |
| 1500 | 1.0 | 0.868896484375 | 0.86737060546875 | 0.01580810546875 | 0.0164794921875 | FAIL |
| 2500 | 0.990234375 | значения перечитать из target JSON | — | — | — | FAIL был заявлен |

**[ФАКТ]** Все три target JSON существуют:

```text
phef/results/phaseB_replay_s524288_rw0125_lr1e4_seed7/
├── eval_keys131072_epoch000750_b8_eval16384.json   (937 B)
├── eval_keys131072_epoch001500_b8_eval16384.json   (938 B)
└── eval_keys131072_epoch002500_b8_eval16384.json   (944 B)
```

**[ВЫВОД]** Checkpoint-limited boundary:

\[
750 < e_{\mathrm{cross}} \le 1500.
\]

\[
\Delta\mathrm{answer} = -0.04779052734375,
\qquad
\Delta\mathrm{retrieval} = -0.0482177734375.
\]

**[ВЫВОД]** Критично: `sel_acc = 1.0` на обоих snapshots. Внутренний probe **не предсказывает** дальнюю capacity. Это прямая мотивация MFR.

### 6.4. Прочее установленное

**[ФАКТ]** Для `S=262144`, `rw=0.25` три seeds были заявлены PASS в published results. Это опровергло прежнее утверждение о принципиальной необучаемости `S=262144`.

**[ФАКТ]** Прежнее утверждение о capacity в десятки тысяч ключей опровергнуто полным PASS на 4M.

**[НЕИЗВЕСТНО]** Почему KD64 run проиграл KD32 в измеренных probes.

---

## 7. Две взаимоисключающие гипотезы деградации

**[ГИПОТЕЗА A] Recency drift.** Phase B при большом \(S=524288\) и низком `retrieval_weight=0.125` смещает баланс в пользу recency-эвристики. Предсказание MFR: рост `same_key_recency_rate` относительно `wrong_key_semantic_rate`, падение \(B_{\mathrm{norm}}\) ниже 1, рост \(\lvert\beta\rvert/\alpha\), ухудшение `margin` в бакетах с большим `tail_distance`.

**[ГИПОТЕЗА B] Semantic collapse.** Деградирует `key_encoder`/`query_encoder`, семантическая разделимость ключей падает. Предсказание MFR: рост `semantic_inversion_rate` и `wrong_key_semantic_rate`, сжатие `delta_sem_positive`, при этом \(\alpha/\lvert\beta\rvert\) относительно стабильно.

**[ВЫВОД]** Гипотезы дают **противоположные** предсказания по одним и тем же метрикам. Один прогон MFR на epoch 750 против 1500 их различает. Ни одна не подтверждена.

---

## 8. Будущая архитектура: CAMP / Capacity Witnesses

**[ГИПОТЕЗА]** Conservation-Aware Memory Plasticity: Phase B должен адаптироваться к длинному \(S\), не разрушая механизм, обеспечивающий дальнюю адресуемость.

Замысел: сохранять не веса, а функционально критичную характеристику. Anchor-набор синтетических эпизодов (несколько тысяч, детерминированные seeds, не используемые для task loss) даёт эталонные сигнатуры на checkpoint с подтверждённо высокой capacity. Затем Phase B получает дополнительный loss, например:

\[
\mathcal L_{\mathrm{margin}} = \frac1N\sum_i \left[\max\left(0,\; m_i^\star - \delta - m_i\right)\right]^2,
\]

или прямое ограничение на \(B_{\mathrm{norm}}\) снизу. Модель вправе менять параметры, но не вправе незаметно потерять запас разделимости правильного адреса.

**[ВЫВОД]** Цель \(\mathcal L\) должна выбираться **по результату MFR**, а не заранее. Если подтвердится гипотеза A, целью станет \(B_{\mathrm{norm}}\); если B — \(\Delta_{\mathrm{sem}}\).

**[ВЫВОД]** Любой CAMP-эксперимент требует baseline с идентичной конфигурацией. Без парного baseline результат не является причинным.

**[НЕИЗВЕСТНО]** Ничего из CAMP не реализовано. Это замысел, не код.

---

## 9. Открытые вопросы

1. **[НЕИЗВЕСТНО]** Скаляры \(\alpha\), \(\beta\), gate для replay epoch 750, 1500, 2500 — не считаны.
2. **[НЕИЗВЕСТНО]** Какая из гипотез §7 верна.
3. **[НЕИЗВЕСТНО]** Проходит ли Early KD32 hard-eval при \(n_{\mathrm{keys}} = 2^{23} = 8388608\).
4. **[НЕИЗВЕСТНО]** Точная epoch внутри \((750, 1500]\), где произошёл crossing.
5. **[НЕИЗВЕСТНО]** Устойчивость boundary по training seeds.
6. **[НЕИЗВЕСТНО]** Механизм: длительность Phase B, \(S\), `rw`, `lr` или взаимодействие.
7. **[НЕИЗВЕСТНО]** Peak VRAM и время MFR при \(S=524288\) — не измерены.
8. **[НЕИЗВЕСТНО]** Почему KD64 проиграл KD32.

---

## 10. Планируемая последовательность

1. Считать \(\alpha\), \(\beta\), gate для replay 750/1500/2500 — read-only, CPU, мгновенно. **Самый дешёвый различающий шаг.**
2. MFR на replay epoch 750 при \(K=131072\), \(S=524288\) — GPU, требует preflight и подтверждения.
3. MFR на replay epoch 1500 при том же протоколе.
4. Сравнить: различить гипотезы §7.
5. Независимый 8M hard-eval Early KD32 через `eval_recall_checkpoint.py` — без изменений в коде.
6. MFR на Early KD32 при 4M и 8M для \(\Delta_{\mathrm{sem}}\) и \(B_{\mathrm{norm}}\) на границе.
7. Только затем CAMP с явным парным baseline.

**[ВЫВОД]** Шаг 1 обязателен перед шагом 2: он почти бесплатен и может сразу указать направление, сделав тяжёлый GPU-прогон более осмысленным.

---

## 11. Read-only команда для шага 1

```bash
cd ~/qubitrino-projects/agi/phef; \
env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python - <<'PY'
import torch

paths = [
    "results/early_s65536_seed7_rw10_e6000_best.pt",
    "results/phaseB_from_early_at_s524288_rw0125_e6000_best.pt",
    "results/phaseB_replay_s524288_rw0125_lr1e4_seed7/epoch_000750.pt",
    "results/phaseB_replay_s524288_rw0125_lr1e4_seed7/epoch_001500.pt",
    "results/phaseB_replay_s524288_rw0125_lr1e4_seed7/epoch_002500.pt",
    "results/phaseB_replay_s524288_rw0125_lr1e4_seed7/best.pt",
]

print(f"{'checkpoint':62s} {'epoch':>6s} {'sel_acc':>9s} "
      f"{'alpha':>10s} {'beta':>10s} {'a/|b|':>10s} {'thr':>10s} {'gate':>8s}")

for path in paths:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"{path:62s} ERROR {type(exc).__name__}: {exc}")
        continue
    sd = obj["state_dict"]
    a = float(sd["score_scale"])
    b = float(sd["recency_scale"])
    g = float(torch.sigmoid(sd["memory_gate"]))
    ratio = a / abs(b) if b != 0.0 else float("inf")
    thr = abs(b) / a if a != 0.0 else float("inf")
    name = path.split("/")[-2][:28] + "/" + path.split("/")[-1] \
        if "/" in path.replace("results/", "") else path
    print(f"{name[:62]:62s} {obj.get('epoch'):>6} "
          f"{obj.get('sel_acc'):>9.6f} {a:>10.6f} {b:>10.6f} "
          f"{ratio:>10.4f} {thr:>10.6f} {g:>8.6f}")
PY
```

**Что проверяет:** фактические \(\alpha\), \(\beta\), gate, отношение \(\alpha/\lvert\beta\rvert\) и порог \(\lvert\beta\rvert/\alpha\) для всех релевантных checkpoints.

**Признак успеха:** таблица с шестью строками, где для replay 750 и 1500 видны конкретные значения. Ничего не создаётся, GPU не используется.

**Как читать результат:** если \(\lvert\beta\rvert/\alpha\) растёт от epoch 750 к 1500, это поддерживает гипотезу A (recency drift). Если отношение стабильно, а деградация есть, это указывает на гипотезу B (semantic collapse) и требует полного MFR для измерения \(\Delta_{\mathrm{sem}}\).

---

## 12. Финальное краткое состояние

**[ФАКТ]** Early KD32 прошёл full independent hard-eval до \(2^{22}\) ключей включительно: `real answer=0.978455`, `real retrieval=0.978271`, `shuffled=0.014038`, `masked=0.016296`.

**[ФАКТ]** Historical Phase B: PASS при 131072 (`0.910400/0.909302`), FAIL при 262144 (`0.878418/0.876099`).

**[ФАКТ]** Replay boundary при \(K=131072\): epoch 750 PASS (`0.916687/0.915588`), epoch 1500 FAIL (`0.868896/0.867371`), controls чистые, `sel_acc = 1.0` на обоих.

**[ФАКТ]** Механизм адресации выведен из кода: \(s_j = \alpha\,\mathrm{sem}_j + \beta\,t_j/(S-1)\); дубликаты одного ключа различимы только через recency; ошибки делятся на два несовместимых класса.

**[ФАКТ]** Определён и валидирован recency budget \(B_{\mathrm{norm}} = \alpha\Delta_{\mathrm{sem}}/\lvert\beta\rvert\) с порогом 1.0.

**[ФАКТ]** MFR v2 создан, скомпилирован, прошёл smoke-валидацию с точным численным тождеством `semantic_lead_rate = retrieval_acc`.

**[ФАКТ]** Early KD32: \(\alpha/\lvert\beta\rvert = 1.4814\), порог \(\Delta_{\mathrm{sem}} > 0.6750\).

**[НЕИЗВЕСТНО]** Механизм capacity loss не установлен; скаляры replay checkpoints не считаны; 8M не проверен. Первое действие нового чата — read-only блок §11, не запуск train/eval.
