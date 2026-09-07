# PHEF — самодостаточный handoff: Phase-B replay boundary, published artifacts, next experiment

**Дата передачи:** 2026-09-05, 12:34 MSK.  
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
2. Не переносить без отдельного измерения между разными checkpoint, epoch, seed, `S`, `n_keys`, `rw`, lr, batch size, eval batch size, числа episodes или другого protocol:
   - качество;
   - capacity;
   - устойчивость по seed;
   - peak VRAM;
   - throughput/время;
   - поведение loss;
   - надёжность controls.
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

Перед переходом в новый чат handoff должен содержать: эти правила, свежий Git/branch/remote status, подтверждённые результаты с protocol, открытые вопросы, опровергнутые гипотезы и первый read-only шаг. Новый большой эксперимент не начинать до этой сверки.

---

## 1. Первый шаг нового чата

**[ФАКТ]** Последняя подтверждённая синхронизация была:

```text
HEAD        = 5304ea9
origin/main = 5304ea9
main...origin/main
```

**[НЕИЗВЕСТНО]** Это состояние могло измениться после 2026-09-05 12:30 MSK. Новый чат обязан начать с read-only проверки ниже и не предлагать новый experiment/Git write до её вывода.

```bash
cd ~/qubitrino-projects/agi; \
printf '%s\n' '===== git ====='; \
git status -sb; \
git branch -vv; \
git remote get-url origin; \
git log -4 --oneline --decorate; \
printf '%s\n' '===== PHEF replay artifacts ====='; \
ls -lh \
  phef/results/phaseB_replay_s524288_rw0125_lr1e4_seed7/eval_keys131072_epoch000750_b8_eval16384.json \
  phef/results/phaseB_replay_s524288_rw0125_lr1e4_seed7/eval_keys131072_epoch001500_b8_eval16384.json \
  phef/results/phaseB_replay_s524288_rw0125_lr1e4_seed7/eval_keys131072_epoch002500_b8_eval16384.json \
  2>&1; \
printf '%s\n' '===== active CUDA jobs ====='; \
pgrep -af 'train_recall.py|eval_recall_checkpoint.py' || true; \
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

**Признак успеха:** Git status показывает текущее branch/upstream state; `git log` показывает фактический HEAD; все три replay eval JSON либо подтверждённо существуют, либо их отсутствие становится явным; active CUDA jobs/GPU state показаны без догадок.

---

## 2. Подтверждённое Git-состояние

**[ФАКТ]** На 2026-09-05 12:30 MSK local `main` и `origin/main` совпадали на:

```text
5304ea9 (HEAD -> main, origin/main)
PHEF: add handoffs and experiment result metadata
```

**[ФАКТ]** Предыдущий published commit:

```text
0a822af
PHEF: record Phase-B replay 131072-key boundary evals
```

**[ФАКТ]** Push выполнился в:

```text
git@gitlab.qubitrinotech.ru:root/agi.git
```

и remote `main` был обновлён:

```text
3de40c7..5304ea9  main -> main
```

### 2.1. Что опубликовано

Commit `0a822af` содержит два full hard-eval JSON replay trajectory:

```text
phef/results/phaseB_replay_s524288_rw0125_lr1e4_seed7/
├── eval_keys131072_epoch000750_b8_eval16384.json
└── eval_keys131072_epoch001500_b8_eval16384.json
```

Commit `5304ea9` содержит:

- канонические handoff-документы №18–26;
- 41 small JSON result metadata artifacts;
- включая replay full-eval JSON epoch 2500:

```text
phef/results/phaseB_replay_s524288_rw0125_lr1e4_seed7/
eval_keys131072_epoch002500_b8_eval16384.json
```

### 2.2. Что намеренно не публиковалось

**[ФАКТ]** На последней проверке следующие категории оставались untracked local artifacts и не вошли в commits/push:

- `phef/results/**/*.pt` — model weights/checkpoints;
- `phef/logs/**` — runtime logs;
- `docs/handoffs/18_PHEF_HANDOFF_draft.md` — draft;
- `docs/handoffs/2026-08-23_19-24_16_PHEF_HANDOFF.md` — dated duplicate handoff №16;
- `phef/experiments/train_recall.py.bak` — backup;
- `main` — empty file.

**[ВЫВОД]** Не выполнять broad staging, cleanup или delete для этих файлов без отдельной инвентаризации, content/size review и явного подтверждения.

---

## 3. PHEF protocol и окружение

### 3.1. Пути и interpreter

```text
Git root:      ~/qubitrino-projects/agi
PHEF root:     ~/qubitrino-projects/agi/phef
Python:        /home/linux/qubitrino-projects/agi/venv/bin/python
Branch:        main
Remote:        git@gitlab.qubitrinotech.ru:root/agi.git
GPU:           RTX 4090, ранее наблюдалось около 24 GiB VRAM
```

Для запуска PHEF Python:

```bash
env LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python
```

Для CUDA allocation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 3.2. Fixed evaluation protocol

Для replay boundary measurement использовалось:

```text
n_keys          = 131072
n_values        = 64
n_segments      = 524288
overwrite_p     = 0.3
key_dim         = 32
dim             = 128
eval_batch_size = 8
eval_batches    = 2048
seed            = 7
episodes        = 8 × 2048 = 16384 per condition
hard_eval       = true
```

Для full independent PASS применяются правила из §0.5.

### 3.3. Shell observations

**[ФАКТ]** В прежней работе наблюдалось:

- `git diff --cached --stat` может завершаться с ненулевым exit code при пустом index; не строить на этом `&&`-цепочку, если пустой index ожидаем.
- `tail -f` показывает только строки, которые пишет приложение; текущий evaluator печатает start и final metrics, но не batch progress.
- Для длинного eval жизненность подтверждать process/GPU/log/JSON facts, а не отсутствием промежуточного output.
- Если ожидается отсутствие output file, не использовать длинную `&&` цепочку, которая оборвётся на `ls` с expected absence.

---

## 4. Главный новый результат: replay epoch boundary

### 4.1. Replay trajectory

**[ФАКТ]** Исследуемая trajectory и checkpoint directory:

```text
results/phaseB_replay_s524288_rw0125_lr1e4_seed7/
```

Доступные saved epoch snapshots, проверенные в этой сессии:

```text
epoch_000750.pt
epoch_001500.pt
epoch_002500.pt
```

**[ФАКТ]** Между epoch 750 и epoch 1500 checkpoint files отсутствовали. Поэтому binary search внутри этого интервала по historical snapshots невозможен.

### 4.2. Full hard-eval results at K=131072

| Checkpoint epoch | Real answer | Real retrieval | Shuffled answer | Shuffled retrieval | Masked answer | Episodes | Hard eval | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 750 | 0.91668701171875 | 0.91558837890625 | 0.01611328125 | 0.0 | 0.01544189453125 | 16384 | true | PASS |
| 1500 | 0.868896484375 | 0.86737060546875 | 0.01580810546875 | 0.0 | 0.0164794921875 | 16384 | true | FAIL |
| 2500 | historical replay JSON was committed; exact values must be re-read from its target JSON before use | historical replay JSON was committed; exact values must be re-read from its target JSON before use | — | — | — | — | — | FAIL was reported in this session |

**[ФАКТ]** Epoch 750 target JSON states:

```text
checkpoint_epoch = 750
checkpoint_sel_acc = 1.0
hard_eval = true
answer_chance = 0.015625
real.answer_acc = 0.91668701171875
real.retrieval_acc = 0.91558837890625
real.episodes = 16384
shuffled.answer_acc = 0.01611328125
shuffled.retrieval_acc = 0.0
masked.answer_acc = 0.01544189453125
```

**[ФАКТ]** Epoch 1500 target JSON states:

```text
checkpoint_epoch = 1500
checkpoint_sel_acc = 1.0
hard_eval = true
answer_chance = 0.015625
real.answer_acc = 0.868896484375
real.retrieval_acc = 0.86737060546875
real.episodes = 16384
shuffled.answer_acc = 0.01580810546875
shuffled.retrieval_acc = 0.0
masked.answer_acc = 0.0164794921875
```

**[ВЫВОД]** При fixed hard protocol и `K=131072` replay trajectory проходит operational threshold на epoch 750 и не проходит к epoch 1500:

\[
750 < e_{\mathrm{cross}} \le 1500.
\]

Это **checkpoint-limited boundary**, а не утверждение о точной epoch crossing point.

**[ВЫВОД]** От epoch 750 к epoch 1500:

\[
\Delta \mathrm{answer} = 0.868896484375 - 0.91668701171875 = -0.04779052734375,
\]

\[
\Delta \mathrm{retrieval} = 0.86737060546875 - 0.91558837890625 = -0.0482177734375.
\]

**[ФАКТ]** Controls в обоих eval остались около chance `0.015625`, shuffled retrieval равен `0.0`; control criteria passed.

### 4.3. Что этот результат не доказывает

- **[НЕИЗВЕСТНО]** Точная epoch, на которой crossing occurred внутри `(750, 1500]`.
- **[НЕИЗВЕСТНО]** Причина degradation: длительность Phase B, `S=524288`, low `rw=0.125`, lr `1e-4`, optimizer trajectory, interaction factors или другое.
- **[НЕИЗВЕСТНО]** Устойчивость boundary по training seeds.
- **[НЕИЗВЕСТНО]** Является ли replay trajectory bitwise/trajectory-identical historical original Phase-B run; одинаковое epoch number не заменяет independent evaluation.
- **[ВЫВОД]** Новый training rerun с частыми snapshots мог бы локализовать boundary новой trajectory, но не может честно реконструировать historical state между отсутствующими checkpoints без полного exact resumption state.

---

## 5. Предыдущий исследовательский контекст

### 5.1. Early / Phase-A KD32 representation

**[ФАКТ]** Early checkpoint:

```text
results/early_s65536_seed7_rw10_e6000_best.pt
```

Подтверждённая config из training metadata:

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
best checkpoint epoch = 4500
best sel_acc = 1.0
```

**[ФАКТ]** Early KD32 full independent hard eval passed at least up to `n_keys=4194304` (`2^22`) with 16384 episodes:

| n_keys | S | Batch × batches | Real answer | Real retrieval | Shuffled answer | Masked answer | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 524288 | 65536 | 16 × 1024 | 0.979187 | 0.979004 | 0.015991 | 0.016357 | PASS |
| 1048576 | 65536 | 8 × 2048 | 0.977966 | 0.977905 | 0.015076 | 0.014832 | PASS |
| 4194304 | 65536 | 2 × 8192 | 0.978455 | 0.978271 | 0.014038 | 0.016296 | PASS |

**[ВЫВОД]** Early KD32 representation has confirmed clean key-capacity lower bound of `2^22`; its upper capacity limit is not established.

### 5.2. Historical Phase-B best checkpoint

**[ФАКТ]** Historical Phase-B checkpoint:

```text
results/phaseB_from_early_at_s524288_rw0125_e6000_best.pt
```

Training metadata config:

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
best checkpoint epoch = 750
best sel_acc = 1.0
```

Historical full independent eval results:

| n_keys | S | Batch × batches | Real answer | Real retrieval | Shuffled answer | Masked answer | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 65536 | 524288 | 8 × 2048 | 0.941345 | 0.940613 | 0.017212 | 0.015137 | PASS |
| 131072 | 524288 | 8 × 2048 | 0.910400 | 0.909302 | 0.016357 | 0.015442 | PASS |
| 262144 | 524288 | 8 × 2048 | 0.878418 | 0.876099 | 0.016541 | 0.015625 | FAIL |
| 1048576 | 524288 | 8 × 2048 | 0.847961 | 0.845520 | 0.015320 | 0.015320 | FAIL |

**[ВЫВОД]** For the historical Phase-B best checkpoint, the operational key-capacity boundary lies between 131072 and 262144 under this protocol.

### 5.3. Established and not established

**[ФАКТ]** Established observations:

1. Early KD32 has full independent clean PASS through at least `2^22` keys.
2. Historical Phase-B best checkpoint retains PASS at `2^17=131072` and FAILs at `2^18=262144`.
3. Replay trajectory at `K=131072` PASSes at epoch 750 and FAILs at epoch 1500.
4. Existing replay snapshots do not permit a tighter historical epoch boundary than `(750, 1500]`.
5. For `S=262144`, `rw=0.25`, three seeds were reported PASS in prior committed results.
6. A prior claim that `S=262144` is intrinsically untrainable was contradicted by those `rw=0.25` results.
7. A prior claim that KD32 capacity is only tens of thousands of keys was contradicted by full PASS through 4M keys.

**[НЕИЗВЕСТНО]** Open scientific questions:

1. Which Phase-B factor or interaction causes loss of early key-capacity: duration, `S`, retrieval weight, learning rate, or another mechanism?
2. Does shorter Phase B preserve more capacity at fixed `S=524288`, `rw=0.125`, and `lr=1e-4`?
3. Is the replay boundary stable across training seeds?
4. Can the Early capacity lower bound be extended beyond `2^22` with a feasible full protocol?
5. Why did the KD64 run underperform KD32 in the measured probes?

---

## 6. Recommended next direction

### 6.1. Do not continue automatically

**[ВЫВОД]** No further eval is required to answer the completed replay-boundary question. The existing evidence is sufficient for:

\[
750 < e_{\mathrm{cross}} \le 1500
\]

at `K=131072` for the measured replay trajectory.

Running another eval of the same snapshots at the same protocol does not refine this boundary. There are no saved checkpoints inside the interval.

### 6.2. Best next experiment if tighter temporal localization is needed

**[ГИПОТЕЗА]** If degradation accumulates over Phase-B updates, a dedicated rerun that saves frequent snapshots could reveal where a new trajectory crosses the `K=131072` full-eval threshold.

**Design — one new experiment, not a reconstruction of history:**

```text
Start:      same verified Early checkpoint
Phase B:    S=524288, rw=0.125, lr=1e-4
Seed:       7 unless a separate multi-seed design is approved
Snapshots:  every 125 or 250 epochs, plus best/final
Evaluation: full independent K=131072 protocol, 16384 episodes
Objective:  localize boundary on the newly generated trajectory
```

**[НЕИЗВЕСТНО]** Do not launch this automatically. Before implementation, inspect the current training script and checkpoint format to establish whether snapshots include optimizer/RNG/data-order state and exactly how snapshot cadence is configured.

### 6.3. Best causal experiment if explanation is needed

**[ГИПОТЕЗА]** Shorter Phase-B exposure may preserve more Early key-capacity while retaining some long-S adaptation.

**Minimal one-variable design:**

```text
Base:         same Early checkpoint
Fixed:        S=524288, rw=0.125, lr=1e-4, seed=7
Variable:     Phase-B duration only
Checkpoints:  explicitly save selected durations/snapshots
Evaluation:   full independent K=131072 and K=262144
Decision:     compare both answer and retrieval plus all controls
```

This is a causal experiment only if no other parameter changes. It requires a new written protocol, output paths, GPU preflight and explicit confirmation before launch.

### 6.4. Alternative priorities

1. **Reproducibility:** repeat Early/Phase-B trajectory with independent training seed(s).
2. **Capacity envelope:** evaluate a selected checkpoint at new `n_keys`, but recognize this answers a capacity curve question rather than epoch boundary.
3. **Evaluator engineering:** add progress, checkpointed partial aggregates and resume support before very long 4M+ evals. This requires read-only code inspection, exact diff, tests and separate confirmation.
4. **Repository hygiene:** later decide whether `.pt` checkpoints should use Git LFS, external storage with manifests/hashes, or remain local. Do not bulk-add them.

---

## 7. Read-only command templates

### 7.1. Git inventory

```bash
cd ~/qubitrino-projects/agi; \
git status --short; \
git diff --check; \
git diff --cached --check; \
git log --oneline -8; \
git branch -vv; \
git remote -v
```

### 7.2. Full target JSON inspection

```bash
cd ~/qubitrino-projects/agi/phef; \
/home/linux/qubitrino-projects/agi/venv/bin/python -c \
'import json; p="results/NAME.json"; print(json.dumps(json.load(open(p)), ensure_ascii=False, indent=2, sort_keys=True))'
```

Do not replace `NAME.json` speculatively in a new experiment; first list/read actual target paths.

### 7.3. Available replay snapshots

```bash
cd ~/qubitrino-projects/agi/phef; \
find results/phaseB_replay_s524288_rw0125_lr1e4_seed7 \
  -maxdepth 1 -type f -name 'epoch_*.pt' -printf '%f\n' \
| sort -V
```

### 7.4. CUDA/evaluator preflight

```bash
cd ~/qubitrino-projects/agi/phef; \
printf '%s\n' '===== active evaluator / train / GPU ====='; \
pgrep -af 'train_recall.py|eval_recall_checkpoint.py' || true; \
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; \
printf '%s\n' '===== candidate paths ====='; \
ls -lh results/CHECKPOINT.pt 2>&1; \
for f in results/TAG.json logs/TAG.log; do \
  if [ -e "$f" ]; then printf 'EXISTS %s\n' "$f"; else printf 'ABSENT %s\n' "$f"; fi; \
done
```

This template contains placeholders and is for design/preflight adaptation only. Before any write launch, replace placeholders with a resolved exact checkpoint and output paths, show the final command, and obtain explicit confirmation.

### 7.5. Monitoring an already confirmed long eval

```bash
cd ~/qubitrino-projects/agi; \
date; \
printf '%s\n' '===== evaluator ====='; \
pgrep -af 'eval_recall_checkpoint.py' || true; \
printf '%s\n' '===== GPU ====='; \
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

For a resolved log path, live follow is:

```bash
tail -n 20 -F logs/RESOLVED_TAG.log
```

Stop `tail` with `Ctrl+C`; it does not stop the evaluator.

---

## 8. Final concise state

**[ФАКТ]** Early KD32 (`S=65536`, `rw=1.0`, seed 7) achieved full independent clean PASS through at least `n_keys=4194304`, with 4M result `real answer=0.978455`, `real retrieval=0.978271`, `shuffled answer=0.014038`, `masked answer=0.016296`.

**[ФАКТ]** Historical Phase B (`S=524288`, `rw=0.125`, `lr=1e-4`) had full PASS at `n_keys=131072` (`0.910400/0.909302`) and full FAIL at `n_keys=262144` (`0.878418/0.876099`).

**[ФАКТ]** The new replay boundary measurements at fixed `K=131072` show epoch 750 PASS (`0.916687/0.915588`) and epoch 1500 FAIL (`0.868896/0.867371`) with clean controls and 16384 episodes.

**[ВЫВОД]** For the measured replay trajectory, the operational capacity threshold at `K=131072` is crossed in the checkpoint-limited interval:

\[
750 < e_{\mathrm{cross}} \le 1500.
\]

**[НЕИЗВЕСТНО]** The mechanism of capacity loss and the exact historical epoch inside this interval remain unestablished. The first action in a new chat is the read-only verification block in §1, not another training/evaluation launch.
