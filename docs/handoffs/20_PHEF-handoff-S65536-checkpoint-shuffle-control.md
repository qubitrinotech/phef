# PHEF — хендофф: S=65536, ранняя остановка, checkpoint и shuffle-контроль

Дата: 26 августа 2026.

## §0. Правила работы (обязательны для агента)

1. Каждое утверждение помечать **[ФАКТ]** / **[ВЫВОД]** / **[НЕИЗВЕСТНО]**.
2. Никогда не угадывать содержимое файлов и не выдумывать числа.
3. У агента **нет доступа к машине пользователя**. Все команды выполняет пользователь и присылает вывод.
4. Перед каждым изменением кода — запрос подтверждения.
5. Ответы на русском, команды — готовые для копирования, одной строкой.
6. Если гипотеза опровергнута измерением — прямо это сказать, а не переформулировать.
7. Перед предложением следующего изменения сначала прочитать нужный код/результат; не делать выводы о не показанных фрагментах.

## §1. Окружение

- Репозиторий: `~/qubitrino-projects/agi`.
- Рабочий каталог PHEF: `~/qubitrino-projects/agi/phef`.
- Пакет: `phef/phef/`; скрипты: `phef/experiments/`.
- Python: `/home/linux/qubitrino-projects/agi/venv/bin/python`.
- Префикс запуска: `cd ~/qubitrino-projects/agi/phef && LC_ALL=C PYTHONPATH=.`.
- Для GPU: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Логи: `/tmp/probe_logs`; каталог пропадает после перезагрузки, перед запуском всегда `mkdir -p /tmp/probe_logs`.
- В git пути имеют префикс `phef/`.
- Remote: `gitlab.qubitrinotech.ru:root/agi.git`, ветка `main`.

## §2. Ранее сделано

### Запушенные оптимизации

| Коммит | Содержание |
|---|---|
| `8e18019` | CPU-генерация slots в `world_gpu` |
| `150e9c7` | `batch()` отдаёт `(slots, codebook)` вместо развёрнутых `event_keys` |
| `d440829` | удвоение указателей в `_sample_slots` |

[ФАКТ] После оптимизаций эпоха: 541.32 мс → 38.44 мс, 1.85 → 26.02 эпох/с; parity-метрики побитово идентичны.

### Незакоммиченные правки в `experiments/train_recall.py`

[ФАКТ] До текущего чата уже были применены и не закоммичены две правки:

1. `quick_acc()` вызывает модель с `hard=HARD_EVAL`; тем самым selection-датчик использует ту же геометрию жёсткого чтения, что и `evaluate` при `--hard-eval`.
2. Эпохальная печать использует `n_batches=8` вместо `2`: 128 эпизодов вместо 32.

[ФАКТ] В текущем чате добавлены и не закоммичены правки в `experiments/train_recall.py`:

1. CLI-аргумент:

```python
parser.add_argument(
    "--best-out",
    type=str,
    default=None,
    help="путь к .pt лучшего checkpoint по sel_acc; без него checkpoint не сохраняется",
)
```

2. При новом максимуме `sel_acc` сохраняется:

```python
if args.best_out:
    best_dir = os.path.dirname(args.best_out)
    if best_dir:
        os.makedirs(best_dir, exist_ok=True)
    torch.save(
        {"epoch": best_epoch, "sel_acc": best_acc, "state_dict": best_state},
        args.best_out,
    )
```

3. В JSON report добавлено:

```python
"best_checkpoint": args.best_out,
```

[ФАКТ] Smoke-проверка `--best-out` прошла: checkpoint `.pt` создан, `checkpoint_epoch == report_best_epoch`, `checkpoint_sel_acc == best_acc`, `state_dict_tensors = 16`.

[ФАКТ] Создан новый незакоммиченный скрипт `experiments/eval_recall_checkpoint.py`. Он загружает `.pt` checkpoint, строит `PHEF` и `BranchingWorld`, выполняет hard-eval real/shuffled/masked и сохраняет JSON. Скрипт должен запускаться с `PYTHONPATH=.`.

[ФАКТ] Перед любым коммитом: проверить `git status --short`, `git diff --check`, синтаксис всех изменённых `.py`; не добавлять результаты/черновики без отдельного решения. Перед `git add` удалить возможные временные `apply_*.py`.

## §3. Результаты ёмкости до S=65536

[ФАКТ] Сетка 3000 эпох, bs=32, `--hard-eval`: `n_keys` 32/64/128 × `n_segments` 64/256/1024 — все 9 PASS, 0.9988–0.9998. Плюс `k512_s4096 = 0.9998` и `k512_s16384 = 0.9999`.

[ФАКТ] Жёсткое чтение на checkpoint значительно лучше мягкого: пример `soft 0.814941` против `hard 0.993652`.

## §4. S=65536: известные результаты

### Старые прогоны

[ФАКТ] При lr=1e-3, 12000 эпох, k512/s65536/bs32: после примерно epoch 5500 retrieval-loss взрывался, к epoch 7300 accuracy падала к нулю.

[ФАКТ] При lr=3e-4, 12000 эпох: loss без такого взрыва; исторически эпохальный `answer_acc` достигал 0.7188 на epoch 6300, 0.6875 на 8100, 0.6250 на 8400, затем деградировал.

[ФАКТ] Прежний selection-датчик был мягким и шумовым при S=65536; `--use-best` выбирал случайные ранние эпохи. Это исправлено добавлением `hard=HARD_EVAL` в `quick_acc`.

### Текущие целевые прогоны с исправленным датчиком

[ФАКТ] Параметры целевого прогона: `n_keys=512`, `n_values=64`, `n_segments=65536`, `batch_size=32`, `epochs=12000`, `lr=3e-4`, `--hard-eval`, `eval_batch_size=16`, `eval_batches=64`, `--use-best`, `acc_every=250`, `acc_n=512`, seed 0.

[ФАКТ] Первый целевой прогон (`cap_k512_s65536_best2.json`) выбрал epoch 4750 с `sel_acc=0.509765625`; финальная оценка после восстановления: real answer 0.509766, real retrieval 1.0, shuffled answer 0.054688, masked answer 0.020508.

[ФАКТ] Повторный целевой прогон с checkpoint (`cap_k512_s65536_best3.json`, `.pt`) снова выбрал epoch 4750 с `sel_acc=0.509765625`; финальная оценка: real answer 0.513672, real retrieval 1.0, shuffled answer 0.055664, masked answer 0.021484.

[ФАКТ] Epoch-level `answer_acc` во втором прогоне: 0.5703 на 4700, 0.5859 на 5100, 0.6094 на 5300. Однако selection-world максимум — 0.5098 на 4750; позже его не превзошёл.

[ФАКТ] К epoch 12000 во втором прогоне `sel_acc=0.0781`, что демонстрирует последующую деградацию после лучшего checkpoint.

[ВЫВОД] Исправленный hard selection-датчик работает содержательно и воспроизводимо выбирает окно качества около epoch 4750; это не случайный выбор ранней эпохи.

### Независимая большая оценка checkpoint

[ФАКТ] Выполнена оценка `results/cap_k512_s65536_best3.pt` (epoch 4750) на 16384 эпизодах: `eval_batch_size=16`, `eval_batches=1024`, seed 0. Два запуска дали одинаковый вывод:

```text
real answer      = 0.504333
real retrieval   = 0.999878
shuffled answer  = 0.045593
shuffled retrieval= 0.062012
masked answer    = 0.015869
```

[ФАКТ] `answer_chance = 1/64 = 0.015625`. `masked answer=0.015869` практически совпадает с chance.

[ВЫВОД] У checkpoint есть сильная зависимость от памяти: real answer около 0.5043 против masked около 0.0159; retrieval практически идеален. Но текущий формальный PASS не достигнут: требуется real answer > 0.90 и shuffled/masked < `2*chance = 0.03125`.

[ВЫВОД] На текущих измерениях нельзя заявлять «S=65536 подтверждён как полный PASS». Корректная промежуточная формулировка: «при S=65536 есть воспроизводимое временное окно качества и надёжная адресация, но answer accuracy около 0.50 и формальные контроли PASS не пройдены».

## §5. Установленная причина shuffled-контроля

[ФАКТ] В `phef/model.py` при `shuffle=True` выполняется:

```python
permutation = torch.randperm(batch_size, device=values.device)
slots = slots[permutation]
codebook = codebook[permutation]
values = values[permutation]
timestamps = timestamps[permutation]
```

[ФАКТ] `query_keys` и `answers` при этом не переставляются.

[ФАКТ] В `phef/world.py` `values` генерируются независимо и равномерно в `[0, n_values)`. `answers` — значение последней позиции для query slot. `codebook` генерируется независимо для каждого эпизода.

[ФАКТ] Обычная случайная перестановка `torch.randperm(batch_size)` допускает fixed points: для batch size 16 среднее число fixed points равно 1, то есть примерно 1/16 эпизодов получают собственную память вместо чужой.

[ВЫВОД] Измеренное `shuffled answer=0.045593` объясняется fixed points. Расчёт для bs=16:

\[
(1/16)\cdot0.504333 + (15/16)\cdot(1/64) \approx 0.04617,
\]

что близко к 0.045593.

[ВЫВОД] Это более экономное объяснение, чем утечка через query: masked-контроль около chance дополнительно согласуется с отсутствием query-bypass.

[ФАКТ] `shuffled retrieval=0.062012` нельзя сравнивать с retrieval chance `1/65536`: после shuffle `scores` относятся к памяти другого эпизода, а `answer_positions` остаются от исходного эпизода. Эта retrieval-метрика в shuffled-контроле не имеет семантики вероятности правильной адресации.

## §6. Обязательная проверка в новом чате

[ВЫВОД] Не применять исправление shuffle вслепую. Новый агент должен сначала запросить и получить актуальный вывод следующей read-only проверки, потому что после хендоффа код мог измениться:

```bash
cd ~/qubitrino-projects/agi/phef && git status --short && git diff --check && git diff -- experiments/train_recall.py phef/model.py experiments/eval_recall_checkpoint.py && sed -n '45,130p' phef/model.py && sed -n '1,280p' phef/world.py && sed -n '1,460p' experiments/train_recall.py && sed -n '1,320p' experiments/eval_recall_checkpoint.py
```

[ВЫВОД] После получения кода нужно проверить:

- действительно ли shuffle всё ещё реализован через `torch.randperm(batch_size)`;
- не был ли уже изменён `model.py`;
- что `eval_recall_checkpoint.py` действительно загружает checkpoint и использует hard eval;
- что `train_recall.py` включает незакоммиченные правки hard selection, `n_batches=8`, `--best-out` и `best_checkpoint`;
- что нет иных незакоммиченных или временных изменений, которые меняют интерпретацию эксперимента.

## §7. План дальнейших экспериментов

### Шаг 1: исправить только shuffle-контроль

[ВЫВОД] Если актуальный код подтверждает указанную реализацию, предложить минимальный патч `phef/model.py`: вместо `torch.randperm(batch_size)` использовать ненулевой циклический сдвиг, чтобы fixed points были невозможны.

Предпочтительная логика:

```python
shift = torch.randint(1, batch_size, (), device=values.device)
permutation = torch.roll(
    torch.arange(batch_size, device=values.device),
    shifts=int(shift.item()),
)
```

[НЕИЗВЕСТНО] Нужно отдельно проверить, поддерживается ли `torch.randint(..., device=...)` и `torch.roll` в используемой версии PyTorch; если нет — подобрать эквивалент после чтения версии/проверки.

[ФАКТ] Это изменение кода. До его применения нужно запросить у пользователя подтверждение с точным diff или точной командой.

### Шаг 2: проверить исправленный контроль на уже сохранённом checkpoint

[ВЫВОД] После минимального патча не надо заново обучать модель. Сначала повторить оценку того же файла `results/cap_k512_s65536_best3.pt` на 16384 эпизодах через `experiments/eval_recall_checkpoint.py`.

Цель:

- `real answer` должен остаться около 0.504;
- `real retrieval` должен остаться около 1.0;
- `masked answer` должен остаться около 1/64;
- `shuffled answer` должен упасть к chance и ниже `2/64=0.03125`, если fixed points были единственной причиной превышения.

[ВЫВОД] Для строго воспроизводимого сравнения использовать те же параметры и отдельные новые имена для JSON и лога; не перезаписывать исходный `cap_k512_s65536_best3_eval16384.json`.

### Шаг 3: принять решение по результату проверки

| Наблюдение после no-fixed-point shuffle | Решение |
|---|---|
| shuffled < 0.03125, masked около chance, real около 0.50 | [ВЫВОД] Shuffle-контроль исправлен; checkpoint показывает реальную memory-dependent способность, но общий PASS всё ещё не достигнут из-за real answer < 0.90. Дальше исследовать улучшение answer accuracy, не трогая уже опровергнутые гипотезы. |
| shuffled остаётся > 0.03125 | [ВЫВОД] Fixed points не объясняют эффект полностью. Нужна отдельная диагностика: проверить соответствие shuffle на уровне данных и возможные корреляции; не менять порог PASS. |
| real или masked резко меняются после изменения shuffle | [ВЫВОД] Патч затронул больше, чем контроль; остановиться и сравнить точный diff/логи до следующих обучений. |

### Шаг 4: новые обучения только после результата шага 2

[ВЫВОД] Если no-fixed-point контроль пройден, следующая научная задача — поднять независимую answer accuracy выше текущих ~0.50. Не выбирать архитектурную или schedule-правку до анализа результата шага 2 и чтения актуального кода.

[ВЫВОД] При выборе следующего обучения сохранять best checkpoint через `--best-out`, оценивать его отдельным скриптом на большой выборке и сохранять отдельные JSON/логи.

## §8. Опровергнутые гипотезы (не воскрешать)

1. «Нет тайбрейкера» — верно только на инициализации; `recency_scale` обучается.
2. «Голова не может декодировать» — неверно: в PASS-прогонах `answer_acc == retrieval_acc`.
3. «Коллапс разрешения bf16 при S≈4096–16384» — не случился.
4. «12000 эпох дожмут результат» — неверно: пик пройден раньше, позднее качество деградирует.
5. «Кросс-энтропия неизбежно разгоняет масштаб» — неверно: при lr=3e-4 обвального срыва как при 1e-3 нет.
6. «Эпохальная точность считается на тренировочном батче» — неверно: это `evaluate` на eval world.
7. «Голова не обучается при S=65536» — неверно: имеются ответные значения около 0.50–0.61, но позже деградируют.
8. «Повышенный shuffled answer доказывает утечку через query» — не подтверждено и является менее экономным объяснением, чем fixed points обычной перестановки; сначала нужен no-fixed-point контроль.

## §9. Артефакты текущего исследования

[ФАКТ] В результатах присутствуют как минимум:

- `results/cap_k512_s65536_best2.json`;
- `results/cap_k512_s65536_best3.json`;
- `results/cap_k512_s65536_best3.pt`;
- `results/cap_k512_s65536_best3_eval16384.json`;
- `results/smoke_best_out.pt`;
- `results/smoke_best_out.json`.

[ФАКТ] Логи текущей сессии:

- `/tmp/probe_logs/cap_s65536_best3.log`;
- `/tmp/probe_logs/cap_s65536_best3_eval16384.log`;
- `/tmp/probe_logs/smoke_best_out.log`.

[НЕИЗВЕСТНО] Файлы в `/tmp/probe_logs` переживут текущую сессию только до перезагрузки; не считать их постоянными артефактами.

## §10. Полезные команды

Синтаксис изменённых файлов:

```bash
cd ~/qubitrino-projects/agi/phef && for f in phef/model.py phef/world.py experiments/train_recall.py experiments/eval_recall_checkpoint.py; do /home/linux/qubitrino-projects/agi/venv/bin/python -c "import ast; ast.parse(open('$f').read()); print('ok $f')"; done
```

Статус и diff:

```bash
cd ~/qubitrino-projects/agi && git status --short && git diff --check && git diff -- phef/experiments/train_recall.py phef/experiments/eval_recall_checkpoint.py phef/phef/model.py
```

Проверка checkpoint без обучения после исправления shuffle (заменить имена `out` и лога на новые):

```bash
mkdir -p /tmp/probe_logs && cd ~/qubitrino-projects/agi/phef && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && LC_ALL=C PYTHONPATH=. /home/linux/qubitrino-projects/agi/venv/bin/python -u experiments/eval_recall_checkpoint.py --checkpoint results/cap_k512_s65536_best3.pt --n-keys 512 --n-values 64 --n-segments 65536 --overwrite-p 0.3 --key-dim 32 --dim 128 --eval-batch-size 16 --eval-batches 1024 --seed 0 --out results/REPLACE_ME.json 2>&1 | tee /tmp/probe_logs/REPLACE_ME.log
```
