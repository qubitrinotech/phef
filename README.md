# PHEF — predictive hysteresis memory

Экспериментальная модель памяти с гистерезисом: замеры ёмкости (`n_segments`),
динамика выхода с плато (`escape_epoch`) и curriculum-маршрут по удвоению
числа сегментов.

Репозиторий: `git@gitlab.qubitrinotech.ru:root/agi.git`

---

## Структура

```text
pyproject.toml                  пакет phef (setuptools, packages.find where=["phef"])
README.md                       этот файл
docs/                           документы, отчёты, хендоффы
  HOLOGRAPHIC_MEMORY_ABLATION.md  отчёт по предыдущей модели (отрицательный результат)
  PHEF формула, ... .md/.pdf      патентный поиск, план экспериментов, коммерциализация
  handoffs/NN_PHEF_HANDOFF.md     хендоффы сессий, 1..16

phef/phef/                      пакет модели
  model.py                      class PHEF (nn.Module)
  world.py                      BranchingWorld (numpy)
  world_gpu.py                  BranchingWorldGPU (побитовой совместимости с numpy нет)
  memory.py

phef/experiments/               скрипты
  analyze_memory.py             основной train/analyze
  probe_gap.py probe_recency.py probe_twostage.py profile_step.py
  collect_capacity_results.py collect_results.py fit_escape.py plot_curve.py
  parity_check.py train_recall.py

phef/results/                   JSON-результаты и веса .pt (версионируются)
phef/logs/                      логи прогонов (версионируются)
phef/output/                    сводные CSV и графики
```

`.gitignore` версионирует `phef/results/**` и `phef/logs/**` принудительно,
включая `.pt`. Исключены только `.ckpt` и `.safetensors`.

---

## Правила запуска

Единственный допустимый интерпретатор:

```text
/home/linux/qubitrino-projects/agi/venv/bin/python
```

Перед запуском обязательно:

```bash
cd ~/qubitrino-projects/agi/phef
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Запуск только через `nohup ... > файл.log 2>&1 &`. `tail` в пайпе с обучением
запрещён — читать уже записанный лог через `grep`/`sed`.

---

## Окружение

Замеры на машине `linux-desktop2`:

| Компонент | Версия |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB |
| Драйвер | 595.84 |
| Python | 3.12.3 |
| torch | 2.13.0+cu130 |
| numpy | 2.5.1 |
| CUDA-стек | nvidia-*-cu13 (cublas 13.1.1.3, cudnn 9.20.0.48, nccl 2.29.7) |

### Восстановление после переустановки системы

1. Установить драйвер NVIDIA.
2. `git clone git@gitlab.qubitrinotech.ru:root/agi.git`
3. `cd agi && python3.12 -m venv venv`
4. `venv/bin/pip install torch==2.13.0 numpy==2.5.1`
5. `venv/bin/pip install -e .`
6. Проверить: `venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

**Неизвестно**, ставился ли torch с `--index-url https://download.pytorch.org/whl/cu130`
или из дефолтного PyPI. `direct_url.json` в `torch-2.13.0.dist-info` отсутствует,
`INSTALLER` = `pip`, `pip.conf` нет — источник индекса не сохранён. Зависимости
`cu13` жёстко прописаны в самом wheel (`Requires-Dist: nvidia-cudnn-cu13==9.20.0.48`
и др.), поэтому установка по имени должна подтянуть их автоматически; если
дефолтный индекс не отдаст сборку `+cu130`, использовать индекс PyTorch.

Веса и результаты в git, поэтому переобучение лестницы после клона не требуется.

---

## Параметры analyze_memory.py

```text
--n-keys --n-values --n-segments --overwrite-p --key-dim --dim
--epochs --batch-size --eval-batch-size --eval-batches --lr
--freeze-recency --recency-value --use-best
--acc-every --acc-n --ckpt-every --accum-steps
--select-metric {soft,hard} --escape-factor --lr-decay
--seed --out --init-weights --world {numpy,gpu}
```

Ключевые:

- `--accum-steps` — gradient accumulation, эффективный batch = `batch_size * accum_steps`, `1` выключено.
- `--ckpt-every` — checkpoint каждые N замеров `acc-every`, `0` выключено. При `acc-every=250` и `ckpt-every=10` checkpoint пишется каждые 2500 эпох; при `ckpt-every=1` — каждые 250 эпох, то есть на каждом замере. Для прогонов короче `acc-every * ckpt-every` эпох срез не запишется ни разу.
- `--select-metric soft` — ридаут пробника для отбора чекпойнта, сопоставим с серией capacity.
- `--escape-factor` — порог выхода с плато в кратности chance (`2.0` => `2/n_values`).

Промежуточные checkpoint пишут `*_last.pt`, `*_best.pt`, `*_history.json`
атомарно. AdamW и RNG не сохраняются — побитового resume нет.

---

## Статус лестницы

| Шаг | Конфигурация | escape_epoch | answer_acc | recency_scale | best_epoch |
|---|---|---:|---:|---:|---:|
| 16384 | batch 256 | — | — | — | — |
| 32768 эталон | init 16384, batch 256 | 250 | 1.000000 | 99.149139 | 34250 |
| 32768 jump | init 8192 напрямую | 250 | 1.000000 | 70.483154 | 36000 |
| 32768 accum-контроль | 128 × 2, eff. 256 | 250 | 1.000000 | 102.347481 | 38000 |
| 131072 probe | 64 × 4, 250 эпох | 250 | 1.000000 | 99.390300 | 250 |
| 131072 траектория | 64 × 4, 2500 эпох, init 32768 | 250 | — | — | в работе |

Эталонные стартовые веса для 131072:
`results/capacity_scan_dim128_curr32768/dim128_seg32768_s0_e40k_curr.pt`
(6 корневых ключей, 16 тензоров, `escape_epoch=250`, `best_epoch=34250`,
`best_acc=0.9912109375`).

Jump 8192 → 32768 даёт ту же top-1 точность при другой геометрии
(`recency_scale` 70.48 против 99.15, `hard_soft_gap` 0.0201 против 0.0109) —
срезать лестницу нельзя.

Probe 131072 занимал 14639 MiB из 24564 MiB при utilization 39–40%.
Память и скорость полного прогона не экстраполируются с probe — только замер.

Замерено 24 августа 2026: 131072 при 64 × 4 идёт ≈ 14.9 с/эпоху
(250 эпох за 3717 с). Отсюда 40000 эпох ≈ 6.9 суток. Фаза `analyze`
у probe доходила до 24018 MiB из 24564 MiB и печатала warning
аллокатора `expandable_segments: memory mapping failed with OOM`.

---

## Правила работы с результатами

- Финальные метрики читать только из корневых полей итогового JSON. При `--use-best` корневые поля и финальный `.pt` описывают выбранные best-веса, а `history[-1]` — последнюю обученную точку; они могут не совпадать.
- Перед использованием каждого `.pt` выполнять `torch.load` и печатать фактические ключи. Перед использованием каждого JSON — `json.load` и фактические ключи.
- Память и скорость — только замер конкретного прогона, не перенос между конфигурациями.
- Расхождения не объяснять рассуждением: помечать как неизвестное и предлагать проверку.
