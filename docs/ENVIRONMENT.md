# Окружение PHEF

## Текущая конфигурация (с 2026-09-07)

| Компонент | Версия |
|---|---|
| ОС | Ubuntu 26.04 |
| Python | 3.14.4 (системный, /usr/bin/python3) |
| Менеджер пакетов | uv 0.12.10 |
| torch | 2.14.0+cu130 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| matplotlib | 3.11.1 |
| triton | 3.8.0 |
| GPU | NVIDIA GeForce RTX 4090, 24 ГБ |
| Драйвер / CUDA | 595.84 / 13.2 |

## Предыдущая конфигурация (Ubuntu 24, до 2026-09-06)

Python 3.12.3, torch 2.13.0, numpy 2.5.1, triton 3.7.1.
Чекпойнты, записанные в torch 2.13, читаются в 2.14 без конвертации.

## Восстановление с нуля

```bash
git clone git@gitlab.qubitrinotech.ru:root/agi.git
cd agi
uv venv .venv
source .venv/bin/activate
uv pip install torch numpy scipy matplotlib tqdm
uv pip install -e .
```

Точные версии — в `requirements-frozen.txt`.

## Проверка

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Артефакты вне git

`phef/results/` (168 файлов .pt), `phef/logs/`, `phef/output/` в репозиторий не входят.
Источник восстановления: `agi-full-2026-09-06.tar.gz` на внешнем диске KINGSTON,
внутри путь `qubitrino-projects/agi`.
