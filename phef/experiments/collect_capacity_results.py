#!/usr/bin/env python3
"""
Сбор и анализ результатов экспериментов по ёмкости памяти.
Ищет JSON-файлы в папках results/capacity_scan*, извлекает метрики,
строит таблицы и графики зависимости точности от n_segments для разных dim.
"""

import json
import glob
import os
import numpy as np
import sys
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib не установлен. Установите: pip install matplotlib")
    plt = None


def extract_metrics_from_json(filepath):
    """Извлекает ключевые метрики из JSON-файла отчёта."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Извлекаем dim, n_segments, seed из config
    config = data.get('config', {})
    dim = config.get('dim', None)
    n_segments = config.get('n_segments', None)
    seed = config.get('seed', None)

    # Если config нет, пробуем извлечь из имени файла
    if dim is None or n_segments is None:
        basename = os.path.basename(filepath)
        # Ожидаем формат: dim64_seg256_s0_e50k.json
        parts = basename.replace('.json', '').split('_')
        for part in parts:
            if part.startswith('dim'):
                dim = int(part.replace('dim', ''))
            elif part.startswith('seg'):
                n_segments = int(part.replace('seg', ''))
            elif part.startswith('s'):
                seed = int(part.replace('s', ''))

    metrics = {
        'dim': dim,
        'n_segments': n_segments,
        'seed': seed,
        'answer_acc': data.get('answer_acc', np.nan),
        'answer_acc_soft': data.get('answer_acc_soft', np.nan),
        'hard_soft_gap': data.get('hard_soft_gap', np.nan),
        'retrieval_acc': data.get('retrieval_acc', np.nan),
        'escape_epoch': data.get('escape_epoch', np.nan),
        'best_epoch': data.get('best_epoch', np.nan),
        'recency_scale': data.get('recency_scale', np.nan),
        'score_scale': data.get('score_scale', np.nan),
        'memory_gate': data.get('memory_gate', np.nan),
        'drop_recent_remote': data.get('drop_recent_remote', np.nan),
        'slot_drop_recent_remote': data.get('slot_drop_recent_remote', np.nan),
        'entropy_norm': data.get('entropy_norm', np.nan),
        'best_sel_acc': data.get('best_sel_acc', np.nan),
    }
    return metrics


def collect_all_results(root_dirs=None):
    """Собирает все JSON-файлы из указанных директорий."""
    if root_dirs is None:
        root_dirs = [
            'results/capacity_scan',
            'results/capacity_scan_dim32',
            'results/capacity_scan_dim128',
            'results/capacity_scan_dim64_extended',
        ]

    all_metrics = []
    for root in root_dirs:
        pattern = os.path.join(root, '*.json')
        for fpath in glob.glob(pattern):
            # Пропускаем файлы, которые могут быть промежуточными или неполными
            if 'test' in fpath or 'smoke' in fpath:
                continue
            try:
                metrics = extract_metrics_from_json(fpath)
                if metrics['dim'] is not None and metrics['n_segments'] is not None:
                    all_metrics.append(metrics)
            except Exception as e:
                print(f"Ошибка при чтении {fpath}: {e}", file=sys.stderr)

    return all_metrics


def group_by_dim_and_seg(metrics_list):
    """Группирует метрики по (dim, n_segments) и вычисляет статистики."""
    groups = defaultdict(list)
    for m in metrics_list:
        key = (m['dim'], m['n_segments'])
        groups[key].append(m)

    results = []
    for (dim, seg), items in sorted(groups.items()):
        answers = [m['answer_acc'] for m in items if not np.isnan(m['answer_acc'])]
        if not answers:
            continue
        result = {
            'dim': dim,
            'n_segments': seg,
            'n_seeds': len(items),
            'answer_mean': np.mean(answers),
            'answer_std': np.std(answers) if len(answers) > 1 else 0.0,
            'answer_min': np.min(answers),
            'answer_max': np.max(answers),
            'escape_mean': np.mean([m['escape_epoch'] for m in items if not np.isnan(m['escape_epoch'])]),
            'best_epoch_mean': np.mean([m['best_epoch'] for m in items if not np.isnan(m['best_epoch'])]),
            'recency_mean': np.mean([m['recency_scale'] for m in items if not np.isnan(m['recency_scale'])]),
            'score_mean': np.mean([m['score_scale'] for m in items if not np.isnan(m['score_scale'])]),
            'drop_mean': np.mean([m['drop_recent_remote'] for m in items if not np.isnan(m['drop_recent_remote'])]),
            'gap_mean': np.mean([m['hard_soft_gap'] for m in items if not np.isnan(m['hard_soft_gap'])]),
        }
        results.append(result)

    return results


def find_capacity(results, threshold=0.99):
    """
    Определяет максимальное n_segments для каждого dim,
    где средняя точность >= threshold.
    """
    capacity = {}
    for dim in sorted(set(r['dim'] for r in results)):
        segs = [r['n_segments'] for r in results if r['dim'] == dim]
        accs = [r['answer_mean'] for r in results if r['dim'] == dim]
        # Сортируем по n_segments
        pairs = sorted(zip(segs, accs))
        max_seg = None
        for seg, acc in pairs:
            if acc >= threshold:
                max_seg = seg
            else:
                break
        if max_seg is not None:
            capacity[dim] = max_seg
        else:
            capacity[dim] = 0
    return capacity


def plot_capacity(results, output_file='capacity_curve.png'):
    """Строит график зависимости точности от n_segments для разных dim."""
    if plt is None:
        print("matplotlib не установлен, пропускаем построение графика.")
        return

    plt.figure(figsize=(12, 8))

    dims = sorted(set(r['dim'] for r in results))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'x']
    colors = plt.cm.tab10(np.linspace(0, 1, len(dims)))

    for i, dim in enumerate(dims):
        data = [r for r in results if r['dim'] == dim]
        segs = [r['n_segments'] for r in data]
        means = [r['answer_mean'] for r in data]
        stds = [r['answer_std'] for r in data]

        plt.errorbar(segs, means, yerr=stds,
                     label=f'dim={dim}',
                     marker=markers[i % len(markers)],
                     color=colors[i],
                     capsize=3,
                     elinewidth=1,
                     linestyle='-')

    plt.axhline(y=0.99, color='red', linestyle='--', label='порог 0.99')
    plt.axhline(y=0.95, color='orange', linestyle='--', label='порог 0.95')
    plt.xscale('log')
    plt.xlabel('n_segments (логарифмическая шкала)')
    plt.ylabel('Точность (answer_acc)')
    plt.title('Зависимость точности от длины эпизода для разных dim')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"График сохранён в {output_file}")
    plt.show()


def print_table(results):
    """Выводит таблицу результатов в консоль."""
    print("\n" + "=" * 100)
    print(f"{'dim':>4} {'n_segments':>10} {'n_seeds':>8} {'answer_mean':>12} {'answer_std':>12} "
          f"{'answer_min':>12} {'answer_max':>12} {'escape':>10} {'best@':>10} "
          f"{'recency':>10} {'score':>10} {'drop':>10}")
    print("-" * 120)

    for r in sorted(results, key=lambda x: (x['dim'], x['n_segments'])):
        print(f"{r['dim']:4d} {r['n_segments']:10d} {r['n_seeds']:8d} "
              f"{r['answer_mean']:12.4f} {r['answer_std']:12.4f} "
              f"{r['answer_min']:12.4f} {r['answer_max']:12.4f} "
              f"{r['escape_mean']:10.0f} {r['best_epoch_mean']:10.0f} "
              f"{r['recency_mean']:10.2f} {r['score_mean']:10.2f} "
              f"{r['drop_mean']:10.4f}")

    print("-" * 120)


def main():
    print("Поиск результатов в папках results/capacity_scan* ...")
    metrics = collect_all_results()
    if not metrics:
        print("Не найдено ни одного JSON-файла с результатами.")
        return

    print(f"Найдено {len(metrics)} прогонов.")
    results = group_by_dim_and_seg(metrics)

    if not results:
        print("Не удалось сгруппировать результаты.")
        return

    print_table(results)

    # Определяем ёмкость
    threshold = 0.99
    capacity = find_capacity(results, threshold=threshold)
    print(f"\nЁмкость (макс. n_segments при точности >= {threshold}):")
    for dim in sorted(capacity.keys()):
        seg = capacity[dim]
        if seg > 0:
            print(f"  dim={dim}: {seg} сегментов")
        else:
            print(f"  dim={dim}: нет данных")

    # Строим график
    plot_capacity(results, output_file='output/capacity_curve.png')

    # Сохраняем таблицу в CSV
    csv_file = 'output/capacity_summary.csv'
    os.makedirs('output', exist_ok=True)
    with open(csv_file, 'w') as f:
        f.write("dim,n_segments,n_seeds,answer_mean,answer_std,answer_min,answer_max,"
                "escape_mean,best_epoch_mean,recency_mean,score_mean,drop_mean\n")
        for r in results:
            f.write(f"{r['dim']},{r['n_segments']},{r['n_seeds']},"
                    f"{r['answer_mean']:.6f},{r['answer_std']:.6f},"
                    f"{r['answer_min']:.6f},{r['answer_max']:.6f},"
                    f"{r['escape_mean']:.0f},{r['best_epoch_mean']:.0f},"
                    f"{r['recency_mean']:.4f},{r['score_mean']:.4f},"
                    f"{r['drop_mean']:.6f}\n")
    print(f"\nТаблица сохранена в {csv_file}")


if __name__ == "__main__":
    main()