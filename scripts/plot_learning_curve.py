"""Learning-curve figure (paper Fig. 4) and its numbers.

    python scripts/plot_learning_curve.py

Input:  data/scaled_fixed/benchmark_n<n>.npz, results/clean/quantum/n<n>/, results/clean/film/n<n>/
Output: results/clean/fig_learning_curve.pdf, results/clean/learning_curve_numbers.json
Plotted: test MSE per seed (small marks) and geometric mean (line), clean data, exact expectations.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUANTUM, CLASSICAL, INK, MUTED, GRID = '#1F5FBF', '#B5470F', '#14213D', '#6B7280', '#E5E7EB'
SIZES = [8, 16, 32, 64, 128, 256, 400]
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'font.family': 'sans-serif', 'axes.titlesize': 9.5})


def quantum_mse(n, split):
    """Test MSE per seed. Also reads the older folders quantum_scaled_fixed and quantum_scaled_fixed_s31_73."""
    out = {}
    for d in [ROOT / f'results/clean/quantum/n{n}', ROOT / f'results/quantum_scaled_fixed/n{n}',
              ROOT / f'results/quantum_scaled_fixed_s31_73/n{n}']:
        if (d / 'metrics.json').exists():
            for run in json.loads((d / 'metrics.json').read_text())['models']['joint_exchange']['runs']:
                out.setdefault(run['seed'], run['metrics'][split]['mse'])
    return out


def film_mse(n, split):
    for d in [ROOT / f'results/clean/film/n{n}', ROOT / f'results/film_scaled_fixed/n{n}']:
        if (d / 'metrics.json').exists():
            m = json.loads((d / 'metrics.json').read_text())
            return {run['seed']: run['metrics'][split]['mse'] for run in m['selected_runs']}
    raise FileNotFoundError(f'no FiLM results for n={n}')


if __name__ == '__main__':
    data = np.load(ROOT / 'data/scaled_fixed/benchmark_n400.npz')
    constant = float(np.mean((data['test_joint_y'] - data['train_y'].mean()) ** 2))
    numbers: dict = {'constant_predictor_mse': constant}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, split, title in zip(axes, ['test_state', 'test_joint'],
                                ['New states, training axes', 'New states, held-out axes']):
        numbers[split] = {}
        for key, get, color, marker, label, sizes in [
                ('quantum', quantum_mse, QUANTUM, 'o', 'QFiLM (exchange)', SIZES),
                ('film', film_mse, CLASSICAL, 's', 'Classical FiLM', SIZES + [1000])]:
            x, g = [], []
            for n in sizes:
                v = np.array(list(get(n, split).values()))
                numbers[split].setdefault(key, {})[n] = v.tolist()
                ax.scatter([n] * len(v), v, s=12, color=color, marker=marker, alpha=0.45, edgecolor='none', zorder=3)
                x.append(n)
                g.append(np.exp(np.log(v).mean()))
            ax.plot(x, g, color=color, lw=2, marker=marker, ms=5.5, mec='white', mew=0.8, zorder=4, label=label)
        ax.axhline(constant, color=MUTED, ls='--', lw=1, zorder=2)
        ax.text(9, constant * 0.012, 'training-mean\nconstant (dashed)', color=MUTED, fontsize=7.5)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ticks = [8, 16, 32, 64, 128, 256, 512, 1024]
        ax.set_xticks(ticks)
        ax.set_xticklabels(map(str, ticks))
        ax.set_xlabel('Training states (5 directions each)')
        ax.set_title(title)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Test MSE')
    axes[0].set_ylim(1e-13, 0.4)
    axes[0].legend(frameon=False, loc='center left', fontsize=8)
    fig.tight_layout()
    out = ROOT / 'results/clean'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'fig_learning_curve.pdf')
    (out / 'learning_curve_numbers.json').write_text(json.dumps(numbers, indent=2) + '\n')
    print('wrote results/clean/fig_learning_curve.pdf, results/clean/learning_curve_numbers.json')
