"""Bell-family figure: predicted D_{Z x I} along rho_t = t Phi+ + (1 - t) Phi-, exact (2t - 1)^2.

    python scripts/plot_bell.py

Left: models trained on clean data (400 states). Right: trained on drift + readout + offset noise.
No model saw a Bell state in training; the Bell inputs are noiseless.
Input:  data/scaled_fixed/benchmark_n400.npz, results/clean/{quantum,film}/n400/,
        results/werner/{quantum,film}/F0.95sd_clean_r0.025_off0.012/
Output: results/bell/fig_bell.pdf, results/bell/bell_numbers.json (MSE of the seed mean to the exact curve)
Lines: seed means; bands: min to max over seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUANTUM, CLASSICAL, INK, MUTED, GRID = '#1F5FBF', '#B5470F', '#14213D', '#6B7280', '#E5E7EB'
NOISY = 'F0.95sd_clean_r0.025_off0.012'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'font.family': 'sans-serif', 'axes.titlesize': 9.5})


def quantum_bell(dirs):
    """Bell predictions per seed, merged over result folders."""
    out = {}
    for d in dirs:
        if (d / 'metrics.json').exists():
            preds = np.load(d / 'predictions.npz')
            for run in json.loads((d / 'metrics.json').read_text())['models']['joint_exchange']['runs']:
                s = run['seed']
                out.setdefault(s, preds[f'joint_exchange__seed{s}__bell__prediction'])
    return np.array(list(out.values()))


def film_bell(dirs):
    for d in dirs:
        if (d / 'metrics.json').exists():
            preds = np.load(d / 'predictions.npz')
            runs = json.loads((d / 'metrics.json').read_text())['selected_runs']
            return np.array([preds[f"{run['key']}__bell"] for run in runs])
    raise FileNotFoundError(f'no FiLM results in {dirs}')


if __name__ == '__main__':
    R = ROOT / 'results'
    t = np.load(ROOT / 'data/scaled_fixed/benchmark_n400.npz')['bell_t']
    exact = (2 * t - 1) ** 2
    panels = [  # title, QFiLM folders, FiLM folders (older layout as fallback)
        ('Trained on clean data',
         [R / 'clean/quantum/n400', R / 'quantum_scaled_fixed/n400', R / 'quantum_scaled_fixed_s31_73/n400'],
         [R / 'clean/film/n400', R / 'film_scaled_fixed/n400']),
        ('Trained under drift + readout noise',
         [R / f'werner/quantum/{NOISY}'], [R / f'werner/film/{NOISY}'])]
    numbers: dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, (title, qdirs, fdirs) in zip(axes, panels):
        ax.plot(t, exact, color=INK, lw=1.2, ls='--', label=r'exact $(2t-1)^2$', zorder=5)
        for P, color, label in [(quantum_bell(qdirs), QUANTUM, 'QFiLM'), (film_bell(fdirs), CLASSICAL, 'Classical FiLM')]:
            ax.fill_between(t, P.min(0), P.max(0), color=color, alpha=0.18, lw=0, zorder=2)
            ax.plot(t, P.mean(0), color=color, lw=2, label=label, zorder=3)
            numbers.setdefault(title, {})[label] = float(np.mean((P.mean(0) - exact) ** 2))
        ax.set_title(title)
        ax.set_xlabel(r'Bell weight $t$')
        ax.grid(color=GRID, lw=0.6, zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(r'$D_{Z\otimes I}$')
    axes[0].legend(frameon=False, fontsize=8, loc='upper center')
    fig.tight_layout()
    out = R / 'bell'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'fig_bell.pdf')
    (out / 'bell_numbers.json').write_text(json.dumps(numbers, indent=2) + '\n')
    print(json.dumps(numbers, indent=2))
    print('wrote results/bell/fig_bell.pdf, results/bell/bell_numbers.json')
