"""Noise-study figure (paper Fig. 5) and its numbers.

    python scripts/plot_noise.py

Input:  data/werner/<scenario>.npz, results/werner/quantum/<scenario>/, results/werner/film/<scenario>/
Output: results/noise/fig_noise.pdf, results/noise/noise_numbers.json
Plotted: test-joint MSE / Var(y), one point per seed; large marker = geometric mean.
Gram-6 is a least-squares fit on the training rows (no training script needed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qem.data import gram_features  # noqa: E402

QUANTUM, CLASSICAL, GRAM, INK, MUTED, GRID = '#1F5FBF', '#B5470F', '#1A8A6B', '#14213D', '#6B7280', '#E5E7EB'
SPLIT = 'test_joint'
SCENARIOS = [  # key, label, group
    ('F0.95_prepared', r'$p=0.067$ ($F_{\rm pure}=0.95$), prepared', 'Fixed preparation noise'),
    ('F0.95_clean', r'$p=0.067$ ($F_{\rm pure}=0.95$), intended', 'Fixed preparation noise'),
    ('F0.50_prepared', r'$p=0.667$ ($F_{\rm pure}=0.50$), prepared', 'Fixed preparation noise'),
    ('F0.50_clean', r'$p=0.667$ ($F_{\rm pure}=0.50$), intended', 'Fixed preparation noise'),
    ('F0.95sd_clean', r'$p\approx0.067$ ($F_{\rm pure}\approx0.95$), drift', 'Drifting preparation'),
    ('F0.50sd_clean', r'$p\approx0.667$ ($F_{\rm pure}\approx0.50$), drift', 'Drifting preparation'),
    ('F0.95_clean_r0.025', r'$p=0.067$ ($F_{\rm pure}=0.95$) + readout', 'With EPR-scale readout noise'),
    ('F0.95sd_clean_r0.025_off0.012', r'$p\approx0.067$ ($F_{\rm pure}\approx0.95$), drift + readout + offset', 'With EPR-scale readout noise'),
]
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'font.family': 'sans-serif'})


def quantum_mse(key):
    """Test MSE per seed; also reads the older split folder quantum_s31_73 if present."""
    out = {}
    for d in [ROOT / f'results/werner/quantum/{key}', ROOT / f'results/werner/quantum_s31_73/{key}']:
        if (d / 'metrics.json').exists():
            for run in json.loads((d / 'metrics.json').read_text())['models']['joint_exchange']['runs']:
                out[run['seed']] = run['metrics'][SPLIT]['mse']
    return out


def film_mse(key):
    m = json.loads((ROOT / f'results/werner/film/{key}/metrics.json').read_text())
    return {run['seed']: run['metrics'][SPLIT]['mse'] for run in m['selected_runs']}


def gram_mse(data):
    def X(split):
        g = gram_features(data[f'{split}_rho'], data[f'{split}_context'])
        return np.column_stack([g, np.ones(len(g))])
    coef, *_ = np.linalg.lstsq(X('train'), data['train_y'], rcond=None)
    return float(np.mean((X(SPLIT) @ coef - data[f'{SPLIT}_y']) ** 2))


if __name__ == '__main__':
    rows, numbers = [], {}
    for key, label, group in SCENARIOS:
        data = np.load(ROOT / f'data/werner/{key}.npz')
        var = float(data[f'{SPLIT}_y'].var())
        q, f, g = quantum_mse(key), film_mse(key), gram_mse(data)
        numbers[key] = {'var_y': var, 'quantum': q, 'film': f, 'gram6': g}
        rows.append((label, group, np.array(list(q.values())) / var, np.array(list(f.values())) / var, g / var))
        gm = lambda v: float(np.exp(np.log(v).mean()))  # noqa: E731
        print(f'{key:<32} QFiLM {gm(rows[-1][2]):.1e}  FiLM {gm(rows[-1][3]):.1e}  Gram-6 {rows[-1][4]:.1e}')

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    y, ypos, starts, prev = 0, [], [], None
    for _, group, *_ in rows:
        if group != prev:
            y += 0.8 if prev is not None else 0
            starts.append((y, group))
            prev = group
        ypos.append(y)
        y += 1
    floor = 1e-13
    ax.axvspan(floor, 1e-6, color='#E4EEFF', alpha=0.6, lw=0, zorder=0)
    ax.axvline(1.0, color=MUTED, ls='--', lw=1, zorder=1)
    ax.text(2.5e-13, ypos[-1] + 0.9, 'numerically exact', color=MUTED, fontsize=7.5, va='center')
    ax.text(1.12, ypos[-1] + 0.9, 'mean predictor', color=MUTED, fontsize=7.5, va='center')
    for yy, (_, _, q, f, g) in zip(ypos, rows):
        for vals, color, marker, dy in [(q, QUANTUM, 'o', -0.2), (f, CLASSICAL, 's', 0.0)]:
            ax.plot([vals.min(), vals.max()], [yy + dy] * 2, color=color, lw=2, alpha=0.35, zorder=2)
            ax.scatter(vals, [yy + dy] * len(vals), s=16, color=color, marker=marker,
                       edgecolor='white', lw=0.6, zorder=3)
            ax.scatter([np.exp(np.log(vals).mean())], [yy + dy], s=55, color=color, marker=marker,
                       edgecolor='white', lw=1.0, zorder=4)
        ax.scatter([max(g, floor * 1.5)], [yy + 0.2], s=50, color=GRAM, marker='^',
                   edgecolor='white', lw=1.0, zorder=4)
        if g < floor:  # below the axis: arrow at the left edge
            ax.annotate('', xy=(floor, yy + 0.2), xytext=(floor * 6, yy + 0.2),
                        arrowprops=dict(arrowstyle='->', color=GRAM, lw=1))
    ax.set_xscale('log')
    ax.set_xlim(floor, 6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(ypos[-1] + 1.3, -1.0)
    for y0, group in starts:
        ax.text(-0.02, y0 - 0.62, group, transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=8, fontweight='bold', color=INK)
    ax.set_xlabel(r'Relative joint-test MSE, $\mathrm{MSE}/\mathrm{Var}(y)$')
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker='o', ls='', color=QUANTUM, ms=7, label='QFiLM (exchange)'),
               Line2D([], [], marker='s', ls='', color=CLASSICAL, ms=7, label='Classical FiLM'),
               Line2D([], [], marker='^', ls='', color=GRAM, ms=7, label='Fitted Gram-6 (physics-informed)')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    out = ROOT / 'results/noise'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'fig_noise.pdf')
    (out / 'noise_numbers.json').write_text(json.dumps(numbers, indent=2) + '\n')
    print('wrote results/noise/fig_noise.pdf, results/noise/noise_numbers.json')
