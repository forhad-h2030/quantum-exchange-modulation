"""Finite-shot figure (paper Fig. 6): test error against state copies per test point.

    python scripts/plot_finite_shots.py

Input:  data/scaled_fixed/benchmark_n{16,400}.npz, results/clean/quantum/n<n>/, results/clean/film/n<n>/
Output: results/clean/fig_shots.pdf, results/clean/finite_shot.json

Models are trained on exact expectations; only inference uses shots. For B copies:
  QFiLM          B/2 joint executions (two copies each)
  FiLM, Gram-6   9-basis tomography, B/9 shots per basis
3 training seeds x 50 shot batches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qem.data import pauli_features  # noqa: E402
from qem.torch_classical import TorchFiLM  # noqa: E402

SPLIT = 'test_joint'
SIZES = [16, 400]
BUDGETS = [18, 54, 180, 540, 1800, 5400, 18000, 54000, 180000]
REPS = 50
SEED = 20261114
OUTCOMES = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])


def quantum_models(n):
    """{seed: (heads, correlation, head weights, head bias)}."""
    out = {}
    for d in [ROOT / f'results/clean/quantum/n{n}', ROOT / f'results/quantum_scaled_fixed/n{n}',
              ROOT / f'results/quantum_scaled_fixed_s31_73/n{n}']:  # the last two: older folder layout
        if not (d / 'metrics.json').exists():
            continue
        preds = np.load(d / 'predictions.npz')
        for run in json.loads((d / 'metrics.json').read_text())['models']['joint_exchange']['runs']:
            s, fs = run['seed'], run['architecture']['fitted_summary']
            if s in out:
                continue
            pre = f'joint_exchange__seed{s}__{SPLIT}__'
            heads, corr = preds[pre + 'heads'], preds[pre + 'correlation']
            w, b = np.asarray(fs['head_weights']), float(fs['head_bias'])
            assert np.max(np.abs(heads @ w + b - preds[pre + 'prediction'])) < 1e-9
            out[s] = (heads, corr, w, b)
    return out


def film_models(n):
    d = ROOT / f'results/clean/film/n{n}'
    if not (d / 'metrics.json').exists():
        d = ROOT / f'results/film_scaled_fixed/n{n}'  # older folder layout
    m = json.loads((d / 'metrics.json').read_text())
    weights, out = np.load(d / 'weights.npz'), {}
    for run in m['selected_runs']:
        f = TorchFiLM(width=run['width'])
        with torch.no_grad():
            f.weights.copy_(torch.tensor(weights[run['key']]))
        out[run['seed']] = f
    return out


def film_predict(model, r15, context):
    with torch.no_grad():
        return model(torch.tensor(r15), torch.tensor(context)).numpy()


def gram_from_r(r16, context):
    """The 6 Gram features of qem.data.gram_features, from a 16-component Pauli vector."""
    r = r16.reshape(-1, 4, 4)[:, 1:, :]
    G = np.einsum('nab,ncb->nac', r, r)
    diag = 0.5 * (1 - context ** 2) * np.diagonal(G, axis1=1, axis2=2)
    cross = np.stack([-context[:, a] * context[:, b] * G[:, a, b] for a, b in [(0, 1), (0, 2), (1, 2)]], axis=1)
    return np.concatenate([diag, cross], axis=1)


def gram_fit(data):
    X = np.column_stack([gram_from_r(pauli_features(data['train_rho']), data['train_context']),
                         np.ones(len(data['train_y']))])
    coef, *_ = np.linalg.lstsq(X, data['train_y'], rcond=None)
    return coef


def gram_predict(coef, r16, context):
    return np.column_stack([gram_from_r(r16, context), np.ones(len(r16))]) @ coef


def quantum_shots(rng, heads, corr, w, b, shots):
    """One batch: shots joint executions per test point."""
    probs = (1 + OUTCOMES[:, 0] * heads[:, :1] + OUTCOMES[:, 1] * heads[:, 1:] + np.outer(corr, OUTCOMES.prod(1))) / 4
    assert probs.min() > -1e-9
    probs = np.clip(probs, 0, None)
    counts = rng.multinomial(shots, probs / probs.sum(1, keepdims=True))
    means = counts @ OUTCOMES / shots
    return means @ w + b


def quantum_exact_variance(heads, corr, w, shots):
    var = (w ** 2 * (1 - heads ** 2)).sum(1) + 2 * w[0] * w[1] * (corr - heads[:, 0] * heads[:, 1])
    return var / shots


def tomography(rng, r16, shots):
    """Pauli vector estimated from `shots` shots in each of the 9 local bases (a, b), a, b in X, Y, Z."""
    m = len(r16)
    R = r16.reshape(m, 4, 4)
    est = np.zeros((m, 4, 4))
    est[:, 0, 0] = 1
    for a in (1, 2, 3):
        for c in (1, 2, 3):
            probs = (1 + OUTCOMES[:, 0] * R[:, a, :1] + OUTCOMES[:, 1] * R[:, :1, c][:, 0][:, None]
                     + OUTCOMES.prod(1) * R[:, a, c][:, None]) / 4
            probs = np.clip(probs, 0, None)
            counts = rng.multinomial(shots, probs / probs.sum(1, keepdims=True))
            s = counts @ OUTCOMES / shots                     # (m, 2) single-qubit means
            corr = counts @ OUTCOMES.prod(1) / shots          # (m,)
            est[:, a, c] = corr
            est[:, a, 0] += s[:, 0] / 3                       # each marginal seen in 3 bases
            est[:, 0, c] += s[:, 1] / 3
    return est.reshape(m, 16)


def stats(values):
    v = np.asarray(values)
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if v.size > 1 else 0.0,
            'p05': float(np.quantile(v, 0.05)), 'p95': float(np.quantile(v, 0.95)),
            'min': float(v.min()), 'max': float(v.max())}


def main():
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(1)
    report = {'split': SPLIT, 'budgets_copies_per_test_point': BUDGETS, 'shot_batches': REPS, 'seed': SEED,
              'training': 'exact expectations; finite-shot inference only',
              'copy_accounting': {'exchange': 'two copies per joint execution',
                                  'tomography': 'one copy per shot, 9 local Pauli bases, equal allocation'},
              'sizes': {}}
    for n in SIZES:
        data = np.load(ROOT / f'data/scaled_fixed/benchmark_n{n}.npz')
        rho, ctx, y = data[f'{SPLIT}_rho'], data[f'{SPLIT}_context'], data[f'{SPLIT}_y']
        r16 = pauli_features(rho)
        q, f, coef = quantum_models(n), film_models(n), gram_fit(data)
        exact = {'exchange': {s: float(np.mean((h @ w + b - y) ** 2)) for s, (h, c, w, b) in q.items()},
                 'film': {s: float(np.mean((film_predict(m, r16[:, 1:], ctx) - y) ** 2)) for s, m in f.items()},
                 'gram6': float(np.mean((gram_predict(coef, r16, ctx) - y) ** 2))}
        rows = []
        for B in BUDGETS:
            row = {'copies': B, 'exchange': {}, 'film': {}}
            for s, (h, c, w, b) in q.items():
                mse = [np.mean((quantum_shots(rng, h, c, w, b, B // 2) - y) ** 2) for _ in range(REPS)]
                row['exchange'][s] = stats(mse)
                row['exchange'][s]['expected'] = float(np.mean((h @ w + b - y) ** 2)
                                                       + np.mean(quantum_exact_variance(h, c, w, B // 2)))
            film_mse = {s: [] for s in f}
            gram_mse = []
            for _ in range(REPS):
                est = tomography(rng, r16, B // 9)
                for s, m in f.items():
                    film_mse[s].append(np.mean((film_predict(m, est[:, 1:], ctx) - y) ** 2))
                gram_mse.append(np.mean((gram_predict(coef, est, ctx) - y) ** 2))
            row['film'] = {s: stats(v) for s, v in film_mse.items()}
            row['gram6'] = stats(gram_mse)
            rows.append(row)
            print(f"n={n} B={B:>6}: exchange {np.mean([row['exchange'][s]['mean'] for s in q]):.2e}  "
                  f"film {np.mean([row['film'][s]['mean'] for s in f]):.3e}  gram6 {row['gram6']['mean']:.2e}",
                  flush=True)
        report['sizes'][n] = {'var_y': float(y.var()), 'exact': exact, 'rows': rows}
    out = ROOT / 'results/clean/finite_shot.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else o) + '\n')
    print('wrote', out)
    plot(report)


def plot(report):
    import matplotlib.pyplot as plt
    QUANTUM, CLASSICAL, GRAM, INK, MUTED, GRID = '#1F5FBF', '#B5470F', '#1A8A6B', '#14213D', '#6B7280', '#E5E7EB'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK,
                         'ytick.color': INK, 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for n, ls in [(16, '-'), (400, '--')]:
        r = report['sizes'][n]
        B = np.array([row['copies'] for row in r['rows']])
        v = r['var_y']
        for key, color, label in [('exchange', QUANTUM, 'exchange layer'), ('film', CLASSICAL, 'classical FiLM')]:
            seeds = r['rows'][0][key].keys()
            mean = np.array([[row[key][s]['mean'] for s in seeds] for row in r['rows']]) / v
            lo = np.array([[row[key][s]['p05'] for s in seeds] for row in r['rows']]).min(1) / v
            hi = np.array([[row[key][s]['p95'] for s in seeds] for row in r['rows']]).max(1) / v
            ax.fill_between(B, lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(B, mean.mean(1), ls, color=color, lw=1.6, marker='o' if n == 16 else None, ms=3.5,
                    label=f'{label}, {n} training states')
        g = np.array([row['gram6']['mean'] for row in r['rows']]) / v
        if n == 16:
            ax.plot(B, g, ':', color=GRAM, lw=1.6, marker='^', ms=3.5, label='fitted Gram-6 on tomography, 16 states')
    ax.axhline(1, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(16, 0.88, 'mean predictor', color=MUTED, fontsize=7.5, ha='left', va='top')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(14, 2.4e5)
    ax.set_xlabel('State copies per test point (exchange: 2 per shot; FiLM, Gram-6: 9-basis tomography)')
    ax.set_ylabel('Relative joint-test MSE, MSE/Var($y$)')
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc='lower left')
    ax.tick_params(colors=INK)
    fig.tight_layout()
    fig.savefig(ROOT / 'results/clean/fig_shots.pdf')
    print('wrote results/clean/fig_shots.pdf')


if __name__ == '__main__':
    main()
