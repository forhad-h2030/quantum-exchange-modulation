"""Readout bit-flip sensitivity of the finite-shot comparison (Fig. 7).

Each measured bit (QFiLM ports, tomography qubits) flips with probability e; gates are ideal.
A flip scales a one-bit mean by 1-2e and a two-bit correlator by (1-2e)^2.
  raw         estimates as measured
  corrected   divided by the known attenuation
  unbiased    (Gram-6) corrected, with unbiased squared estimates
Also: the analytic circuit with its fitted head (results/baselines/n64 from scripts/baseline_study.py) and the
direction-specific transverse-6 estimator (qem/local_estimators.py; unbiased squares, raw or corrected).
3 seeds, 50 shot batches. Default: realistic noise, 64 training states (scripts/noise_study.py runs);
--clean: the earlier noise-free study at 16 training states.

Output: results/paper_qip/readout_flip[_clean].json, paper/qip/fig_readout.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from paper_qip_finite_shot import (BUDGETS, CLEAN, OUTCOMES, SPLIT, data_path, film_models,  # noqa: E402
                                   film_predict, gram_fit, quantum_models, stats)
from qem import gate_noise as gn  # noqa: E402
from qem import local_estimators as le  # noqa: E402
from qem.data import pauli_features  # noqa: E402

N_TRAIN = 16 if CLEAN else 64
FLIP_RATES = [0.0, 0.005, 0.01, 0.02, 0.03]
REPS = 50
SEED = 20261115


def sample_means(rng, m1, m2, m12, shots):
    """+-1 outcome means of two bits with means m1, m2 and correlator m12, `shots` shots per row."""
    probs = (1 + OUTCOMES[:, 0] * m1[:, None] + OUTCOMES[:, 1] * m2[:, None] + np.outer(m12, OUTCOMES.prod(1))) / 4
    probs = np.clip(probs, 0, None)
    counts = rng.multinomial(shots, probs / probs.sum(1, keepdims=True))
    s = counts @ OUTCOMES / shots
    return s[:, 0], s[:, 1], counts @ OUTCOMES.prod(1) / shots


def check_flip_equivalence(rng, e=0.03, shots=200_000):
    """Explicit bit flips on sampled outcomes reproduce the attenuated means."""
    m1, m2, m12 = np.array([0.6]), np.array([-0.3]), np.array([-0.1])
    probs = (1 + OUTCOMES[:, 0] * m1 + OUTCOMES[:, 1] * m2 + OUTCOMES.prod(1) * m12) / 4
    out = OUTCOMES[rng.choice(4, size=shots, p=probs)]
    out = out * np.where(rng.random(out.shape) < e, -1, 1)
    f = 1 - 2 * e
    got = [out[:, 0].mean(), out[:, 1].mean(), out.prod(1).mean()]
    want = [f * m1[0], f * m2[0], f * f * m12[0]]
    assert np.allclose(got, want, atol=5 / np.sqrt(shots)), (got, want)


def quantum_predict(rng, heads, corr, w, b, shots, e, correct):
    f = 1 - 2 * e
    sA, sB, _ = sample_means(rng, f * heads[:, 0], f * heads[:, 1], f * f * corr, shots)
    means = np.column_stack([sA, sB]) / (f if correct else 1)
    return means @ w + b


def tomography(rng, r16, shots, e):
    """Raw Pauli means from 9-basis tomography, and the number of outcomes behind each."""
    f = 1 - 2 * e
    m = len(r16)
    R = r16.reshape(m, 4, 4)
    est = np.zeros((m, 4, 4))
    est[:, 0, 0] = 1
    for a in (1, 2, 3):
        for c in (1, 2, 3):
            sA, sB, cor = sample_means(rng, f * R[:, a, 0], f * R[:, 0, c], f * f * R[:, a, c], shots)
            est[:, a, c] = cor
            est[:, a, 0] += sA / 3
            est[:, 0, c] += sB / 3
    n_out = np.full((4, 4), shots, dtype=float)
    n_out[:, 0] = n_out[0, :] = 3 * shots
    return est, n_out


def attenuation(e):
    f = 1 - 2 * e
    A = np.full((4, 4), f * f)
    A[:, 0] = A[0, :] = f
    A[0, 0] = 1
    return A


def gram_features_from_G(G, context):
    diag = 0.5 * (1 - context ** 2) * np.diagonal(G, axis1=1, axis2=2)
    cross = np.stack([-context[:, a] * context[:, b] * G[:, a, b] for a, b in [(0, 1), (0, 2), (1, 2)]], axis=1)
    return np.concatenate([diag, cross], axis=1)


def gram_G(raw, n_out, e, mode):
    """Gram matrix G_ac = sum_b r_ab r_cb; only diagonal squares need the unbiased fix."""
    A = attenuation(e)
    r = raw if mode == 'raw' else raw / A
    rr = r[:, 1:, :]
    G = np.einsum('nab,ncb->nac', rr, rr)
    if mode == 'unbiased':
        N, Af = n_out[1:, :], A[1:, :]
        sq = (N * raw[:, 1:, :] ** 2 - 1) / ((N - 1) * Af ** 2)
        idx = np.arange(3)
        G[:, idx, idx] = sq.sum(2)
    return G


def main():
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(1)
    check_flip_equivalence(rng)
    data = np.load(data_path(N_TRAIN))
    rho, ctx, y = data[f'{SPLIT}_rho'], data[f'{SPLIT}_context'], data[f'{SPLIT}_y']
    r16 = pauli_features(rho)
    q, fm, coef = quantum_models(N_TRAIN), film_models(N_TRAIN), gram_fit(data)
    gram = lambda G: np.column_stack([gram_features_from_G(G, ctx), np.ones(len(G))]) @ coef
    # analytic circuit with head fitted on the exact (ideal-gate) training expectations
    an = np.load(ROOT / f'results/baselines/n{N_TRAIN}/analytic_expectations.npz')
    ah, ac = an['0__test_joint__heads'], an['0__test_joint__corr']
    aw, ab = gn.fit_head(an['0__train__heads'], data['train_y'])
    # transverse-6: affine calibration on exact transverse values of the training inputs
    tX = np.column_stack([le.transverse_exact(pauli_features(data['train_rho']), data['train_context']),
                          np.ones(len(data['train_y']))])
    tcoef, *_ = np.linalg.lstsq(tX, data['train_y'], rcond=None)

    rows = []
    for e in FLIP_RATES:
        for B in BUDGETS:
            shots_q, shots_t = B // 2, B // 9
            acc = {k: [] for k in ['exchange_raw', 'exchange_corrected', 'film_raw', 'film_corrected',
                                   'gram6_raw', 'gram6_corrected', 'gram6_corrected_unbiased',
                                   'analytic_raw', 'analytic_corrected', 'transverse6_raw', 'transverse6_corrected']}
            for _ in range(REPS):
                for s, (h, c, w, b) in q.items():
                    acc['exchange_raw'].append(np.mean((quantum_predict(rng, h, c, w, b, shots_q, e, False) - y) ** 2))
                    acc['exchange_corrected'].append(np.mean((quantum_predict(rng, h, c, w, b, shots_q, e, True) - y) ** 2))
                for cor_, key in ((False, 'analytic_raw'), (True, 'analytic_corrected')):
                    acc[key].append(np.mean((quantum_predict(rng, ah, ac, aw, ab, shots_q, e, cor_) - y) ** 2))
                fr = 1 - 2 * e
                for cor_, key in ((False, 'transverse6_raw'), (True, 'transverse6_corrected')):
                    t = le.transverse_estimate(rng, r16, ctx, B // 6, fA=fr, fB=np.array([1, fr, fr, fr]), correct=cor_)
                    acc[key].append(np.mean((tcoef[0] * t + tcoef[1] - y) ** 2))
                raw, n_out = tomography(rng, r16, shots_t, e)
                cor = raw / attenuation(e)
                for s, model in fm.items():
                    acc['film_raw'].append(np.mean((film_predict(model, raw.reshape(-1, 16)[:, 1:], ctx) - y) ** 2))
                    acc['film_corrected'].append(np.mean((film_predict(model, cor.reshape(-1, 16)[:, 1:], ctx) - y) ** 2))
                for mode, key in [('raw', 'gram6_raw'), ('corrected', 'gram6_corrected'),
                                  ('unbiased', 'gram6_corrected_unbiased')]:
                    acc[key].append(np.mean((gram(gram_G(raw, n_out, e, mode)) - y) ** 2))
            row = {'flip_rate': e, 'copies': B, **{k: stats(v) for k, v in acc.items()}}
            rows.append(row)
            print(f"e={e:.3f} B={B:>6}: " + '  '.join(f"{k} {row[k]['mean']:.2e}" for k in acc), flush=True)
    report = {'split': SPLIT, 'training_states': N_TRAIN, 'flip_rates': FLIP_RATES, 'budgets': BUDGETS,
              'shot_batches': REPS, 'seed': SEED, 'var_y': float(y.var()),
              'model': 'symmetric independent bit flip with probability e on every measured bit; ideal gates',
              'notes': 'exchange and FiLM entries pool 3 training seeds x shot batches; Gram-6 is deterministic',
              'rows': rows}
    out = ROOT / f"results/paper_qip/readout_flip{'_clean' if CLEAN else ''}.json"
    out.write_text(json.dumps(report, indent=2) + '\n')
    print('wrote', out)
    plot(report)


def plot(report):
    import matplotlib.pyplot as plt
    from paper_qip_figures import GRAM, INK, MUTED, QUANTUM, style  # noqa: E402
    TRANS = '#7A3E9D'
    v, rows = report['var_y'], report['rows']
    get = lambda e, key: np.array([[r['copies'], r[key]['mean'] / v] for r in rows if r['flip_rate'] == e])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={'width_ratios': [1.35, 1]})

    ax = axes[0]
    e_show = 0.02
    for key, color, label in [('exchange', QUANTUM, 'QFiLM'), ('gram6', GRAM, 'Gram-6'),
                              ('transverse6', TRANS, 'transverse-6')]:
        corrected = 'gram6_corrected_unbiased' if key == 'gram6' else f'{key}_corrected'
        raw = get(e_show, f'{key}_raw')
        cor = get(e_show, corrected)
        tag = 'corrected'
        ax.plot(cor[:, 0], cor[:, 1], '-', color=color, lw=1.6, marker='o', ms=3, label=f'{label}, {tag}')
        ax.plot(raw[:, 0], raw[:, 1], '--', color=color, lw=1.3, label=f'{label}, raw')
    ideal = get(0.0, 'exchange_raw')
    ax.plot(ideal[:, 0], ideal[:, 1], color=MUTED, lw=0.9, ls=':', label='QFiLM, $e=0$')
    ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('State copies per test point')
    ax.set_ylabel('Relative joint-test MSE')
    ax.set_title(f'(a) readout flip rate $e={e_show:.0%}$'.replace('%', r'\%'), loc='left', color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=6.6, loc='upper right', ncol=1)

    ax = axes[1]
    B_show = 18000
    rates = np.array(report['flip_rates'])
    for key, color, label in [('exchange', QUANTUM, 'QFiLM'), ('gram6', GRAM, 'Gram-6'), ('transverse6', TRANS, 'transverse-6')]:
        corrected = 'gram6_corrected_unbiased' if key == 'gram6' else f'{key}_corrected'
        for k, ls, mk in [(f'{key}_raw', '--', None), (corrected, '-', 'o')]:
            vals = [next(r[k]['mean'] for r in rows if r['flip_rate'] == e and r['copies'] == B_show) / v for e in rates]
            ax.plot(100 * rates, vals, ls, color=color, lw=1.5, marker=mk, ms=3)
    ax.set_yscale('log')
    ax.text(0.02, 0.62, 'solid: corrected\ndashed: raw', transform=ax.transAxes, color=MUTED, fontsize=7.5, va='top')
    ax.set_xlabel('Readout flip rate $e$ (%)')
    ax.set_title(f'(b) {B_show:,} copies per test point', loc='left', color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(ROOT / 'paper/qip/fig_readout.pdf')
    fig.savefig(ROOT / f"results/paper_qip/fig_readout{'_clean' if CLEAN else ''}.png", dpi=150)


if __name__ == '__main__':
    main()
