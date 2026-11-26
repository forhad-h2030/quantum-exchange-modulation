#!/usr/bin/env python3
"""Gate-noise study: the trained QFiLM models run as compiled circuits with depolarizing gate errors.

No retraining. For each two-qubit error rate p2 (single-qubit p1 = p2 / 10):
  QFiLM, 16 and 2 rotations (40 and 32 CNOTs), 3 seeds, on the 800 test-joint pairs:
    trained head     the head from training on ideal gates
    recalibrated     the 3 head parameters refitted by least squares on the noisy-circuit outputs of the
                     320 training pairs (a calibration step, no circuit parameters change)
  Gram-6 from 9-basis tomography whose basis-change gates carry the same p1:
    raw              as measured
    corrected        divided by the known attenuation (1 - p1) per non-Z basis
Exact expectations and finite copies (same budgets and shot model as scripts/paper_qip_finite_shot.py).
Input noise (realistic scenario, 64 training states) is unchanged; gate noise acts inside the circuit.

    python scripts/gate_noise_study.py [--train-states 64] [--jobs 4]

Output: results/gate_noise/n<N>/{expectations.npz, summary.json}, results/figures/gate_noise_n<N>.{pdf,png}
Circuit compilation and simulation: qem/gate_noise.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

SCENARIO = 'F0.95sd_clean_r0.025_off0.012'
MODELS = {'qfilm16': ('joint_exchange', 'QFiLM, 16 rotations'), 'qfilm2': ('joint_exchange_w2', 'QFiLM, 2 rotations')}
P2 = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
BUDGETS = [180, 540, 1800, 5400, 18000, 54000, 180000]
REPS = 50
SEED = 20261125


def load_weights(n, folder, key):
    from noisy_bell import quantum_weights
    return quantum_weights(ROOT / f'results/noise_study/n{n}/{folder}/{SCENARIO}', key)


def expectations_job(args):
    """Noisy-circuit port expectations for one (model, seed, p2) on the test and training pairs."""
    n, folder, key, seed, p2 = args
    import torch
    torch.set_num_threads(1)
    from qem import gate_noise as gn
    from qem.pennylane_models import TorchQuantumModel
    d = np.load(ROOT / f'data/noise_study/n{n}/{SCENARIO}.npz')
    model = TorchQuantumModel(key, initial_parameters=load_weights(n, folder, key)[seed])
    noise = gn.GateNoise(p1=p2 / 10, p2=p2)
    out = {}
    for split in ('test_joint', 'train'):
        h, c = gn.qfilm_expectations(model, d[f'{split}_rho'], d[f'{split}_angles'], noise)
        out[split] = (h, c)
    return args, out, gn.head_of(model)


def compute_expectations(n, jobs, cache):
    """{(folder, seed, p2): {'test_joint': (heads, corr), 'train': ..., 'head': (w, b)}}, cached in an .npz."""
    store = dict(np.load(cache)) if cache.exists() else {}
    todo = []
    for folder, (key, _) in MODELS.items():
        for seed in sorted(load_weights(n, folder, key)):
            for p2 in P2:
                if f'{folder}__{seed}__{p2:g}__test_joint__heads' not in store:
                    todo.append((n, folder, key, seed, p2))
    if todo:
        print(f'computing {len(todo)} noisy-circuit evaluations with {jobs} jobs', flush=True)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for (n_, folder, key, seed, p2), out, (w, b) in ex.map(expectations_job, todo):
                pre = f'{folder}__{seed}__{p2:g}'
                for split, (h, c) in out.items():
                    store[f'{pre}__{split}__heads'], store[f'{pre}__{split}__corr'] = h, c
                store[f'{pre}__head'] = np.r_[w, b]
                print(f'  {folder} seed {seed} p2={p2:g}', flush=True)
                np.savez(cache, **store)
    res = {}
    for k in store:
        if k.endswith('__head'):
            folder, seed, p2, _ = k.split('__')
            pre = k[:-len('__head')]
            res[(folder, int(seed), float(p2))] = {
                'test_joint': (store[f'{pre}__test_joint__heads'], store[f'{pre}__test_joint__corr']),
                'train': (store[f'{pre}__train__heads'], store[f'{pre}__train__corr']),
                'head': (store[k][:2], float(store[k][2]))}
    return res


def tomography(rng, r16, shots, A):
    """9-basis tomography with per-entry attenuation A (4, 4): raw Pauli means, shape (m, 4, 4)."""
    from paper_qip_readout_flip import sample_means
    m = len(r16)
    R = r16.reshape(m, 4, 4)
    est = np.zeros((m, 4, 4))
    est[:, 0, 0] = 1
    for a in (1, 2, 3):
        for c in (1, 2, 3):
            fa, fc = np.sqrt(A[a, a]), np.sqrt(A[c, c])
            sA, sB, cor = sample_means(rng, fa * R[:, a, 0], fc * R[:, 0, c], fa * fc * R[:, a, c], shots)
            est[:, a, c] = cor
            est[:, a, 0] += sA / 3
            est[:, 0, c] += sB / 3
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-states', type=int, default=64)
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    n = a.train_states
    from qem import gate_noise as gn
    from qem.data import pauli_features
    from paper_qip_finite_shot import gram_fit, gram_from_r, quantum_shots
    gn.self_test()

    out = ROOT / f'results/gate_noise/n{n}'
    out.mkdir(parents=True, exist_ok=True)
    E = compute_expectations(n, a.jobs, out / 'expectations.npz')
    d = np.load(ROOT / f'data/noise_study/n{n}/{SCENARIO}.npz')
    y, y_tr, ctx = d['test_joint_y'], d['train_y'], d['test_joint_context']
    var = float(y.var())
    rel = lambda pred: float(np.mean((pred - y) ** 2) / var)  # noqa: E731
    r16 = pauli_features(d['test_joint_rho'])
    coef = gram_fit(d)
    gram = lambda R: np.column_stack([gram_from_r(R.reshape(-1, 16), ctx), np.ones(len(R))]) @ coef  # noqa: E731
    rng = np.random.default_rng(SEED)

    rows = []
    for p2 in P2:
        A = gn.tomography_attenuation(p2 / 10)
        row = {'p2': p2, 'p1': p2 / 10, 'exact': {}, 'copies': {}}
        heads_by = {}
        for folder in MODELS:
            seeds = sorted(s for (f, s, p) in E if f == folder and p == p2)
            ex_tr, ex_re, hs = [], [], []
            for s in seeds:
                e = E[(folder, s, p2)]
                (h, c), (htr, _), (w, b) = e['test_joint'], e['train'], e['head']
                wr, br = gn.fit_head(htr, y_tr)
                ex_tr.append(rel(h @ w + b))
                ex_re.append(rel(h @ wr + br))
                hs.append((h, c, (w, b), (wr, br)))
            heads_by[folder] = hs
            row['exact'][f'{folder}_trained_head'] = float(np.mean(ex_tr))
            row['exact'][f'{folder}_recalibrated'] = float(np.mean(ex_re))
        row['exact']['gram6_raw'] = rel(gram(r16.reshape(-1, 4, 4) * A))
        row['exact']['gram6_corrected'] = rel(gram(r16.reshape(-1, 4, 4)))
        for B in BUDGETS:
            acc = {}
            for folder, hs in heads_by.items():
                for tag, which in (('trained_head', 0), ('recalibrated', 1)):
                    vals = []
                    for _ in range(REPS):
                        for h, c, head_trained, head_recal in hs:
                            w, b = (head_trained, head_recal)[which]
                            vals.append(rel(quantum_shots(rng, h, c, w, b, B // 2)))
                    acc[f'{folder}_{tag}'] = vals
            g_raw, g_cor = [], []
            for _ in range(REPS):
                est = tomography(rng, r16, B // 9, A)
                g_raw.append(rel(gram(est)))
                g_cor.append(rel(gram(est / A)))
            acc['gram6_raw'], acc['gram6_corrected'] = g_raw, g_cor
            row['copies'][str(B)] = {k: float(np.mean(v)) for k, v in acc.items()}
        rows.append(row)
        print(f"p2={p2:g}: exact " + ', '.join(f'{k} {v:.4f}' for k, v in row['exact'].items()), flush=True)

    # copies below which the better QFiLM (recalibrated) beats corrected Gram-6, per p2
    for row in rows:
        best = [min(row['copies'][str(B)][f'{f}_recalibrated'] for f in MODELS) for B in BUDGETS]
        gr = [row['copies'][str(B)]['gram6_corrected'] for B in BUDGETS]
        wins = [B for B, q, g in zip(BUDGETS, best, gr) if q < g]
        row['qfilm_better_at_copies'] = wins
    counts = {}
    from qem.pennylane_models import TorchQuantumModel
    for folder, (key, _) in MODELS.items():
        m = TorchQuantumModel(key, initial_parameters=load_weights(n, folder, key)[17])
        counts[folder] = gn.gate_counts(gn.compile_qfilm(m, d['test_joint_angles'][:1]))
    summary = {'train_states': n, 'scenario': SCENARIO, 'split': 'test_joint', 'var_y': var,
               'p1_over_p2': 0.1, 'budgets': BUDGETS, 'shot_batches': REPS, 'gate_counts_per_sign': counts,
               'noise_model': 'depolarizing after every gate: p1 (single-qubit), p2 (CNOT); ideal readout',
               'recalibration': 'head (w_A, w_B, w_0) refitted by least squares on noisy-circuit outputs of the '
                                'training pairs; circuit parameters unchanged',
               'rows': rows}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('wrote', out / 'summary.json')
    plot(summary, n)


def plot(s, n):
    import matplotlib.pyplot as plt
    rows = s['rows']
    p2 = np.array([r['p2'] for r in rows])
    x = np.where(p2 == 0, 3e-5, p2)
    col = {'qfilm16': '#1f4e8c', 'qfilm2': '#5fa8e0', 'gram6': '#b5470f'}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    for f, (_, lab) in MODELS.items():
        ax.plot(x, [r['exact'][f'{f}_trained_head'] for r in rows], '--', color=col[f], marker='o', ms=3.5,
                label=f'{lab}, trained head')
        ax.plot(x, [r['exact'][f'{f}_recalibrated'] for r in rows], '-', color=col[f], marker='o', ms=3.5,
                label=f'{lab}, recalibrated')
    ax.plot(x, [r['exact']['gram6_raw'] for r in rows], '-', color=col['gram6'], marker='^', ms=3.5,
            label='Gram-6 (tomography)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(['0'] + [f'{v:g}' for v in p2[1:]], fontsize=8)
    ax.set_xlabel('two-qubit gate error $p_2$ ($p_1=p_2/10$)')
    ax.set_ylabel('relative test MSE (exact expectations)')
    ax.legend(frameon=False, fontsize=7)
    ax.set_title('(a) gate noise, exact expectations', fontsize=9)
    ax = axes[1]
    B = np.array(s['budgets'])
    for r, ls in ((rows[0], ':'), (next(r for r in rows if r['p2'] == 1e-3), '-'),
                  (next(r for r in rows if r['p2'] == 1e-2), '--')):
        tag = f"$p_2$={r['p2']:g}"
        ax.plot(B, [r['copies'][str(b)]['qfilm2_recalibrated'] for b in B], ls, color=col['qfilm2'],
                label=f'QFiLM 2 rot., {tag}')
        ax.plot(B, [r['copies'][str(b)]['gram6_corrected'] for b in B], ls, color=col['gram6'],
                label=f'Gram-6, {tag}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('state copies per prediction')
    ax.set_ylabel('relative test MSE')
    ax.legend(frameon=False, fontsize=7, ncol=1)
    ax.set_title('(b) finite copies (QFiLM recalibrated, Gram-6 corrected)', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(ROOT / f'results/figures/gate_noise_n{n}.{ext}', dpi=180)
    print(f'-> results/figures/gate_noise_n{n}.pdf, .png')


if __name__ == '__main__':
    main()
