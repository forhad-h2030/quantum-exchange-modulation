#!/usr/bin/env python3
"""Referee baselines: analytic two-copy circuit, bias-corrected local estimators, calibration budget.

No training. Realistic noise, 64 training states, 800 test-joint pairs; QFiLM = the trained models (3 seeds).
  analytic, fixed     the analytic circuit (V = exp(-i pi P/4), eta = pi/4, chi = 0), output <Z_B1> - <Z_A1>
  analytic, fitted    the same circuit with its 3-parameter head fitted by least squares on the training pairs
  Gram-6              9-basis tomography, naive squares or unbiased squares, fitted Gram-6 coefficients
  transverse-6        direction-specific: 6 local settings (qubit A along u, v perpendicular to n; qubit B along
                      X, Y, Z), unbiased squares, affine calibration fitted on the training pairs
Parts: (A) exact expectations, (B) finite copies, (C) depolarizing gate noise (qem/gate_noise.py; QFiLM values from
results/gate_noise), (D) head calibration from a finite number of copies per training pair.

    python scripts/baseline_study.py [--jobs 4]          (needs results/gate_noise/n64 from gate_noise_study.py)

Output: results/baselines/n64/{analytic_expectations.npz, summary.json}; paper Fig. 6 (paper/qip/fig_shots.pdf) and
Fig. 8 (paper/qip/fig_gates.pdf), PNG copies results/figures/baselines_{copies,gates}_n64.png
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

N = 64
SCENARIO = 'F0.95sd_clean_r0.025_off0.012'
P2 = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
BUDGETS = [180, 540, 1800, 5400, 18000, 54000, 180000]
CAL_COPIES = [180, 1800, 18000, None]                 # copies per training pair for the head fit; None = exact
CAL_TEST = [180, 1800, 18000]
REPS = 50
CAL_REPS = 20
SEED = 20261126


def analytic_job(p2):
    from qem import gate_noise as gn
    d = np.load(ROOT / f'data/noise_study/n{N}/{SCENARIO}.npz')
    noise = gn.GateNoise(p1=p2 / 10, p2=p2)
    return p2, {s: gn.analytic_expectations(d[f'{s}_rho'], d[f'{s}_angles'], noise) for s in ('test_joint', 'train')}


def analytic_cache(jobs, path):
    store = dict(np.load(path)) if path.exists() else {}
    todo = [p for p in P2 if f'{p:g}__test_joint__heads' not in store]
    if todo:
        print(f'analytic circuit: {len(todo)} gate-noise levels', flush=True)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for p2, out in ex.map(analytic_job, todo):
                for s, (h, c) in out.items():
                    store[f'{p2:g}__{s}__heads'], store[f'{p2:g}__{s}__corr'] = h, c
                np.savez(path, **store)
                print(f'  p2={p2:g}', flush=True)
    return {p: {s: (store[f'{p:g}__{s}__heads'], store[f'{p:g}__{s}__corr']) for s in ('test_joint', 'train')}
            for p in P2}


def qfilm_cache():
    """{(folder, seed, p2): {'test_joint': (h, c), 'train': (h, c), 'head': (w, b)}} from gate_noise_study."""
    store = np.load(ROOT / f'results/gate_noise/n{N}/expectations.npz')
    out = {}
    for k in store.files:
        if k.endswith('__head'):
            folder, seed, p2, _ = k.split('__')
            pre = k[:-len('__head')]
            out[(folder, int(seed), float(p2))] = {
                s: (store[f'{pre}__{s}__heads'], store[f'{pre}__{s}__corr']) for s in ('test_joint', 'train')}
            out[(folder, int(seed), float(p2))]['head'] = (store[k][:2], float(store[k][2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    from qem import gate_noise as gn
    from qem import local_estimators as le
    from qem.data import pauli_features

    out = ROOT / f'results/baselines/n{N}'
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / f'data/noise_study/n{N}/{SCENARIO}.npz')
    y, ytr = d['test_joint_y'], d['train_y']
    n_te, n_tr = d['test_joint_context'], d['train_context']
    r_te, r_tr = pauli_features(d['test_joint_rho']), pauli_features(d['train_rho'])
    var = float(y.var())
    rel = lambda p: float(np.mean((p - y) ** 2) / var)  # noqa: E731
    print('analytic circuit check (noiseless):', gn.analytic_check(d['test_joint_rho'][:50], d['test_joint_angles'][:50]))
    AN, QF = analytic_cache(a.jobs, out / 'analytic_expectations.npz'), qfilm_cache()
    rng = np.random.default_rng(SEED)
    lin = lambda X, coef: np.column_stack([X, np.ones(len(X))]) @ coef  # noqa: E731

    def fit(X, t):
        coef, *_ = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), t, rcond=None)
        return coef

    # classical fits on exact features of the (noisy) training inputs
    g_coef = fit(le.gram6_features(le.gram_exact(r_tr), n_tr), ytr)
    t_coef = fit(le.transverse_exact(r_tr, n_tr)[:, None], ytr)

    def analytic_head(p2):
        h, _ = AN[p2]['train']
        w, b = gn.fit_head(h, ytr)
        return w, b

    def qfilm_runs(folder, p2, recal):
        runs = []
        for (f, s, p), e in QF.items():
            if f == folder and p == p2:
                (h, c), (htr, _) = e['test_joint'], e['train']
                w, b = gn.fit_head(htr, ytr) if recal else e['head']
                runs.append((h, c, w, b))
        return runs

    # ---------------- (A) exact expectations, ideal gates
    h0, c0 = AN[0.0]['test_joint']
    wa, ba = analytic_head(0.0)
    A = {'analytic_fixed': rel(h0 @ gn.ANALYTIC_HEAD[0]),
         'analytic_fitted': rel(h0 @ wa + ba),
         'qfilm16': float(np.mean([rel(h @ w + b) for h, c, w, b in qfilm_runs('qfilm16', 0.0, False)])),
         'qfilm2': float(np.mean([rel(h @ w + b) for h, c, w, b in qfilm_runs('qfilm2', 0.0, False)])),
         'gram6': rel(lin(le.gram6_features(le.gram_exact(r_te), n_te), g_coef)),
         'transverse6_calibrated': rel(lin(le.transverse_exact(r_te, n_te)[:, None], t_coef)),
         'plug_in': rel(le.transverse_exact(r_te, n_te))}
    print('A exact:', {k: round(v, 4) for k, v in A.items()}, flush=True)
    print('  analytic fitted head:', np.round(wa, 3), round(ba, 4), flush=True)

    # ---------------- (B) finite copies, ideal gates
    from paper_qip_finite_shot import quantum_shots
    Bres = {}
    for B in BUDGETS:
        acc = {k: [] for k in ['analytic_fitted', 'qfilm16', 'qfilm2', 'gram6_naive', 'gram6_unbiased',
                               'transverse6_unbiased']}
        for _ in range(REPS):
            acc['analytic_fitted'].append(rel(quantum_shots(rng, h0, c0, wa, ba, B // 2)))
            for folder in ('qfilm16', 'qfilm2'):
                for h, c, w, b in qfilm_runs(folder, 0.0, False):
                    acc[folder].append(rel(quantum_shots(rng, h, c, w, b, B // 2)))
            est, n_out = le.tomography9(rng, r_te, B // 9)
            acc['gram6_naive'].append(rel(lin(le.gram6_features(le.gram_from_estimates(est, n_out, unbiased=False), n_te), g_coef)))
            acc['gram6_unbiased'].append(rel(lin(le.gram6_features(le.gram_from_estimates(est, n_out), n_te), g_coef)))
            acc['transverse6_unbiased'].append(rel(lin(le.transverse_estimate(rng, r_te, n_te, B // 6)[:, None], t_coef)))
        Bres[str(B)] = {k: float(np.mean(v)) for k, v in acc.items()}
        print(f'B B={B}:', {k: round(v, 3) for k, v in Bres[str(B)].items()}, flush=True)

    # ---------------- (C) gate noise: analytic (recalibrated) vs QFiLM 2 (recalibrated) vs unbiased Gram-6
    Cres = []
    for p2 in P2:
        f = le.axis_factors(p1=p2 / 10)
        h, c = AN[p2]['test_joint']
        w, b = analytic_head(p2)
        qmean = lambda folder, recal: float(np.mean([rel(hh @ ww + bb) for hh, cc, ww, bb in qfilm_runs(folder, p2, recal)]))  # noqa: E731
        row = {'p2': p2, 'exact': {'analytic_fixed': rel(h @ gn.ANALYTIC_HEAD[0]), 'analytic_recalibrated': rel(h @ w + b),
                                   'qfilm2_trained': qmean('qfilm2', False), 'qfilm2_recalibrated': qmean('qfilm2', True),
                                   'qfilm16_trained': qmean('qfilm16', False), 'qfilm16_recalibrated': qmean('qfilm16', True),
                                   'gram6': A['gram6'], 'transverse6_calibrated': A['transverse6_calibrated']},
               'copies': {}}
        for B in BUDGETS:
            an = [rel(quantum_shots(rng, h, c, w, b, B // 2)) for _ in range(REPS)]
            qf = [rel(quantum_shots(rng, hh, cc, ww, bb, B // 2)) for _ in range(REPS)
                  for hh, cc, ww, bb in qfilm_runs('qfilm2', p2, True)]
            gr, tr = [], []
            for _ in range(REPS):
                est, n_out = le.tomography9(rng, r_te, B // 9, f)
                gr.append(rel(lin(le.gram6_features(le.gram_from_estimates(est, n_out, f), n_te), g_coef)))
                tr.append(rel(lin(le.transverse_estimate(rng, r_te, n_te, B // 6, fA=1 - p2 / 10, fB=f)[:, None], t_coef)))
            row['copies'][str(B)] = {'analytic_recalibrated': float(np.mean(an)), 'qfilm2_recalibrated': float(np.mean(qf)),
                                     'gram6_unbiased': float(np.mean(gr)), 'transverse6_unbiased': float(np.mean(tr))}
        Cres.append(row)
        print(f"C p2={p2:g}: exact {row['exact']}", flush=True)

    # ---------------- (D) head calibration from finite copies per training pair
    from qem.local_estimators import sample_two_bit
    Dres = []
    for p2 in (1e-3, 1e-2):
        for name in ('analytic', 'qfilm2'):
            if name == 'analytic':
                circuits = [(AN[p2]['train'], AN[p2]['test_joint'])]
            else:
                circuits = [(QF[k]['train'], QF[k]['test_joint']) for k in QF if k[0] == 'qfilm2' and k[2] == p2]
            for K in CAL_COPIES:
                res = {str(B): [] for B in CAL_TEST}
                for _ in range(CAL_REPS):
                    for (htr, ctr), (h, c) in circuits:
                        if K is None:
                            hcal = htr
                        else:
                            sA, sB, _ = sample_two_bit(rng, htr[:, 0], htr[:, 1], ctr, K // 2)
                            hcal = np.column_stack([sA, sB])
                        w, b = gn.fit_head(hcal, ytr)
                        for B in CAL_TEST:
                            res[str(B)].append(rel(quantum_shots(rng, h, c, w, b, B // 2)))
                Dres.append({'p2': p2, 'model': name, 'calibration_copies_per_pair': K,
                             'test': {B: float(np.mean(v)) for B, v in res.items()}})
                print(f'D p2={p2:g} {name} K={K}:', {B: round(float(np.mean(v)), 3) for B, v in res.items()}, flush=True)

    summary = {'train_states': N, 'scenario': SCENARIO, 'var_y': var, 'budgets': BUDGETS, 'shot_batches': REPS,
               'analytic_circuit': 'V = exp(-i pi P/4) on B1, eta = pi/4, chi = 0; 32 CNOTs; fixed output <Z_B1> - <Z_A1>',
               'analytic_fitted_head': {'w': wa.tolist(), 'b': ba},
               'A_exact': A, 'B_finite_copies': Bres, 'C_gate_noise': Cres, 'D_calibration': Dres,
               'notes': 'QFiLM: mean over 3 seeds of the trained models; Gram-6 and transverse-6 coefficients fitted on '
                        'exact features of the noisy training inputs; calibration pairs = the 320 training pairs'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('wrote', out / 'summary.json')
    plot(summary)


def plot(s):
    """Paper Fig. 6 (paper/qip/fig_shots.pdf) and Fig. 8 (paper/qip/fig_gates.pdf); PNG copies in results/figures."""
    import matplotlib.pyplot as plt
    from paper_qip_figures import GRAM, INK, MUTED, QUANTUM, style  # noqa: F401  (also sets rcParams)
    Q16, TRANS = '#7FB0E8', '#7A3E9D'
    B = np.array(s['budgets'])

    # ---- Fig. 6: copies per prediction, ideal gates
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    get = lambda k: [s['B_finite_copies'][str(b)][k] for b in B]  # noqa: E731
    for k, c, ls, lw, lab in (('qfilm2', QUANTUM, '-', 1.6, 'QFiLM, 2 rotations'),
                              ('qfilm16', Q16, '-', 1.3, 'QFiLM, 16 rotations'),
                              ('analytic_fitted', INK, '-', 1.3, 'analytic circuit, fitted head'),
                              ('gram6_naive', GRAM, '--', 1.3, 'Gram-6, naive squares'),
                              ('gram6_unbiased', GRAM, '-', 1.6, 'Gram-6, unbiased squares'),
                              ('transverse6_unbiased', TRANS, '-', 1.6, 'transverse-6, unbiased squares')):
        ax.plot(B, get(k), ls, color=c, lw=lw, marker='o', ms=2.5, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('State copies per prediction'); ax.set_ylabel('Relative joint-test MSE')
    style(ax)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / 'paper/qip/fig_shots.pdf')
    fig.savefig(ROOT / f'results/figures/baselines_copies_n{N}.png', dpi=180)
    plt.close(fig)

    # ---- Fig. 8: gate errors
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    C = s['C_gate_noise']
    p2 = np.array([r['p2'] for r in C])
    x = np.where(p2 == 0, 3e-5, p2)
    ax = axes[0]
    for k, c, ls, lab in (('qfilm16_trained', Q16, '--', 'QFiLM 16 rot., trained head'),
                          ('qfilm2_trained', QUANTUM, '--', 'QFiLM 2 rot., trained head'),
                          ('analytic_fixed', INK, '--', 'analytic, fixed output'),
                          ('qfilm16_recalibrated', Q16, '-', 'QFiLM 16 rot., recalibrated'),
                          ('qfilm2_recalibrated', QUANTUM, '-', 'QFiLM 2 rot., recalibrated'),
                          ('analytic_recalibrated', INK, '-', 'analytic, recalibrated'),
                          ('gram6', GRAM, '-', 'Gram-6')):
        ax.plot(x, [r['exact'][k] for r in C], ls, color=c, marker='o', ms=2.5, lw=1.3, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(['0'] + [f'{v:g}' for v in p2[1:]], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel('Two-qubit gate error $p_2$ ($p_1=p_2/10$)'); ax.set_ylabel('Relative joint-test MSE')
    ax.set_title('(a) exact expectations', loc='left', color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=6.3, loc='upper left')
    ax = axes[1]
    for pv, ls in ((1e-3, '-'), (1e-2, '--')):
        row = next(r for r in C if r['p2'] == pv)
        for k, c, lab in (('qfilm2_recalibrated', QUANTUM, 'QFiLM 2 rot.'), ('analytic_recalibrated', INK, 'analytic'),
                          ('gram6_unbiased', GRAM, 'Gram-6'), ('transverse6_unbiased', TRANS, 'transverse-6')):
            ax.plot(B, [row['copies'][str(b)][k] for b in B], ls, color=c, lw=1.3,
                    label=lab if pv == 1e-3 else None)
    ax.plot([], [], '-', color=MUTED, label='$p_2=10^{-3}$'); ax.plot([], [], '--', color=MUTED, label='$p_2=10^{-2}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('State copies per prediction')
    ax.set_title('(b) finite copies, recalibrated heads', loc='left', color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=6.3)
    ax = axes[2]
    for pv, ls in ((1e-3, '-'), (1e-2, '--')):
        for name, c, lab in (('qfilm2', QUANTUM, 'QFiLM 2 rot.'), ('analytic', INK, 'analytic')):
            rows = [r for r in s['D_calibration'] if r['p2'] == pv and r['model'] == name]
            K = [r['calibration_copies_per_pair'] or 1e6 for r in rows]
            ax.plot(K, [r['test']['18000'] for r in rows], ls, color=c, marker='o', ms=2.5, lw=1.3,
                    label=f'{lab}, $p_2=10^{{{int(np.log10(pv))}}}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([180, 1800, 18000, 1e6]); ax.set_xticklabels(['180', '1,800', '18,000', 'exact'])
    ax.minorticks_off()
    ax.set_xlabel('Calibration copies per training pair')
    ax.set_title('(c) head calibration, 18,000 test copies', loc='left', color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=6.3)
    fig.tight_layout()
    fig.savefig(ROOT / 'paper/qip/fig_gates.pdf')
    fig.savefig(ROOT / f'results/figures/baselines_gates_n{N}.png', dpi=180)
    plt.close(fig)
    print('-> paper/qip/fig_shots.pdf, paper/qip/fig_gates.pdf (PNG copies in results/figures)')


if __name__ == '__main__':
    main()
