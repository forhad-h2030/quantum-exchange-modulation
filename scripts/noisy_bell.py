"""Bell-family response of the noise-trained models, on noisy Bell inputs (README figure, paper Fig. 9).

Bell family rho_t = t Phi+ + (1 - t) Phi-, P = Z(x)I, intended label (2t - 1)^2.
Each test input gets the training noise of the realistic scenario, as in scripts/noisy_direction_map.py:
depolarizing p ~ N(0.067, 0.03) drawn per state and redraw, Pauli-estimate noise (sigma 0.025,
offset 0.012), nearest physical rho_hat. QFiLM is read out with finite copies; FiLM sees rho_hat.
Line = mean over seeds of each model's mean over the redraws; band = +-1 s.d. over the redraws
within a model, pooled over seeds.

    python scripts/noisy_bell.py --train-states 64     # README: QFiLM 16 and 2, FiLM (results/noise_study/n64)
    python scripts/noisy_bell.py --paper               # paper Fig. 9: the 400-state models

Output: results/figures/noisy_bell_n<N>.{png,pdf,json}; with --paper also paper/qip/fig_bell.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from qem.data import pauli_features  # noqa: E402
from qem.noise_injection import depolarize, measure_features, project_physical, state_from_features  # noqa: E402
from qem.pennylane_models import TorchQuantumModel  # noqa: E402
from qem.torch_classical import TorchFiLM  # noqa: E402
from noisy_direction_map import OFFSET, P_MEAN, P_SD, SCENARIO, SIGMA, film_predict, qfilm_shots  # noqa: E402
from paper_qip_figures import CLASSICAL, INK, QUANTUM, noise_source, style  # noqa: E402

QFILM2 = '#6fa8e8'


def quantum_weights(d, key):
    """{seed: weights} from d/weights.npz and d/seed*/weights.npz."""
    out = {}
    for f in [d / 'weights.npz', *sorted(d.glob('seed*/weights.npz'))]:
        if f.exists():
            w = np.load(f)
            for k in w.files:
                if k.startswith(f'{key}__seed'):
                    out.setdefault(int(k.split('seed')[-1]), w[k])
    return out


def film_models(d):
    m = json.loads((d / 'metrics.json').read_text())
    w = np.load(d / 'weights.npz')
    out = {}
    for run in m['selected_runs']:
        f = TorchFiLM(width=run['width'])
        with torch.no_grad():
            f.weights.copy_(torch.tensor(w[run['key']]))
        out[run['seed']] = f
    return out


def evaluate(data_path, quantum, film_dir, repeats, copies, rng):
    """{label: (mean curve, band s.d., per-seed means)} on noisy Bell inputs, plus the clean-input check."""
    d = np.load(data_path)
    t, rho, ctx, ang = d['bell_t'], d['bell_rho'], d['bell_context'], d['bell_angles']
    models = {lab: [(TorchQuantumModel(key, initial_parameters=w), w) for _, w in sorted(ws.items())]
              for lab, key, ws in quantum}
    films = list(film_models(film_dir).values())
    reps = {lab: [[] for _ in m] for lab, m in models.items()}
    reps['Classical FiLM'] = [[] for _ in films]
    for _ in range(repeats):
        p = np.clip(rng.normal(P_MEAN, P_SD, len(t)), 0, 1)              # drift: one p per state and redraw
        r_prep = pauli_features(depolarize(rho, p))[:, 1:]
        rho_hat = noisy_inputs_rows(r_prep, rng)
        for lab, ms in models.items():
            for i, (m, w) in enumerate(ms):
                reps[lab][i].append(qfilm_shots(m, w, rho_hat, ang, copies, rng))
        for i, f in enumerate(films):
            reps['Classical FiLM'][i].append(film_predict(f, rho_hat, ctx))
    out = {}
    for lab, r in reps.items():
        r = np.array(r)                                                  # (seeds, repeats, t)
        out[lab] = (r.mean((0, 1)), r.std(1, ddof=1).mean(0), r.mean(1))
    clean = {lab: float(np.mean([np.mean((qfilm_shots(m, w, rho, ang, 0, rng) - (2 * t - 1) ** 2) ** 2)
                                 for m, w in ms])) for lab, ms in models.items()}
    return t, out, clean


def noisy_inputs_rows(r_prep, rng):
    """Pauli-estimate noise and projection, one row per state."""
    return project_physical(state_from_features(measure_features(r_prep, rng, SIGMA, OFFSET)))


def plot(t, curves, colors, title, paths, figsize):
    import matplotlib.pyplot as plt
    exact = (2 * t - 1) ** 2
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, exact, color=INK, lw=1.2, ls='--', label=r'exact $(2t-1)^2$', zorder=5)
    for lab, (mean, sd, _) in curves.items():
        ax.fill_between(t, mean - sd, mean + sd, color=colors[lab], alpha=0.18, lw=0, zorder=2)
        ax.plot(t, mean, color=colors[lab], lw=2, label=lab, zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r'Bell weight $t$')
    ax.set_ylabel(r'$D_{Z\otimes I}$')
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc='upper center')
    fig.tight_layout()
    for p in paths:
        fig.savefig(p, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-states', type=int, default=64)
    ap.add_argument('--paper', action='store_true', help='the 400-state models of the paper; writes paper/qip/fig_bell.pdf')
    ap.add_argument('--repeats', type=int, default=30, help='noise + shot redraws per Bell state')
    ap.add_argument('--copies', type=int, default=20000, help='copies per prediction (two per execution)')
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = np.random.default_rng(2027)

    if a.paper:
        n = 400
        data_path, qdirs, fdir = noise_source(SCENARIO)
        ws = {}
        for qd in qdirs:
            ws.update(quantum_weights(qd, 'joint_exchange'))
        quantum = [('QFiLM', 'joint_exchange', ws)]
        colors = {'QFiLM': QUANTUM, 'Classical FiLM': CLASSICAL}
    else:
        n = a.train_states
        rdir = ROOT / f'results/noise_study/n{n}'
        data_path, fdir = ROOT / f'data/noise_study/n{n}/{SCENARIO}.npz', rdir / f'film/{SCENARIO}'
        quantum = [('QFiLM, 16 rotations', 'joint_exchange', quantum_weights(rdir / f'qfilm16/{SCENARIO}', 'joint_exchange')),
                   ('QFiLM, 2 rotations', 'joint_exchange_w2', quantum_weights(rdir / f'qfilm2/{SCENARIO}', 'joint_exchange_w2'))]
        colors = {'QFiLM, 16 rotations': QUANTUM, 'QFiLM, 2 rotations': QFILM2, 'Classical FiLM': CLASSICAL}
    quantum = [q for q in quantum if q[2]]

    t, curves, clean = evaluate(data_path, quantum, fdir, a.repeats, a.copies, rng)
    exact = (2 * t - 1) ** 2
    summary = {'train_states': n, 'scenario': SCENARIO, 'repeats': a.repeats, 'copies': a.copies,
               'var_exact': float(exact.var()),
               'mse_of_mean_curve': {lab: float(np.mean((c[0] - exact) ** 2)) for lab, c in curves.items()},
               'mse_single_prediction': {lab: float(np.mean(c[1] ** 2) + np.mean((c[0] - exact) ** 2))
                                         for lab, c in curves.items()},
               'mean_band_sd': {lab: float(c[1].mean()) for lab, c in curves.items()},
               'clean_input_mse_exact_readout': clean,
               'seeds': {lab: len(c[2]) for lab, c in curves.items()}}
    out = ROOT / f'results/figures/noisy_bell_n{n}'
    paths = [out.with_suffix('.png'), out.with_suffix('.pdf')]
    if a.paper:
        paths.append(ROOT / 'paper/qip/fig_bell.pdf')
    plot(t, curves, colors, f'Noisy Bell inputs, trained with {n} states', paths,
         (4.2, 2.9) if a.paper else (4.6, 3.1))
    out.with_suffix('.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    print('wrote', ', '.join(str(p.relative_to(ROOT)) for p in paths))


if __name__ == '__main__':
    main()
