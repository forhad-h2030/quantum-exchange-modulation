#!/usr/bin/env python3
"""Directional response D_P(n) over the sphere, reconstructed under realistic noise.

    python scripts/noisy_direction_map.py                     # 64 training states (default)
    python scripts/noisy_direction_map.py --train-states 400  # full noise-study training set

Scenario: drift + readout + offset (F0.95sd_clean_r0.025_off0.012). Needs:
  - data/noise_study/n<N>/<scenario>.npz   (scripts/noise_study.py data)
  - QFiLM runs for the two rotation counts in --compare and every seed in --seeds
    (scripts/noise_study.py rotations); classical FiLM is trained here if missing
    -> results/noisy_direction_map/n<N>/film/
Evaluation on one held-out test state (--state typical: median spread of Gram eigenvalues):
  the state is prepared with depolarizing p drawn once per redraw (drift), each direction gets
  fresh Pauli-estimate noise (sigma, offset) and the nearest physical rho_hat
  (qem/noise_injection.py); QFiLM is read out with finite copies, FiLM sees the same rho_hat.
  The target is the exact D_P of the intended (clean) state.
Figure: (a) exact | QFiLM k0 | QFiLM k1 maps (first seed, one noise draw); (b) equator cut theta = pi/2:
  dot = mean over seeds of each model's mean over the redraws, bar = s.d. over the redraws within a
  model, pooled over seeds; FiLM mean +- s.d.
  -> results/figures/noisy_direction_map_n<N>.{pdf,png}, results/noisy_direction_map/n<N>/summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from qem.pennylane_models import TorchQuantumModel  # noqa: E402
from qem.torch_classical import TorchFiLM  # noqa: E402
from qem.data import pauli_features  # noqa: E402
from qem.noise_injection import depolarize, measure_features, state_from_features, project_physical  # noqa: E402
from plot_direction_map import gram, exact_map, fit_gram  # noqa: E402

SCENARIO = 'F0.95sd_clean_r0.025_off0.012'
P_MEAN, P_SD, SIGMA, OFFSET = 0.0666667, 0.03, 0.025, 0.012
SPLITS = ['train', 'validation', 'test_state', 'test_context', 'test_joint']


def dataset(n_states):
    path = ROOT / f'data/noise_study/n{n_states}/{SCENARIO}.npz'
    if not path.exists():
        sys.exit(f'missing {path.relative_to(ROOT)}; run: python scripts/noise_study.py data '
                 f'--train-states {n_states} --scenarios {SCENARIO}')
    return path


def train_film(data, out):
    f = out / 'film'
    if not (f / 'film.pt').exists():
        print('training classical FiLM (3 seeds, validation-selected) ...', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/train_torch_classical.py'),
                        '--data', str(data), '--output', str(f), '--seeds', '17,31,73'], check=True)
    return f


def load_weights(rot, k, key, seeds):
    """Weights per seed from k<k>/weights.npz or the per-seed folders k<k>/seed<s>/weights.npz."""
    found = {}
    for f in [rot / f'k{k}' / 'weights.npz', *sorted((rot / f'k{k}').glob('seed*/weights.npz'))]:
        if f.exists():
            w = np.load(f)
            for s in seeds:
                if s not in found and f'{key}__seed{s}' in w.files:
                    found[s] = w[f'{key}__seed{s}']
    missing = [s for s in seeds if s not in found]
    return found, missing


def noisy_inputs(r_prep, n_dir, rng):
    """Fresh Pauli-estimate noise per direction, then the nearest physical state."""
    r_hat = measure_features(np.repeat(r_prep[None], n_dir, 0), rng, SIGMA, OFFSET)
    return project_physical(state_from_features(r_hat))


def qfilm_shots(model, w, rho_hat, angles, copies, rng):
    with torch.no_grad():
        out = model(rho_hat, angles)
    heads, corr = out['heads'].numpy(), out['correlation'].numpy()
    h = model.head_start
    wa, wb, w0 = w[h], w[h + 1], w[h + 2]
    if copies <= 0:
        return out['prediction'].numpy()
    shots = copies // 2
    st = np.array([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    prob = np.clip((1 + heads[:, :1] * st[:, 0] + heads[:, 1:] * st[:, 1] + corr[:, None] * st[:, 0] * st[:, 1]) / 4,
                   0, None)
    prob /= prob.sum(1, keepdims=True)
    means = np.array([rng.multinomial(shots, p) for p in prob]) @ st / shots
    return wa * means[:, 0] + wb * means[:, 1] + w0


def film_predict(film, rho_hat, n):
    with torch.no_grad():
        x = torch.tensor(pauli_features(rho_hat)[:, 1:], dtype=torch.float64)
        return film(x, torch.tensor(n, dtype=torch.float64)).numpy()


def sphere(theta, phi):
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-states', type=int, default=64)
    ap.add_argument('--copies', type=int, default=20000, help='copies per direction (two per execution)')
    ap.add_argument('--compare', default='16,2', help='two rotation counts k, from scripts/noise_study.py rotations')
    ap.add_argument('--seeds', default='17,31,73,101,137', help='training seeds pooled in the error bars')
    ap.add_argument('--state', choices=['typical', 'anisotropic'], default='typical',
                    help='test state: median or largest spread of Gram eigenvalues')
    ap.add_argument('--repeats', type=int, default=30, help='noise + shot redraws for the equator error bars')
    ap.add_argument('--grid', default='45,90')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(2026)

    data_path = dataset(a.train_states)
    out = ROOT / f'results/noisy_direction_map/n{a.train_states}'

    ks = [int(x) for x in a.compare.split(',')]
    seeds = [int(x) for x in a.seeds.split(',')]
    rot = ROOT / f'results/noise_study/n{a.train_states}/rotations/{SCENARIO}'
    mkey = lambda k: 'joint_exchange' if k == 16 else f'joint_exchange_w{k}'  # noqa: E731
    models = {}
    for k in ks:
        wts, missing = load_weights(rot, k, mkey(k), seeds)
        if missing:
            sys.exit(f'k={k}: no trained run for seeds {missing}; run: python scripts/noise_study.py rotations '
                     f'--train-states {a.train_states} --scenarios {SCENARIO} --rotations {a.compare} --seeds {a.seeds}')
        models[k] = [(TorchQuantumModel(mkey(k), initial_parameters=wts[s]), wts[s]) for s in seeds]
    k0, k1 = ks
    fdir = train_film(data_path, out)
    ck = torch.load(fdir / 'film.pt', weights_only=False)
    film = TorchFiLM(width=ck['width'])
    film.load_state_dict(ck['state_dict'])
    film.eval()
    ftest = ck['selected_run']['metrics']

    d = np.load(data_path)
    ids, first = np.unique(d['test_joint_state_id'], return_index=True)
    states = d['test_joint_rho_clean'][first]
    spread = np.array([np.ptp(np.linalg.eigvalsh(gram(s))) for s in states])
    k = int(np.argmax(spread)) if a.state == 'anisotropic' else int(np.argsort(spread)[len(spread) // 2])
    rho = states[k]
    p_used = float(np.clip(rng.normal(P_MEAN, P_SD), 0, 1))           # drift: one draw per state
    r_prep = pauli_features(depolarize(rho[None], p_used))[0, 1:]

    # (a) maps over the sphere: one noise realisation, finite copies
    nt, nph = (int(v) for v in a.grid.split(','))
    TH, PH = np.meshgrid(np.linspace(0, np.pi, nt), np.linspace(0, 2 * np.pi, nph), indexing='ij')
    n = sphere(TH, PH).reshape(-1, 3)
    ang = np.stack([TH.ravel(), PH.ravel()], -1)
    exact = exact_map(rho, n)
    rho_hat = noisy_inputs(r_prep, len(n), rng)
    q_map = qfilm_shots(*models[k0][0], rho_hat, ang, a.copies, rng)      # maps: first seed of each model
    q1_map = qfilm_shots(*models[k1][0], rho_hat, ang, a.copies, rng)
    f_map = film_predict(film, rho_hat, n)

    # (b) equator cut, theta = pi/2, with repeat draws of the noise and the shots
    ph = np.linspace(0, 2 * np.pi, 73)
    ne = sphere(np.full_like(ph, np.pi / 2), ph)
    ange = np.stack([np.full_like(ph, np.pi / 2), ph], -1)
    exact_e = exact_map(rho, ne)
    q_rep, q1_rep, f_rep = [], [], []
    for _ in range(a.repeats):
        rr = pauli_features(depolarize(rho[None], float(np.clip(rng.normal(P_MEAN, P_SD), 0, 1))))[0, 1:]
        rh = noisy_inputs(rr, len(ne), rng)
        q_rep += [qfilm_shots(m, w, rh, ange, a.copies, rng) for m, w in models[k0]]    # all seeds, same input
        q1_rep += [qfilm_shots(m, w, rh, ange, a.copies, rng) for m, w in models[k1]]
        f_rep.append(film_predict(film, rh, ne))
    q_rep, q1_rep, f_rep = np.array(q_rep), np.array(q1_rep), np.array(f_rep)

    def per_model(rep_, n_models):
        """(draws*models, dirs) -> dot, bar, seed spread.
        Dot: mean over models of each model's mean over the noise draws.
        Bar: each model's s.d. over the draws, pooled over models (root mean variance).
        Seed spread: s.d. of the per-model means (reported, not drawn)."""
        r = rep_.reshape(a.repeats, n_models, -1)
        means = r.mean(0)
        return means.mean(0), np.sqrt(r.var(0).mean(0)), means.std(0)

    q_dot, q_bar, q_seed = per_model(q_rep, len(models[k0]))
    q1_dot, q1_bar, q1_seed = per_model(q1_rep, len(models[k1]))

    G_exact, G_fit = gram(rho), fit_gram(n, q_map)
    rel = lambda m, e: np.mean((m - e) ** 2) / np.var(e)  # noqa: E731
    names = {k0: f'qfilm_{k0}rot', k1: f'qfilm_{k1}rot'}
    summary = {'train_states': a.train_states, 'state': a.state, 'state_id': int(ids[k]),
               'gram_eigenvalue_spread': float(spread[k]), 'p_prep': p_used, 'copies': a.copies,
               'seeds_pooled': seeds, 'repeats': a.repeats,
               'map_relative_mse': {names[k0]: float(rel(q_map, exact)), names[k1]: float(rel(q1_map, exact)),
                                    'film': float(rel(f_map, exact))},
               'equator_mse_of_mean': {names[k0]: float(np.mean((q_dot - exact_e) ** 2)),
                                       names[k1]: float(np.mean((q1_dot - exact_e) ** 2)),
                                       'film': float(np.mean((f_rep.mean(0) - exact_e) ** 2))},
               'equator_noise_sd_within_model': {names[k0]: float(q_bar.mean()), names[k1]: float(q1_bar.mean())},
               'equator_seed_sd_of_means': {names[k0]: float(q_seed.mean()), names[k1]: float(q1_seed.mean())},
               'equator_mean_sd': {names[k0]: float(q_rep.std(0).mean()), names[k1]: float(q1_rep.std(0).mean()),
                                   'film': float(f_rep.std(0).mean())},
               'film_test_joint_mse': ftest['test_joint']['mse'],
               'gram_eigenvalues': {'exact': np.linalg.eigvalsh(G_exact).tolist(),
                                    'qfilm_fit': np.linalg.eigvalsh(G_fit).tolist()}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

    # ---------------- figure ----------------
    # Style of results/physics_applications/directional_response_map: viridis maps in degrees, phi in
    # [-180, 180) so the held-out cap (around phi = 0) is contiguous, and a diverging prediction - exact row.
    plt.rcParams.update({'font.size': 9.5, 'axes.linewidth': 0.7})
    crim, blue, film_c, gate2_c = '#c0392b', '#1f4e8c', '#b7791f', '#5fa8e0'
    wrap = lambda deg: (np.asarray(deg) + 180) % 360 - 180  # noqa: E731
    order = np.argsort(wrap(np.degrees(PH[0])))
    phd, thd = wrap(np.degrees(PH[0]))[order], np.degrees(TH[:, 0])
    grid = lambda m: m.reshape(TH.shape)[:, order]  # noqa: E731
    train_n = np.unique(d['train_context'].round(6), axis=0)
    tr_th = np.degrees(np.arccos(train_n[:, 2]))
    tr_ph = wrap(np.degrees(np.arctan2(train_n[:, 1], train_n[:, 0])))
    cap = np.linspace(-np.arccos(0.55), np.arccos(0.55), 200)            # boundary n_x = 0.55
    cap_th = np.degrees(np.arcsin(np.clip(0.55 / np.cos(cap), -1, 1)))
    cap_ph, cap_th = np.degrees(np.r_[cap, cap[::-1]]), np.r_[cap_th, 180 - cap_th[::-1]]
    mir_ph = wrap(cap_ph + 180)                                          # mirror cap n_x < -0.55: same axes
    cut = np.r_[0, np.flatnonzero(np.abs(np.diff(mir_ph)) > 180) + 1, len(mir_ph)]

    fig = plt.figure(figsize=(10.4, 8.6))
    W, H, X0, DX = 0.235, 0.205, 0.07, 0.268
    vmin, vmax = exact.min(), exact.max()
    maps = [('Exact $D_P$', grid(exact)), (f'QFiLM, {k0} rotations', grid(q_map)),
            (f'QFiLM, {k1} rotations', grid(q1_map))]
    diffs = [None, grid(q_map - exact), grid(q1_map - exact)]
    lim = np.quantile(np.abs(np.concatenate([q_map - exact, q1_map - exact])), 0.99)

    def decorate(ax, row, col):
        ax.plot(cap_ph, cap_th, color=crim, lw=1.6, ls='--',   # white edge keeps it visible on red pixels
                path_effects=[pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal()])
        for i0, i1 in zip(cut[:-1], cut[1:]):
            ax.plot(mir_ph[i0:i1], cap_th[i0:i1], color='0.85', lw=1.3, ls=':')
        ax.scatter(tr_ph, tr_th, s=12, facecolor='none', edgecolor='white', lw=0.7, zorder=3)
        ax.set_xlim(-180, 180); ax.set_ylim(0, 180)
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([0, 45, 90, 135, 180])
        ax.set_xlabel(r'Azimuth $\varphi$ (degrees)')
        if col == 0 or (row == 1 and col == 1):
            ax.set_ylabel(r'Polar angle $\theta$ (degrees)')
        else:
            ax.set_yticklabels([])

    for i, (name, m) in enumerate(maps):
        ax = fig.add_axes([X0 + DX * i, 0.73, W, H])
        im = ax.pcolormesh(phd, thd, m, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
        ax.set_title(name)
        decorate(ax, 0, i)
    cb = fig.colorbar(im, cax=fig.add_axes([X0 + DX * 3 - 0.02, 0.73, 0.012, H]))
    cb.set_label(r'$D_P(\mathbf{n})$')
    for i, m in enumerate(diffs):
        if m is None:
            continue
        ax = fig.add_axes([X0 + DX * i, 0.43, W, H])
        imd = ax.pcolormesh(phd, thd, m, cmap='coolwarm', vmin=-lim, vmax=lim, shading='auto', rasterized=True)
        ax.set_title(f'Prediction $-$ exact, {ks[i - 1]} rotations')
        decorate(ax, 1, i)
    cb = fig.colorbar(imd, cax=fig.add_axes([X0 + DX * 3 - 0.02, 0.43, 0.012, H]))
    cb.set_label('prediction $-$ exact')
    fig.text(X0, 0.60, f'One noise draw,\n{a.copies:,} copies\nper direction.\n\n'
             'White circles:\ntraining directions.\n\nRed dashed:\nheld-out cap $n_x>0.55$\n(test directions\nlie inside).\n\nGrey dotted: mirror\ncap $n_x<-0.55$, same\naxes, not held out.',
             fontsize=8.5, va='top')
    fig.text(0.005, 0.965, '(a)', fontsize=12, weight='bold')

    # (b) equator cut, same degree axis
    o = np.argsort(wrap(np.degrees(ph[:-1])))
    x = wrap(np.degrees(ph[:-1]))[o]
    sel = lambda v: np.asarray(v)[:-1][o]  # noqa: E731
    ax = fig.add_axes([X0, 0.07, DX * 3 - 0.035, 0.24])
    ax.axvspan(-np.degrees(np.arccos(0.55)), np.degrees(np.arccos(0.55)), color=crim, alpha=0.08, lw=0)
    for lo, hi in ((-180, -180 + np.degrees(np.arccos(0.55))), (180 - np.degrees(np.arccos(0.55)), 180)):
        ax.axvspan(lo, hi, color='0.5', alpha=0.10, lw=0)
    ax.plot(x, sel(exact_e), color='k', lw=1.2, label=r'exact $D_P(\mathbf{n})$', zorder=5)
    ax.errorbar(x - 0.9, sel(q_dot), yerr=sel(q_bar), fmt='o', ms=3.4, color=blue, mfc='white', mew=0.9,
                elinewidth=0.8, capsize=0, label=f'QFiLM, {k0} rotations', zorder=4)
    ax.errorbar(x + 0.9, sel(q1_dot), yerr=sel(q1_bar), fmt='D', ms=2.8, color=gate2_c, mew=0.6,
                elinewidth=0.8, capsize=0, label=f'QFiLM, {k1} rotations', zorder=3)
    fm, fs = sel(f_rep.mean(0)), sel(f_rep.std(0))
    ax.plot(x, fm, color=film_c, lw=1.4, ls='--', label='classical FiLM')
    ax.fill_between(x, fm - fs, fm + fs, color=film_c, alpha=0.15, lw=0)
    eq = np.abs(tr_th - 90) < 9
    ax.plot(tr_ph[eq], np.full(eq.sum(), ax.get_ylim()[0]), '^', color='k', mfc='white', ms=5, mew=0.6,
            clip_on=False)
    ax.set_xlim(-180, 180); ax.set_xticks(np.arange(-180, 181, 45))
    ax.set_xlabel(r'Azimuth $\varphi$ (degrees), equator $\theta=90^\circ$')
    ax.set_ylabel(r'$D_P(\mathbf{n})$')
    ax.legend(frameon=False, fontsize=8.5, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.16))
    ax.text(0.5, 0.04, 'red: held-out cap; grey: mirror cap (same axes, not held out)', transform=ax.transAxes, fontsize=8, color=crim, ha='center')
    fig.text(0.005, 0.345, '(b)', fontsize=12, weight='bold')

    figs = ROOT / 'results/figures'
    figs.mkdir(parents=True, exist_ok=True)
    stem = f'noisy_direction_map_n{a.train_states}' + ('' if a.state == 'typical' else '_anisotropic')
    fig.savefig(figs / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(figs / f'{stem}.png', dpi=200, bbox_inches='tight')
    print(f'-> results/figures/{stem}.pdf, .png')


if __name__ == '__main__':
    main()
