#!/usr/bin/env python3
"""Exact vs reconstructed directional response D_P(n) over the sphere, for one held-out state.

    python scripts/plot_direction_map.py [--copies 2000] [--state-index -1]

(a) D_P(n) on a (phi, theta) grid: exact, and QFiLM reconstructed from finite copies.
(b) The 3x3 Gram matrix G that fixes the map, D_P = (Tr G - n^T G n)/2: exact, and the
    least-squares fit to the reconstructed QFiLM map.
Uses the saved clean-data 16-state QFiLM (seed 17).
No training. Output: results/figures/direction_map.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qem.pennylane_models import TorchQuantumModel

PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.])]


def gram(rho):
    r = np.array([[np.trace(rho @ np.kron(PAULI[a], PAULI[b])).real for b in range(4)] for a in range(4)])
    return r[1:, :] @ r[1:, :].T


def exact_map(rho, n):
    G = gram(rho)
    return 0.5 * (np.trace(G) - np.einsum('ki,ij,kj->k', n, G, n))


def fit_gram(n, y):
    """Least-squares G from D(n) = sum_i (1-n_i^2)/2 G_ii - sum_{i<j} n_i n_j G_ij."""
    i, j = np.triu_indices(3, 1)
    X = np.column_stack([0.5 * (1 - n ** 2), -n[:, i] * n[:, j]])
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    G = np.diag(c[:3])
    G[i, j] = G[j, i] = c[3:]
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--copies', type=int, default=2000, help='state copies per direction (two per execution)')
    ap.add_argument('--state-index', type=int, default=-1, help='test-joint state; -1 picks the most anisotropic')
    ap.add_argument('--grid', default='45,90', help='theta,phi grid points')
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = np.random.default_rng(2026)

    data = np.load(ROOT / 'data/scaled_fixed/benchmark_n16.npz')
    ids, first = np.unique(data['test_joint_state_id'], return_index=True)
    states = data['test_joint_rho'][first]
    if a.state_index < 0:
        spread = [np.ptp(np.linalg.eigvalsh(gram(s))) for s in states]
        k = int(np.argmax(spread))
    else:
        k = a.state_index
    rho = states[k]

    nt, nph = (int(v) for v in a.grid.split(','))
    theta = np.linspace(0, np.pi, nt)
    phi = np.linspace(0, 2 * np.pi, nph)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    n = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
    angles = np.stack([TH.ravel(), PH.ravel()], -1)
    exact = exact_map(rho, n)

    # QFiLM: exact port means and correlation, then exact sampling of the joint (+-1, +-1) outcomes.
    w = np.load(ROOT / 'results/clean/quantum/n16/weights.npz')['joint_exchange__seed17']
    model = TorchQuantumModel('joint_exchange', initial_parameters=w)
    with torch.no_grad():
        out = model(np.repeat(rho[None], len(n), 0), angles)
    heads, corr = out['heads'].numpy(), out['correlation'].numpy()
    q_exact = out['prediction'].numpy()
    h = model.head_start
    wa, wb, w0 = w[h], w[h + 1], w[h + 2]
    shots = a.copies // 2
    s_t = np.array([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    prob = np.clip((1 + heads[:, :1] * s_t[:, 0] + heads[:, 1:] * s_t[:, 1] + corr[:, None] * s_t[:, 0] * s_t[:, 1]) / 4, 0, None)
    prob /= prob.sum(1, keepdims=True)
    counts = np.array([rng.multinomial(shots, p) for p in prob])
    means = counts @ s_t / shots
    q_shot = wa * means[:, 0] + wb * means[:, 1] + w0

    maps = [('Exact', exact), (f'Reconstructed (QFiLM, {a.copies:,} copies)', q_shot)]
    grams = [('Exact', gram(rho)), ('Reconstructed (QFiLM)', fit_gram(n, q_shot))]
    for name, m in maps:
        print(f'{name:24s} MSE vs exact {np.mean((m - exact) ** 2):.2e}')
    print(f"{'QFiLM exact expectations':24s} MSE vs exact {np.mean((q_exact - exact) ** 2):.2e}")
    for name, G in grams:
        print(f'{name:16s} G eigenvalues {np.round(np.linalg.eigvalsh(G), 3)}')

    # ---- figure, in the style of Jin, Ye, Ma, PRA 114, L040401, Fig. 3 ----
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 10, 'axes.linewidth': 0.6})
    fig = plt.figure(figsize=(6.6, 6.6))
    vmin, vmax = min(m.min() for _, m in maps), max(m.max() for _, m in maps)
    train = np.unique(data['train_context'].round(6), axis=0)
    tr_th, tr_ph = np.arccos(train[:, 2]), np.mod(np.arctan2(train[:, 1], train[:, 0]), 2 * np.pi)
    cap_ph = np.linspace(-np.arccos(0.55), np.arccos(0.55), 200)
    cap_th1 = np.arcsin(np.clip(0.55 / np.cos(cap_ph), -1, 1))
    for i, (name, m) in enumerate(maps):
        ax = fig.add_axes([0.09 + 0.40 * i, 0.56, 0.34, 0.36])
        im = ax.pcolormesh(PH / np.pi, TH / np.pi, m.reshape(TH.shape), cmap='Blues', vmin=vmin, vmax=vmax,
                           shading='auto', rasterized=True)
        for sgn in (0, 2):  # cap n_x > 0.55 wraps around phi = 0
            ph = np.mod(cap_ph, 2 * np.pi)
            seg = (ph < 1) if sgn == 0 else (ph > 1)
            ax.plot(ph[seg] / np.pi, cap_th1[seg] / np.pi, color='#c0392b', lw=0.9, ls='--')
            ax.plot(ph[seg] / np.pi, 1 - cap_th1[seg] / np.pi, color='#c0392b', lw=0.9, ls='--')
        ax.scatter(tr_ph / np.pi, tr_th / np.pi, s=10, facecolor='white', edgecolor='k', lw=0.5, zorder=3)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(r'$\varphi/\pi$')
        if i == 0:
            ax.set_ylabel(r'$\theta/\pi$')
        else:
            ax.set_yticklabels([])
        ax.set_xlim(0, 2)
        ax.set_ylim(1, 0)
        ax.tick_params(direction='in', length=3)
    cax = fig.add_axes([0.87, 0.56, 0.02, 0.36])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(direction="in", length=3)
    cb.set_label(r"$D_{P}(\mathbf{n})$")
    fig.text(0.01, 0.95, '(a)', fontsize=12)

    lab = ['x', 'y', 'z']
    xx, yy = np.meshgrid(np.arange(3), np.arange(3), indexing='ij')
    gmax = max(np.abs(G).max() for _, G in grams)
    cmap = plt.get_cmap('Blues')
    for i, (name, G) in enumerate(grams):
        ax = fig.add_axes([0.02 + 0.45 * i, 0.04, 0.44, 0.42], projection='3d')
        vals = G.ravel()
        ax.bar3d(xx.ravel() - 0.3, yy.ravel() - 0.3, np.minimum(vals, 0), 0.6, 0.6, np.abs(vals),
                 color=cmap(0.25 + 0.75 * np.abs(vals) / gmax), edgecolor='white', linewidth=0.3, shade=False)
        ax.set_xticks(range(3), lab)
        ax.set_yticks(range(3), lab)
        ax.set_zlim(min(0, min(G.min() for _, G in grams)), gmax)
        ax.set_title(name, fontsize=11, pad=-2)
        ax.view_init(elev=28, azim=-50)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor('#dddddd')
        ax.tick_params(labelsize=8, pad=0)
        if i == 0:
            ax.text2D(0.02, 0.62, r'$G_{ij}$', transform=ax.transAxes, fontsize=11)
    fig.text(0.01, 0.47, '(b)', fontsize=12)

    out = ROOT / 'results/figures'
    fig.savefig(out / 'direction_map.pdf', bbox_inches='tight')
    fig.savefig(out / 'direction_map.png', dpi=200, bbox_inches='tight')
    print('state', int(ids[k]), '-> results/figures/direction_map.pdf, .png')


if __name__ == '__main__':
    main()
