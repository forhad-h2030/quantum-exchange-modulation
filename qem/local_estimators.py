"""Classical estimators of the dephasing response from local Pauli measurements, with finite shots.

    est, n_out = tomography9(rng, r16, shots)                 # 9-basis tomography: Pauli means and shot counts
    G = gram_from_estimates(est, n_out, unbiased=True)        # Gram matrix with unbiased squares
    X = gram6_features(G, n)                                   # the 6 Gram features of Eq. (gram)
    D = transverse_estimate(rng, r16, n, shots)               # 6-setting, direction-specific estimate of D_n

Pauli index order I, X, Y, Z; r16[:, 4a + b] = Tr[rho (sigma_a (x) sigma_b)].
Each measured mean of N outcomes +-1 satisfies E[r_hat^2] = f^2 r^2 + (1 - f^2 r^2) / N for attenuation f,
so (N r_hat^2 - 1) / ((N - 1) f^2) is an unbiased estimate of r^2. Products of means from different
settings are independent and need no correction.
"""
from __future__ import annotations

import numpy as np

OUTCOMES = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])


def sample_two_bit(rng, m1, m2, m12, shots):
    """Sample `shots` outcomes of two +-1 bits with means m1, m2 and correlator m12 (one row per state).

    Returns the empirical means of bit 1, bit 2 and their product.
    """
    probs = (1 + OUTCOMES[:, 0] * m1[:, None] + OUTCOMES[:, 1] * m2[:, None] + np.outer(m12, OUTCOMES.prod(1))) / 4
    probs = np.clip(probs, 0, None)
    counts = rng.multinomial(shots, probs / probs.sum(1, keepdims=True))
    s = counts @ OUTCOMES / shots
    return s[:, 0], s[:, 1], counts @ OUTCOMES.prod(1) / shots


def axis_factors(p1=0.0, flip=0.0):
    """Attenuation of a measured mean per Pauli axis (I, X, Y, Z): one depolarizing basis-change gate for X and
    Y (factor 1 - p1), none for Z; a bit flip with probability `flip` multiplies every mean by 1 - 2 flip."""
    return np.array([1.0, 1 - p1, 1 - p1, 1.0]) * np.array([1.0] + [1 - 2 * flip] * 3)


def tomography9(rng, r16, shots, f=None):
    """Raw Pauli means from `shots` shots in each of the 9 local bases, and the outcomes behind each mean.

    f: per-axis attenuation (axis_factors), default none. Marginals pool the 3 bases that contain them.
    """
    f = np.ones(4) if f is None else np.asarray(f)
    m = len(r16)
    R = r16.reshape(m, 4, 4)
    est = np.zeros((m, 4, 4))
    est[:, 0, 0] = 1
    for a in (1, 2, 3):
        for c in (1, 2, 3):
            sA, sB, cor = sample_two_bit(rng, f[a] * R[:, a, 0], f[c] * R[:, 0, c], f[a] * f[c] * R[:, a, c], shots)
            est[:, a, c] = cor
            est[:, a, 0] += sA / 3
            est[:, 0, c] += sB / 3
    n_out = np.full((4, 4), float(shots))
    n_out[:, 0] = n_out[0, :] = 3 * shots
    return est, n_out


def gram_from_estimates(est, n_out, f=None, unbiased=True):
    """G_ac = sum_b r_ab r_cb (a, c in X, Y, Z; b in I, X, Y, Z) from estimated Pauli means.

    Means are divided by the known attenuation f x f. With unbiased=True the diagonal squares use
    (N r_hat^2 - 1) / ((N - 1) f^2); off-diagonal products come from independent settings.
    """
    f = np.ones(4) if f is None else np.asarray(f)
    A = np.outer(f, f)
    r = (est / A)[:, 1:, :]
    G = np.einsum('nab,ncb->nac', r, r)
    if unbiased:
        N, Af = n_out[1:, :], A[1:, :]
        sq = (N * est[:, 1:, :] ** 2 - 1) / ((N - 1) * Af ** 2)
        idx = np.arange(3)
        G[:, idx, idx] = sq.sum(2)
    return G


def gram6_features(G, n):
    """The six Gram features: (1 - n_i^2) G_ii / 2 and -n_i n_j G_ij (i < j)."""
    diag = 0.5 * (1 - n ** 2) * np.diagonal(G, axis1=1, axis2=2)
    cross = np.stack([-n[:, a] * n[:, b] * G[:, a, b] for a, b in [(0, 1), (0, 2), (1, 2)]], axis=1)
    return np.concatenate([diag, cross], axis=1)


def gram_exact(r16):
    r = r16.reshape(-1, 4, 4)[:, 1:, :]
    return np.einsum('nab,ncb->nac', r, r)


def transverse_axes(n):
    """Orthonormal u, v perpendicular to each unit vector n."""
    e = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    u = np.cross(n, e)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u, np.cross(n, u)


def transverse_exact(r16, n):
    """D_n = (1/2) sum_b (r_ub^2 + r_vb^2), b in I, X, Y, Z, u and v perpendicular to n."""
    R = r16.reshape(-1, 4, 4)
    out = 0
    for w in transverse_axes(n):
        rw = np.einsum('na,nab->nb', w, R[:, 1:, :])
        out = out + (rw ** 2).sum(1)
    return 0.5 * out


def transverse_estimate(rng, r16, n, shots, unbiased=True, fA=1.0, fB=None, correct=True):
    """Direction-specific estimate of D_n from 6 local settings: qubit A along u or v, qubit B along X, Y or Z.

    `shots` per setting (6 * shots copies in total). fA: attenuation of qubit A's mean (basis change along u, v
    and readout); fB: per-axis attenuation of qubit B (I, X, Y, Z), as axis_factors. With correct=True the means
    are divided by the known attenuation; squares are unbiased if requested.
    """
    fB = np.ones(4) if fB is None else np.asarray(fB)
    R = r16.reshape(-1, 4, 4)
    att = np.array([fA, fA * fB[1], fA * fB[2], fA * fB[3]]) if correct else np.ones(4)
    total = 0
    for w in transverse_axes(n):
        rw = np.einsum('na,nab->nb', w, R[:, 1:, :])            # (m, 4): r_wI, r_wX, r_wY, r_wZ
        est = np.zeros_like(rw)
        for b in (1, 2, 3):
            sA, _, cor = sample_two_bit(rng, fA * rw[:, 0], fB[b] * R[:, 0, b], fA * fB[b] * rw[:, b], shots)
            est[:, b] = cor
            est[:, 0] += sA / 3
        N = np.array([3.0 * shots, shots, shots, shots])
        sq = (N * est ** 2 - 1) / (N - 1) if unbiased else est ** 2
        total = total + (sq / att ** 2).sum(1)
    return 0.5 * total
