"""Noise model for the benchmark states (slide "Noise: preparation and readout").

Eq. 5 and Eq. 6 are equations of our QIP paper (paper/qip/qfilm_noise_qip.tex, Sec. 4), not of the references.

qem/scaled_data.py applies three steps to each benchmark state rho:

  Step 1  preparation noise   prepare()             rho_prep = (1 - p) rho + p I/4         Eq. 5; [1] Eq. D1
          once per state; with p_sd > 0, p is drawn per state (drift).
          For a Bell input this is the Werner state of [1], Eq. D1, with F = 1 - 3p/4.
  Step 2  feature readout      measure_features()    r_hat_Q = (1 - p) r_Q + b_Q + eps_Q    Eq. 6; sigma from [2]
          r_Q = Tr(rho Q); the code passes Tr(rho_prep Q) = (1 - p) r_Q directly (15 Paulis).
          eps ~ N(0, sigma^2), b ~ N(0, offset_sd^2),
          fresh draw per state-direction row and per Pauli, so b is extra white noise, not drift.
  Step 3  rebuild and project  state_from_features(), project_physical()                  [3]
          rho_hat = nearest valid state to (I + sum_Q r_hat_Q Q) / 4. rho_hat is the model input.

Paper values: p = 0.067 and 0.667 (pure-state fidelity 1 - 3p/4 = 0.95 and 0.50, as in [1]);
p_sd = 0.03 and 0.1; sigma = 0.025 (median relative noise over 40 repeat EPR trace pairs, data of [2]);
offset_sd = 0.012. Not modelled here: shot sampling, qubit bit flips, gate errors.

[1] A. R. Morgillo, M. F. Sacchi, C. Macchiavello, Quantum Mach. Intell. 7, 47 (2025), Appendix 4.
[2] E. J. Little et al., Nat. Commun. 14, 7029 (2023).
[3] J. A. Smolin, J. M. Gambetta, G. Smith, Phys. Rev. Lett. 108, 070502 (2012).
"""
from __future__ import annotations

import numpy as np

from .data import I, PAULI, STRINGS

I4 = np.eye(4, dtype=complex)


# Step 1: preparation noise (Eq. 5) -------------------------------------------------

def depolarize(rho, p):
    """(1 - p) rho + p I/4, Eq. 5 ([1], Eq. D1). Used by prepare()."""
    p = np.asarray(p, dtype=float)[..., None, None]
    return (1 - p) * rho + p * I4 / 4


def random_axis(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def single_qubit_rotation(axis, angle):
    """exp(-i angle/2 axis.sigma)."""
    angle = np.asarray(angle, dtype=float)
    n_dot_sigma = np.einsum('nk,kij->nij', axis, PAULI[1:])
    c = np.cos(angle / 2)[:, None, None]
    s = np.sin(angle / 2)[:, None, None]
    return c * I - 1j * s * n_dot_sigma


def rotation_error(rho, delta, rng):
    """Optional over-rotation by delta * pi/2 about a random axis per qubit. Off in the paper (delta = 0)."""
    n = len(rho)
    delta = np.broadcast_to(np.asarray(delta, dtype=float), (n,))
    u1 = single_qubit_rotation(random_axis(rng, n), delta * np.pi / 2)
    u2 = single_qubit_rotation(random_axis(rng, n), delta * np.pi / 2)
    u = np.einsum('nij,nkl->nikjl', u1, u2).reshape(n, 4, 4)
    return u @ rho @ np.conj(np.swapaxes(u, 1, 2))


def prepare(rho, rng, p=0.0, p_sd=0.0, delta=0.0, delta_sd=0.0):
    """Step 1, preparation noise, Eq. 5 ([1], Eq. D1). Once per state.

    p_sd > 0 draws p per state (drift). Returns (rho_prep, p_used, delta_used).
    """
    n = len(rho)
    p_used = np.clip(rng.normal(p, p_sd, n) if p_sd > 0 else np.full(n, float(p)), 0.0, 1.0)
    delta_used = rng.normal(delta, delta_sd, n) if delta_sd > 0 else np.full(n, float(delta))
    out = depolarize(rho, p_used)
    if np.any(delta_used != 0):
        out = rotation_error(out, delta_used, rng)
    return out, p_used, delta_used


# Step 2: feature-readout noise (Eq. 6) ---------------------------------------------

def measure_features(r_true, rng, sigma, offset_sd=0.0):
    """Step 2, feature-readout noise, Eq. 6 (sigma from [2]): r_true + b + eps, shape (n, 15).

    r_true: exact Pauli expectations of the prepared state, (1 - p) r_Q in Eq. 6; one row per state-direction pair.
    b and eps are drawn independently for every row and Pauli.
    """
    n, k = r_true.shape
    b = rng.normal(0.0, offset_sd, (n, k)) if offset_sd > 0 else 0.0
    return r_true + b + rng.normal(0.0, sigma, (n, k))


# Step 3: rebuild and project [3] -----------------------------------------------------

def state_from_features(r):
    """Step 3a: (I + sum_Q r_Q Q) / 4 from the noisy features; may be unphysical."""
    return (I4 + np.einsum('nk,kij->nij', r, STRINGS[1:])) / 4


def project_physical(rho, floor=1e-9):
    """Step 3b: nearest physical state by eigenvalue clipping [3]. Gives the model input rho_hat."""
    rho = (rho + np.conj(np.swapaxes(rho, 1, 2))) / 2
    w, v = np.linalg.eigh(rho)
    out = np.empty_like(w)
    for i in range(len(w)):
        lam = w[i][::-1].copy()
        acc, d = 0.0, len(lam)
        for j in range(d - 1, -1, -1):
            if lam[j] + acc / (j + 1) < 0:
                acc += lam[j]
                lam[j] = 0.0
            else:
                lam[:j + 1] += acc / (j + 1)
                break
        out[i] = np.maximum(lam[::-1], floor)
    out /= out.sum(axis=1, keepdims=True)
    return np.einsum('nij,nj,nkj->nik', v, out, np.conj(v))
