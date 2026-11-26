"""Gate-level QFiLM circuit with gate noise: compilation, density-matrix simulation, tomography attenuation.

Reusable pieces (no training here):

    ops = compile_qfilm(model, angles, sign=+1)          # native gates: 1-qubit unitaries and CNOTs
    gate_counts(ops)                                      # {'single_qubit': .., 'cnot': ..}
    heads, corr = qfilm_expectations(model, rho, angles, GateNoise(p1=1e-4, p2=1e-3))
    max_compilation_error(model, rho, angles)             # noiseless compiled circuit vs TorchQuantumModel
    tomography_attenuation(p1)                            # Pauli-estimate attenuation of 9-basis tomography

Wires: 0, 1 = copy A (A1, A2); 2, 3 = copy B (B1, B2); 4 = ancilla of the exchange (starts in |0>).
Wire 0 is the most significant bit, as in PennyLane.

Native gate set: arbitrary single-qubit unitaries and CNOT. The exchange exp(-i eta S) uses the ancilla:
H, two controlled-swaps, H, RZ(2 eta), H, two controlled-swaps, H (as global_exchange in
qem/pennylane_models.py); each controlled-swap is CNOT, Toffoli (6 CNOTs), CNOT. Adjacent single-qubit gates
on one wire are fused into one gate, as a compiler would.

Noise model: after every single-qubit gate, depolarizing noise with probability p1 on its wire; after every
CNOT, two-qubit depolarizing noise with probability p2 on its wires:
    rho -> (1 - p) rho + p (I / 2^k) (x) Tr_wires(rho),   k = 1 or 2.
For k = 2 the average gate infidelity is 3 p2 / 4 (k = 1: p1 / 2). State preparation, idling, crosstalk and
readout are not included here (readout bit flips: scripts/paper_qip_readout_flip.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from qem.pennylane_models import TorchQuantumModel, _linked_angle

N_WIRES = 5
ANCILLA = 4
_LETTERS = 'abcdefghijklmnopqrstuvwxyz'

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.diag([1, np.exp(1j * np.pi / 4)])
TDG = T.conj().T
PAULI = {'X': X, 'Y': Y, 'Z': Z}


@dataclass(frozen=True)
class GateNoise:
    """Depolarizing probabilities after each single-qubit gate (p1) and each CNOT (p2)."""
    p1: float = 0.0
    p2: float = 0.0


# ---------------------------------------------------------------- gates
def rotation(pauli: str, angle) -> np.ndarray:
    """exp(-i angle P / 2) for a batch of angles: shape (N, 2, 2), or (2, 2) for a scalar."""
    angle = np.asarray(angle, dtype=float)
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return c[..., None, None] * I2 - 1j * s[..., None, None] * PAULI[pauli]


def u1(wire, matrix):
    return ('u1', (wire,), np.asarray(matrix, dtype=complex))


def cx(control, target):
    return ('cx', (control, target), None)


def zz(angle, a, b):
    """exp(-i angle ZZ / 2) = CNOT, RZ(angle) on the target, CNOT."""
    return [cx(a, b), u1(b, rotation('Z', angle)), cx(a, b)]


def two_qubit_rotation(word, angle, a, b):
    """exp(-i angle PQ / 2) for PQ in XX, YY, ZZ, by a local basis change around zz."""
    basis = {'Z': I2, 'X': H, 'Y': H @ np.diag([1, -1j])}       # basis @ P @ basis^dagger = Z
    p, q = word
    va, vb = basis[p], basis[q]
    return [u1(a, va), u1(b, vb), *zz(angle, a, b), u1(a, va.conj().T), u1(b, vb.conj().T)]


def toffoli(c1, c2, t):
    """Standard 6-CNOT Toffoli (Nielsen and Chuang, Fig. 4.9)."""
    return [u1(t, H), cx(c2, t), u1(t, TDG), cx(c1, t), u1(t, T), cx(c2, t), u1(t, TDG), cx(c1, t),
            u1(c2, T), u1(t, T), u1(t, H), cx(c1, c2), u1(c1, T), u1(c2, TDG), cx(c1, c2)]


def cswap(c, a, b):
    return [cx(b, a), *toffoli(c, a, b), cx(b, a)]


def exchange(eta, control=ANCILLA, pairs=((0, 2), (1, 3))):
    """exp(-i eta S) on the data registers, S the swap of copy A and copy B; ancilla starts and ends in |0>."""
    swaps = [g for a, b in pairs for g in cswap(control, a, b)]
    return [u1(control, H), *swaps, u1(control, H), u1(control, rotation('Z', 2 * np.asarray(eta))),
            u1(control, H), *swaps, u1(control, H)]


def fuse_single_qubit(ops):
    """Merge runs of single-qubit gates on one wire (later gates multiply from the left)."""
    pending, out = {}, []

    def flush(wire):
        if wire in pending:
            out.append(u1(wire, pending.pop(wire)))
    for kind, wires, matrix in ops:
        if kind == 'u1':
            w = wires[0]
            pending[w] = matrix @ pending[w] if w in pending else matrix
        else:
            for w in wires:
                flush(w)
            out.append((kind, wires, matrix))
    for w in sorted(pending):
        flush(w)
    return out


# ---------------------------------------------------------------- compile the trained model
def _parameters(model: TorchQuantumModel):
    return model.parameters_vector.detach()


def compile_qfilm(model: TorchQuantumModel, angles, sign: float = 1.0, fuse: bool = True):
    """Native-gate circuit of one exchange sign, for a batch of directions angles = (theta, phi), shape (N, 2)."""
    if not model._is_exchange_family:
        raise ValueError('compile_qfilm supports the exchange (QFiLM) models only')
    params = _parameters(model)
    ang = torch.as_tensor(np.asarray(angles), dtype=torch.float64)
    b_wires = (2, 3)
    ops = []
    for kind, payload, links in model._specifications:
        angle = _linked_angle(params, ang, links).numpy()
        if kind == 'cnot':
            ops.append(cx(b_wires[payload[0]], b_wires[payload[1]]))
            continue
        active = [(p, b_wires[i]) for i, p in enumerate(payload) if p != 'I']
        if len(active) == 1:
            (p, w), = active
            ops.append(u1(w, rotation(p, angle)))
        elif len(active) == 2 and active[0][0] == active[1][0]:
            ops += two_qubit_rotation(active[0][0] * 2, angle, active[0][1], active[1][1])
        else:
            raise ValueError(f'unsupported rotation {payload}')
    eta = float(params[model._reference.eta_index])
    ops += exchange(sign * eta)
    theta, phi = np.asarray(angles)[:, 0], np.asarray(angles)[:, 1]
    for w in (0, 2):                                     # rotate the direction onto z, then measure Z
        ops += [u1(w, rotation('Z', -phi)), u1(w, rotation('Y', -theta))]
    return fuse_single_qubit(ops) if fuse else ops


def gate_counts(ops):
    return {'single_qubit': sum(k == 'u1' for k, _, _ in ops), 'cnot': sum(k == 'cx' for k, _, _ in ops)}


# ---------------------------------------------------------------- batched density-matrix simulator
def _apply(rho, matrix, wires, n=N_WIRES):
    """rho -> U rho U^dagger on `wires`; rho (N, 2^n, 2^n), U (2^k, 2^k) or (N, 2^k, 2^k)."""
    N, k = rho.shape[0], len(wires)
    U = np.broadcast_to(matrix if matrix.ndim == 3 else matrix[None], (N, 2 ** k, 2 ** k))
    t = rho.reshape((N,) + (2,) * (2 * n))
    rows, cols = [1 + w for w in wires], [1 + n + w for w in wires]
    t = np.moveaxis(t, rows, range(-k, 0))
    shape = t.shape
    t = np.einsum('n...j,nij->n...i', t.reshape(shape[:-k] + (2 ** k,)), U).reshape(shape)
    t = np.moveaxis(t, range(-k, 0), rows)
    t = np.moveaxis(t, cols, range(-k, 0))
    shape = t.shape
    t = np.einsum('n...j,nij->n...i', t.reshape(shape[:-k] + (2 ** k,)), U.conj()).reshape(shape)
    t = np.moveaxis(t, range(-k, 0), cols)
    return t.reshape(rho.shape)


_CNOT = np.eye(4, dtype=complex)[[0, 1, 3, 2]]


def depolarize_wires(rho, p, wires, n=N_WIRES):
    """(1 - p) rho + p (I / 2^k) (x) Tr_wires(rho)."""
    if p == 0:
        return rho
    N, k = rho.shape[0], len(wires)
    t = rho.reshape((N,) + (2,) * (2 * n))
    row, col = list(_LETTERS[:n]), list(_LETTERS[n:2 * n])
    traced = col.copy()
    for w in wires:
        traced[w] = row[w]
    keep = [i for i in range(n) if i not in wires]
    red_idx = ''.join(row[i] for i in keep) + ''.join(col[i] for i in keep)
    red = np.einsum('Z' + ''.join(row) + ''.join(traced) + '->Z' + red_idx, t)
    eye = (np.eye(2 ** k) / 2 ** k).reshape((2,) * (2 * k))
    eye_idx = ''.join(row[w] for w in wires) + ''.join(col[w] for w in wires)
    mixed = np.einsum(f'Z{red_idx},{eye_idx}->Z' + ''.join(row) + ''.join(col), red, eye)
    return ((1 - p) * t + p * mixed).reshape(rho.shape)


def simulate(rho, ops, noise: GateNoise = GateNoise()):
    """Run native ops on a batch of 5-qubit density matrices with gate noise."""
    for kind, wires, matrix in ops:
        if kind == 'u1':
            rho = depolarize_wires(_apply(rho, matrix, wires), noise.p1, wires)
        else:
            rho = depolarize_wires(_apply(rho, _CNOT, wires), noise.p2, wires)
    return rho


def _z_signs(wire, n=N_WIRES):
    bits = (np.arange(2 ** n) >> (n - 1 - wire)) & 1
    return 1 - 2 * bits


def initial_state(rho):
    """rho (x) rho (x) |0><0| on the 5 wires, for a batch rho of shape (N, 4, 4)."""
    rho = np.asarray(rho, dtype=complex)
    zero = np.diag([1.0, 0.0]).astype(complex)
    pair = np.einsum('nij,nkl->nikjl', rho, rho).reshape(len(rho), 16, 16)
    return np.einsum('nij,kl->nikjl', pair, zero).reshape(len(rho), 32, 32)


def port_expectations(rho, angles, branches, noise: GateNoise = GateNoise(), chunk: int = 200):
    """Port expectations for a probabilistic mixture of circuits.

    branches: [(weight, builder)], builder(angles_chunk) -> native ops. Returns heads (N, 2) = <Z_A1>, <Z_B1>
    and corr (N,) = <Z_A1 Z_B1>, averaged over the branches with their weights.
    """
    rho, angles = np.asarray(rho), np.asarray(angles, dtype=float)
    sa, sb = _z_signs(0), _z_signs(2)
    heads, corr = [], []
    for i in range(0, len(rho), chunk):
        r0, a = initial_state(rho[i:i + chunk]), angles[i:i + chunk]
        v = 0
        for weight, build in branches:
            if weight == 0:
                continue
            diag = np.real(np.diagonal(simulate(r0, build(a), noise), axis1=1, axis2=2))
            v = v + weight * np.stack([diag @ sa, diag @ sb, diag @ (sa * sb)], -1)
        heads.append(v[:, :2])
        corr.append(v[:, 2])
    return np.concatenate(heads), np.concatenate(corr)


def qfilm_expectations(model: TorchQuantumModel, rho, angles, noise: GateNoise = GateNoise(),
                       fuse: bool = True, chunk: int = 200):
    """Trained QFiLM: both exchange signs, mixed with weights cos^2(chi/2) and sin^2(chi/2), as in training."""
    chi = float(_parameters(model)[model._reference.chi_index])
    q = np.cos(chi / 2) ** 2
    return port_expectations(rho, angles, [(q, lambda a: compile_qfilm(model, a, 1.0, fuse)),
                                           (1 - q, lambda a: compile_qfilm(model, a, -1.0, fuse))], noise, chunk)


def compile_analytic(angles, fuse: bool = True):
    """The analytic one-layer circuit: V = exp(-i pi P / 4) on B1 with P = n.sigma, eta = pi/4, chi = 0.

    Noiselessly, <Z_B1> - <Z_A1> = D_P(rho) (checked by analytic_check). No trained parameters.
    """
    angles = np.asarray(angles, dtype=float)
    theta, phi = angles[:, 0], angles[:, 1]
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    n_sigma = np.einsum('ni,ijk->njk', n, np.stack([X, Y, Z]))
    V = np.cos(np.pi / 4) * I2 - 1j * np.sin(np.pi / 4) * n_sigma
    ops = [u1(2, V), *exchange(np.pi / 4)]
    for w in (0, 2):
        ops += [u1(w, rotation('Z', -phi)), u1(w, rotation('Y', -theta))]
    return fuse_single_qubit(ops) if fuse else ops


def analytic_expectations(rho, angles, noise: GateNoise = GateNoise(), chunk: int = 200):
    """Port expectations of the analytic circuit (one exchange sign, chi = 0)."""
    return port_expectations(rho, angles, [(1.0, compile_analytic)], noise, chunk)


ANALYTIC_HEAD = (np.array([-1.0, 1.0]), 0.0)          # D_P = <Z_B1> - <Z_A1>


def dephasing_response(rho, angles):
    """Exact D_P(rho) = Tr rho^2 - Tr(rho P rho P), P = (n.sigma) (x) I, for a batch."""
    rho, angles = np.asarray(rho), np.asarray(angles, dtype=float)
    theta, phi = angles[:, 0], angles[:, 1]
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    P = np.einsum('nij,kl->nikjl', np.einsum('ni,ijk->njk', n, np.stack([X, Y, Z])), I2).reshape(-1, 4, 4)
    rp = rho @ P
    return np.real(np.einsum('nij,nji->n', rho, rho) - np.einsum('nij,nji->n', rp, rp))


def analytic_check(rho, angles):
    """Largest |(<Z_B1> - <Z_A1>) - D_P| of the noiseless compiled analytic circuit."""
    heads, _ = analytic_expectations(rho, angles)
    w, b = ANALYTIC_HEAD
    return float(np.abs(heads @ w + b - dephasing_response(rho, angles)).max())


def head_of(model: TorchQuantumModel):
    """Trained head weights (w_A, w_B) and bias w_0."""
    p, h = _parameters(model).numpy(), model.head_start
    return p[h:h + 2], float(p[h + 2])


def fit_head(heads, y):
    """Least-squares recalibration of the 3 head parameters on (noisy-circuit) port expectations."""
    A = np.column_stack([heads, np.ones(len(heads))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[:2], float(coef[2])


def max_compilation_error(model: TorchQuantumModel, rho, angles):
    """Largest difference between the noiseless compiled circuit and the trained PennyLane model."""
    heads, corr = qfilm_expectations(model, rho, angles)
    with torch.no_grad():
        ref = model(torch.as_tensor(np.asarray(rho)), torch.as_tensor(np.asarray(angles, dtype=float)))
    return float(max(np.abs(heads - ref['heads'].numpy()).max(), np.abs(corr - ref['correlation'].numpy()).max()))


# ---------------------------------------------------------------- tomography under gate noise
def tomography_attenuation(p1: float = 0.0, flip: float = 0.0):
    """(4, 4) factors multiplying the Pauli estimates r_ab of 9-basis tomography (index order I, X, Y, Z).

    Measuring X or Y needs one (fused) single-qubit basis-change gate, Z none; a depolarizing gate multiplies
    the measured mean by (1 - p1). A bit flip with probability `flip` multiplies it by (1 - 2 flip).
    """
    f = np.array([1.0, 1 - p1, 1 - p1, 1.0]) * np.array([1.0] + [1 - 2 * flip] * 3)
    return np.outer(f, f)


def self_test(seed=0):
    """Check the native decompositions against exact matrices."""
    rng = np.random.default_rng(seed)
    def unitary(ops, n):
        U = np.eye(2 ** n, dtype=complex)
        for kind, wires, m in ops:
            g = _CNOT if kind == 'cx' else m
            perm = list(wires) + [w for w in range(n) if w not in wires]
            P = np.eye(2 ** n)[[int(''.join(str((i >> (n - 1 - perm.index(w))) & 1) for w in range(n)), 2)
                                for i in range(2 ** n)]]
            full = P.T @ np.kron(g, np.eye(2 ** (n - len(wires)))) @ P
            U = full @ U
        return U
    ccx = np.eye(8, dtype=complex)[[0, 1, 2, 3, 4, 5, 7, 6]]
    assert np.allclose(unitary(toffoli(0, 1, 2), 3), ccx)
    sw = np.eye(8, dtype=complex)[[0, 1, 2, 3, 4, 6, 5, 7]]
    assert np.allclose(unitary(cswap(0, 1, 2), 3), sw)
    for word in ('XX', 'YY', 'ZZ'):
        a = rng.uniform(-3, 3)
        P = np.kron(PAULI[word[0]], PAULI[word[1]])
        assert np.allclose(unitary(two_qubit_rotation(word, a, 0, 1), 2),
                           np.cos(a / 2) * np.eye(4) - 1j * np.sin(a / 2) * P)
    return True
