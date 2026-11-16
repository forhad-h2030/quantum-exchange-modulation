"""Benchmark datasets for the QIP paper.

Three state families, directions with a held-out cap (n_x > 0.55), and target
y = D_P(rho) with P = (n.sigma) x I. Optional preparation and readout noise (qem.noise_injection).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import STRINGS, directions, fragility, observable, pauli_features, random_state
from .noise_injection import measure_features, prepare, project_physical, state_from_features

EVAL_ID_BASE = 1_000_000  # id offset for evaluation states


def split_directions(rng, count, heldout, two_sided=False):
    """Directions in or out of the cap; two_sided uses |n_x| > 0.55."""
    if not two_sided:
        return directions(rng, count, heldout)
    rows = []
    while len(rows) < count:
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        if (abs(n[0]) > 0.55) == heldout:
            rows.append(n)
    return np.stack(rows)


def make_scaled_benchmark(destination, n_train_states, seed=20261009, per_state=5,
                          n_val_states=200, n_test_states=200, n_context_states=100,
                          p_prep=0.0, p_sd=0.0, delta=0.0, delta_sd=0.0,
                          readout_sigma=0.0, readout_offset_sd=0.0, fixed_eval=False,
                          target_state='prepared', two_sided=False):
    """Build one dataset and save it as .npz plus a .json manifest.

    fixed_eval: the same validation and test sets for every training size; training sets nested.
    Preparation noise is applied once per state; readout noise per state-direction row.
    target_state: 'prepared' labels the noisy state, 'clean' the intended one.
    The models read ``rho``: rho_prep, or the projected state rebuilt from noisy features.
    """
    rng = np.random.default_rng(seed)
    total = n_train_states + n_val_states + n_test_states
    a, b = n_train_states, n_train_states + n_val_states
    if fixed_eval:
        train_rng, eval_rng = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])
        pairs = [random_state(train_rng, i) for i in range(a)]
        pairs += [random_state(eval_rng, j) for j in range(total - a)]
        states, families = zip(*pairs)
        clean = np.stack(states)
        # Step 1, preparation noise (Eq. 5): noise_injection.prepare, once per state.
        prep = [prepare(clean[i:i + 1], np.random.default_rng([seed, 4 if i < a else 5, i if i < a else i - a]),
                        p_prep, p_sd, delta, delta_sd) for i in range(total)]
        states = np.concatenate([s for s, _, _ in prep])
        p_used = np.concatenate([p for _, p, _ in prep])
        delta_used = np.concatenate([d for _, _, d in prep])
        dir_rng = np.random.default_rng([seed, 3])
        outside = split_directions(dir_rng, 24, False, two_sided)
        cap = split_directions(dir_rng, 12, True, two_sided)
    else:
        states, families = zip(*(random_state(rng, i) for i in range(total)))
        clean = np.stack(states)
        # Step 1, preparation noise (Eq. 5).
        states, p_used, delta_used = prepare(clean, np.random.default_rng(seed + 1), p_prep, p_sd, delta, delta_sd)
        outside = split_directions(rng, 24, False, two_sided)
        cap = split_directions(rng, 12, True, two_sided)

    def stable_id(idx):
        return int(idx) if not fixed_eval or idx < a else EVAL_ID_BASE + int(idx) - a
    specs = {
        'train': (np.arange(a), outside[:18], per_state, 0),
        'validation': (np.arange(a, b), outside[18:], 6, 18),
        'test_state': (np.arange(b, total), outside[18:], 6, 18),
        'test_context': (np.arange(min(n_context_states, n_train_states)), cap, 4, 24),
        'test_joint': (np.arange(b, total), cap, 4, 24),
    }
    if target_state not in ('prepared', 'clean'):
        raise ValueError("target_state must be 'prepared' or 'clean'")
    out, manifest = {}, {'seed': seed, 'per_state_train': per_state, 'splits': {}, 'fixed_eval': bool(fixed_eval),
                         'context_holdout': 'two-sided |n_x| > 0.55' if two_sided else 'one-sided n_x > 0.55',
                         'target_state': target_state, 'pure_state_fidelity': 1 - 0.75 * p_prep,
                         'note': 'scaled benchmark; same structure as qem.data.make_benchmark',
                         'preparation_noise': {'p_prep': p_prep, 'p_sd': p_sd, 'delta': delta, 'delta_sd': delta_sd,
                                               'model': 'rho_prep = U[(1-p) rho + p I/4]U^dag; y computed from rho_prep',
                                               'mean_fidelity_to_clean': None},
                         'readout_noise': {'sigma': readout_sigma, 'offset_sd': readout_offset_sd,
                                           'model': 'r_hat = r_true + b + eps, b ~ N(0, offset_sd^2) per row, eps ~ N(0, sigma^2) iid; '
                                                    'rho = physical projection of (I + sum r_hat P)/4',
                                           'source': 'results/epr_readout_noise/summary.json (EPR repeat pairs)' if readout_sigma else None,
                                           'equivalent_shots': float(1 / readout_sigma ** 2) if readout_sigma else None,
                                           'floor_mse_y_of_rho_hat': {}}}
    noise_rng = np.random.default_rng(seed + 2)
    for split_k, (split, (ids, pool, k, id_offset)) in enumerate(specs.items()):
        choice_rng = np.random.default_rng([seed, 10 + split_k]) if fixed_eval else rng
        state_ids, context_ids = [], []
        for idx in ids:
            selected = choice_rng.choice(len(pool), k, replace=False)
            state_ids.extend([idx] * k)
            context_ids.extend(selected)
        sid = np.array(state_ids)
        sr = states[sid]
        c = pool[np.array(context_ids)]
        P = observable(c)
        y = fragility(clean[sid] if target_state == 'clean' else sr, P)   # label: intended or prepared state
        r_true = pauli_features(sr)[:, 1:]
        if readout_sigma > 0 or readout_offset_sd > 0:
            # Step 2, feature-readout noise (Eq. 6): noise_injection.measure_features, per row and Pauli.
            if fixed_eval:
                r_hat = np.concatenate([
                    measure_features(r_true[j:j + k], np.random.default_rng([seed, 20 + split_k, stable_id(idx)]),
                                     readout_sigma, readout_offset_sd)
                    for j, idx in zip(range(0, len(sid), k), ids)])
            else:
                r_hat = measure_features(r_true, noise_rng, readout_sigma, readout_offset_sd)
            rho_in = project_physical(state_from_features(r_hat))   # step 3: rebuild and project, rho_hat
            manifest['readout_noise']['floor_mse_y_of_rho_hat'][split] = float(np.mean((fragility(rho_in, P) - y) ** 2))
        else:
            r_hat, rho_in = r_true, sr
        values = {'rho': rho_in, 'rho_prep': sr, 'rho_clean': clean[sid], 'r_true': r_true, 'r_hat': r_hat,
                  'p_prep': p_used[sid], 'delta': delta_used[sid], 'context': c,
                  'angles': np.stack([np.arccos(c[:, 2]), np.arctan2(c[:, 1], c[:, 0])], axis=1),
                  'P': P, 'y': y,
                  'state_id': np.array([stable_id(i) for i in state_ids]), 'context_id': np.array(context_ids) + id_offset,
                  'family': np.array([families[i] for i in state_ids])}
        for key, v in values.items():
            out[f'{split}_{key}'] = v
        manifest['splits'][split] = {'examples': len(sr), 'states': len(ids), 'context_pool': len(pool),
                                     'y_mean': float(values['y'].mean()),
                                     'y_clean_mean': float(fragility(clean[sid], P).mean())}
    # Uhlmann fidelity of each prepared state to its clean state.
    from scipy.linalg import sqrtm
    fid = []
    for rho_p, rho_c in zip(states, clean):
        sa = np.asarray(sqrtm(rho_p))
        fid.append(float(np.real(np.trace(np.asarray(sqrtm(sa @ rho_c @ sa)))) ** 2))
    fid = np.asarray(fid)
    manifest['preparation_noise'].update({
        'mean_fidelity_to_clean': float(np.mean(fid)),
        'std_fidelity_to_clean': float(np.std(fid, ddof=1)) if len(fid) > 1 else 0.0,
        'min_fidelity_to_clean': float(np.min(fid)),
        'max_fidelity_to_clean': float(np.max(fid)),
        'n_unique_states_for_fidelity': int(len(fid)),
    })
    plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    minus = np.array([1, 0, 0, -1]) / np.sqrt(2)
    t = np.linspace(0, 1, 51)
    out['bell_t'] = t
    out['bell_rho'] = t[:, None, None] * np.outer(plus, plus) + (1 - t[:, None, None]) * np.outer(minus, minus)
    out['bell_context'] = np.tile([0., 0., 1.], (len(t), 1))
    out['bell_angles'] = np.zeros((len(t), 2))
    out['bell_P'] = np.broadcast_to(STRINGS[12], (len(t), 4, 4)).copy()
    out['bell_y'] = (2 * t - 1) ** 2
    assert not set(out['train_state_id']) & set(out['test_joint_state_id'])
    held = (lambda c: np.abs(c[:, 0]) > .55) if two_sided else (lambda c: c[:, 0] > .55)
    assert not np.any(held(out['train_context'])) and np.all(held(out['test_joint_context']))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **out)
    destination.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train-states', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=20261009)
    p.add_argument('--per-state', type=int, default=5)
    p.add_argument('--p-prep', type=float, default=0.0, help='depolarizing preparation probability (mean)')
    p.add_argument('--p-sd', type=float, default=0.0, help='per-state spread of p_prep')
    p.add_argument('--delta', type=float, default=0.0, help='over-rotation, fraction of a 90-degree pulse')
    p.add_argument('--delta-sd', type=float, default=0.0)
    p.add_argument('--readout-sigma', type=float, default=0.0,
                   help='white noise on each Pauli expectation')
    p.add_argument('--readout-offset-sd', type=float, default=0.0,
                   help='offset on each Pauli expectation, drawn per row')
    p.add_argument('--fixed-eval', action='store_true',
                   help='same evaluation sets for every size; nested training sets')
    p.add_argument('--two-sided', action='store_true', help='hold out |n_x| > 0.55 instead of n_x > 0.55')
    p.add_argument('--target', choices=['prepared', 'clean'], default='prepared',
                   help='label from the prepared or the intended state')
    args = p.parse_args()
    print(json.dumps(make_scaled_benchmark(args.output, args.train_states, args.seed, args.per_state,
                                           p_prep=args.p_prep, p_sd=args.p_sd, delta=args.delta,
                                           delta_sd=args.delta_sd, readout_sigma=args.readout_sigma,
                                           readout_offset_sd=args.readout_offset_sd,
                                           fixed_eval=args.fixed_eval, target_state=args.target,
                                           two_sided=args.two_sided), indent=2))
