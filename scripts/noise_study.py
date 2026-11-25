#!/usr/bin/env python3
"""Noise study: data -> training -> noise table, Bell figure, learning curve, rotation study.

    python scripts/noise_study.py all --train-states 64 --jobs 4         # data, train, report
    python scripts/noise_study.py all --train-states 8,16,32,64,128,256 \
        --scenarios F0.95sd_clean_r0.025_off0.012                       # learning curve
    python scripts/noise_study.py rotations --train-states 64 --scenarios F0.95sd_clean_r0.025_off0.012

Scenarios: the 8 of scripts/paper_qip_data.py (default), 'clean', or --custom "name:p=..,p_sd=..,sigma=..,
offset=..,target=clean|prepared". Models: qfilm16 (default), qfilm2, qfilm1, qnn, film.
Data in data/noise_study/n<N>/, results in results/noise_study/n<N>/. Finished seeds are skipped;
new QFiLM/QNN seeds go to seed<s>/; FiLM's seed list cannot be extended (joint model selection).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from paper_qip_data import SCENARIOS as PAPER_SCENARIOS  # noqa: E402

SCENARIOS: dict[str, dict] = {'clean': dict(), **PAPER_SCENARIOS}
LABELS = {
    'clean': 'no noise',
    'F0.95_prepared': 'p=0.067 (F=0.95), prepared label',
    'F0.95_clean': 'p=0.067 (F=0.95), intended label',
    'F0.50_prepared': 'p=0.667 (F=0.50), prepared label',
    'F0.50_clean': 'p=0.667 (F=0.50), intended label',
    'F0.95sd_clean': 'p~0.067 (F~0.95), drift',
    'F0.50sd_clean': 'p~0.667 (F~0.50), drift',
    'F0.95_clean_r0.025': 'p=0.067 (F=0.95) + readout',
    'F0.95sd_clean_r0.025_off0.012': 'p~0.067 (F~0.95), drift + readout + offset',
    'drift_prep': 'p~0.067 (F~0.95), drift, prepared label',            # README --custom examples
    'realistic_prep': 'p~0.067 (F~0.95), drift + readout + offset, prepared label',
}
MODELS = {'qfilm16': 'joint_exchange', 'qfilm2': 'joint_exchange_w2', 'qfilm1': 'joint_exchange_w1',
          'qnn': 'two_copy_qnn'}
SPLITS = ['train', 'validation', 'test_state', 'test_context', 'test_joint']
SEEDS = None   # from --seeds
COLORS = {'qfilm16': '#1F5FBF', 'qfilm2': '#5FA8E0', 'qfilm1': '#5FA8E0', 'qnn': '#7B3FA0', 'film': '#B5470F'}
NAMES = {'qfilm16': 'QFiLM, 16 rotations', 'qfilm2': 'QFiLM, 2 rotations', 'qfilm1': 'QFiLM, 1 rotation',
         'qnn': 'Two-copy QNN (no exchange)', 'film': 'Classical FiLM'}


def parse_custom(text):
    name, _, body = text.partition(':')
    keys = {'p': 'p_prep', 'p_sd': 'p_sd', 'sigma': 'readout_sigma', 'offset': 'readout_offset_sd',
            'target': 'target_state'}
    cfg = {}
    for item in filter(None, body.split(',')):
        k, _, v = item.partition('=')
        if k.strip() not in keys:
            sys.exit(f'unknown key {k!r} in --custom; use p, p_sd, sigma, offset, target')
        cfg[keys[k.strip()]] = v.strip() if k.strip() == 'target' else float(v)
    short = {'p_prep': 'p', 'p_sd': 'sd(p)', 'readout_sigma': 'sigma', 'readout_offset_sd': 'offset'}
    parts = [f'{short[k]}={v:.3g}' for k, v in cfg.items() if k in short and v]
    LABELS.setdefault(name.strip(), f"{name.strip()}: " + ', '.join(parts) + f", {cfg.get('target_state', 'prepared')} label")
    return name.strip(), cfg


def run(cmd, out):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'train.log', 'w') as log:
        subprocess.run([str(c) for c in cmd], check=True, stdout=log, stderr=subprocess.STDOUT)


def paths(n):
    return ROOT / f'data/noise_study/n{n}', ROOT / f'results/noise_study/n{n}'


# ---------------------------------------------------------------- step 1: data
def step_data(scenarios, n):
    from qem.scaled_data import make_scaled_benchmark
    ddir, _ = paths(n)
    for name, cfg in scenarios.items():
        dst = ddir / f'{name}.npz'
        if dst.exists():
            print(f'data   {name}: exists')
            continue
        m = make_scaled_benchmark(dst, n, fixed_eval=True, **cfg)
        print(f"data   {name}: {n} states, mean fidelity to intended "
              f"{m['preparation_noise']['mean_fidelity_to_clean']:.4f}", flush=True)


# ---------------------------------------------------------------- step 2: training
def quantum_runs_in(d, key):
    """(run, predictions) per seed from d/metrics.json and d/seed<s>/metrics.json."""
    seen = {}
    for m in [d / 'metrics.json', *sorted(d.glob('seed*/metrics.json'))]:
        if m.exists():
            preds = np.load(m.parent / 'predictions.npz')
            for r in json.loads(m.read_text())['models'][key]['runs']:
                seen.setdefault(r['seed'], (r, preds))
    return [seen[s] for s in sorted(seen) if SEEDS is None or s in SEEDS]


def film_seeds(d):
    m = d / 'metrics.json'
    return {r['seed'] for r in json.loads(m.read_text())['selected_runs']} if m.exists() else set()


def train_job(model, name, n, seeds, threads):
    ddir, rdir = paths(n)
    out = rdir / model / name
    data = ddir / f'{name}.npz'
    want = [int(x) for x in seeds.split(',')]
    if model == 'film':
        have = film_seeds(out)
        if have:
            if set(want) <= have:
                return f'train  {model:7s} {name}: exists (seeds {sorted(have)})'
            return f'train  {model:7s} {name}: SKIPPED, has seeds {sorted(have)}; delete the folder to retrain'
        run([sys.executable, ROOT / 'scripts/train_torch_classical.py', '--data', data, '--output', out,
             '--seeds', seeds], out)
        return f'train  {model:7s} {name}: done'
    have = {r['seed'] for r, _ in quantum_runs_in(out, MODELS[model])} if out.exists() else set()
    missing = [s for s in want if s not in have]
    if not missing:
        return f'train  {model:7s} {name}: exists (seeds {want})'
    jobs = [(out, ','.join(map(str, missing)))] if not have else [(out / f'seed{s}', str(s)) for s in missing]
    for dst, ss in jobs:
        run([sys.executable, ROOT / 'scripts/train_torch.py', '--models', MODELS[model], '--data', data,
             '--output', dst, '--seeds', ss, '--threads', str(threads)], dst)
    return f'train  {model:7s} {name}: trained seeds {missing}'


def step_train(scenarios, n, models, seeds, jobs, threads):
    for name in scenarios:
        if not (paths(n)[0] / f'{name}.npz').exists():
            sys.exit(f'missing data for {name}; run the data step first')
    tasks = [(m, s) for s in scenarios for m in models]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for msg in pool.map(lambda t: train_job(t[0], t[1], n, seeds, threads), tasks):
            print(msg, flush=True)


# ---------------------------------------------------------------- step 3: report
def load_quantum(d, model):
    key = MODELS[model]
    return {r['seed']: {'mse': r['metrics']['test_joint']['mse'],
                        'bell': preds[f'{key}__seed{r["seed"]}__bell__prediction']}
            for r, preds in quantum_runs_in(d, key)}


def load_film(d):
    m = json.loads((d / 'metrics.json').read_text())
    preds = np.load(d / 'predictions.npz')
    return {r['seed']: {'mse': r['metrics']['test_joint']['mse'], 'bell': preds[f"{r['key']}__bell"]}
            for r in m['selected_runs'] if SEEDS is None or r['seed'] in SEEDS}


def gram(data):
    """Gram-6 fit and plug-in (functions of scripts/paper_qip_figures.py)."""
    import matplotlib
    matplotlib.use('Agg')
    from paper_qip_figures import gram_fit, gram_plugin
    return gram_fit(data)['test_joint'], gram_plugin(data)['test_joint']


def fmt(x):
    return '<1e-26' if x < 1e-26 else f'{x:.2g}'


def step_report(scenarios, n, models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ddir, rdir = paths(n)
    numbers, rows = {}, []
    for name in scenarios:
        data = np.load(ddir / f'{name}.npz')
        var = float(data['test_joint_y'].var())
        entry = {'label': LABELS.get(name, name), 'test_joint_var': var, 'relative_mse': {}, 'seeds': {}, 'bell': {}}
        for model in models:
            d = rdir / model / name
            if not (d / 'metrics.json').exists():
                continue
            runs = load_film(d) if model == 'film' else load_quantum(d, model)
            v = np.array([r['mse'] for r in runs.values()]) / var
            entry['relative_mse'][model] = float(v.mean())
            entry['seeds'][model] = {int(s): float(r['mse'] / var) for s, r in runs.items()}
            entry['bell'][model] = np.mean([r['bell'] for r in runs.values()], 0).tolist()
        fit, plug = gram(data)
        entry['relative_mse']['gram6_fit'], entry['relative_mse']['plug_in'] = fit / var, plug / var
        numbers[name] = entry
        rows.append((name, entry))
    cols = [m for m in models] + ['gram6_fit', 'plug_in']
    head = {'qfilm16': 'QFiLM 16', 'qfilm2': 'QFiLM 2', 'qfilm1': 'QFiLM 1', 'qnn': 'Two-copy QNN',
            'film': 'Classical FiLM', 'gram6_fit': 'Gram-6 (fit)',
            'plug_in': 'Plug-in'}
    md = [f'Test MSE / Var(y) on 800 held-out pairs, {n} training states x 5 directions, mean over seeds.', '',
          '| Scenario | ' + ' | '.join(head[c] for c in cols) + ' |', '|---' * (len(cols) + 1) + '|']
    tex = [r'\begin{tabular}{l' + 'l' * len(cols) + r'}\toprule',
           'Scenario & ' + ' & '.join(head[c] for c in cols) + r'\\\midrule']
    for name, e in rows:
        cells = [fmt(e['relative_mse'][c]) if c in e['relative_mse'] else '--' for c in cols]
        md.append(f"| {e['label']} | " + ' | '.join(cells) + ' |')
        tex.append(e['label'].replace('~', r'$\approx$') + ' & ' + ' & '.join(cells) + r'\\')
    tex.append(r'\bottomrule\end{tabular}')
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / 'table_noise.md').write_text('\n'.join(md) + '\n')
    (rdir / 'table_noise.tex').write_text('\n'.join(tex) + '\n')

    t = np.load(ddir / f'{rows[0][0]}.npz')['bell_t']
    exact = (2 * t - 1) ** 2
    k = len(rows)
    nc = 2 if k == 4 else min(3, k)
    nr = -(-k // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.9 * nr), sharex=True, sharey=True, squeeze=False)
    for ax, (name, e) in zip(axes.ravel(), rows):
        ax.plot(t, exact, 'k--', lw=1.2, label=r'exact $(2t-1)^2$', zorder=5)
        for model in models:
            if model in e['bell']:
                b = np.array(e['bell'][model])
                ax.plot(t, b, color=COLORS[model], lw=1.8, label=NAMES[model])
                e.setdefault('bell_mse', {})[model] = float(np.mean((b - exact) ** 2))
        ax.set_title(textwrap.fill(e['label'], 42), fontsize=8.5)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[k:]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel(r'Bell weight $t$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$D_{Z\otimes I}$')
    axes[0, 0].legend(fontsize=7, frameon=False, loc='upper center')
    fig.tight_layout()
    fig.savefig(rdir / 'fig_bell.pdf')
    fig.savefig(rdir / 'fig_bell.png', dpi=160)
    for e in numbers.values():
        e.pop('bell')
    (rdir / 'numbers.json').write_text(json.dumps(numbers, indent=1) + '\n')
    print((rdir / 'table_noise.md').read_text())
    print(f'wrote {rdir.relative_to(ROOT)}/table_noise.md, .tex, fig_bell.pdf, .png, numbers.json')


# ---------------------------------------------------------------- learning curve
def shot_errors(d, model, data, copies, rng):
    """Per-seed test MSE at `copies` copies: QFiLM exact shot variance, FiLM 9-basis tomography."""
    from paper_qip_finite_shot import film_predict, quantum_exact_variance, tomography
    from qem.data import pauli_features
    y = data['test_joint_y']
    if model == 'film':
        import torch
        from qem.torch_classical import TorchFiLM
        m = json.loads((d / 'metrics.json').read_text())
        weights = np.load(d / 'weights.npz')
        est = tomography(rng, pauli_features(data['test_joint_rho']), copies // 9)
        out = {}
        for run in m['selected_runs']:
            if SEEDS is not None and run['seed'] not in SEEDS:
                continue
            f = TorchFiLM(width=run['width'])
            with torch.no_grad():
                f.weights.copy_(torch.tensor(weights[run['key']]))
            out[run['seed']] = float(np.mean((film_predict(f, est[:, 1:], data['test_joint_context']) - y) ** 2))
        return out
    key = MODELS[model]
    out = {}
    for r, preds in quantum_runs_in(d, key):
        pre = f"{key}__seed{r['seed']}__test_joint__"
        w = np.asarray(r['architecture']['fitted_summary']['head_weights'])
        shot = np.mean(quantum_exact_variance(preds[pre + 'heads'], preds[pre + 'correlation'], w, copies // 2))
        out[r['seed']] = float(np.mean((preds[pre + 'prediction'] - y) ** 2) + shot)
    return out


def gram_shot_error(data, copies, rng):
    """Fitted Gram-6 on 9-basis tomography of the model input, copies/9 shots per basis."""
    from paper_qip_finite_shot import gram_fit as fit, gram_predict, tomography
    from qem.data import pauli_features
    est = tomography(rng, pauli_features(data['test_joint_rho']), copies // 9)
    return float(np.mean((gram_predict(fit(data), est, data['test_joint_context']) - data['test_joint_y']) ** 2))


def step_curve(scenarios, sizes, models, copies):
    """Test MSE / Var(y) against training states; solid: `copies` copies, dashed: exact expectations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(20261121)
    out = ROOT / 'results/noise_study'
    names = list(scenarios)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names) + 2.4, 3.6), sharey=True, squeeze=False)
    record = {}
    for ax, name in zip(axes[0], names):
        rec = record.setdefault(name, {'label': LABELS.get(name, name), 'copies': copies, 'sizes': {}})
        curves = {m: {'shots': [], 'exact': []} for m in models + ['gram6_fit']}
        for n in sizes:
            ddir, rdir = paths(n)
            if not (ddir / f'{name}.npz').exists():
                continue
            data = np.load(ddir / f'{name}.npz')
            var = float(data['test_joint_y'].var())
            row = {}
            for m in models:
                d = rdir / m / name
                if not (d / 'metrics.json').exists():
                    continue
                runs = load_film(d) if m == 'film' else load_quantum(d, m)
                ex = [float(r['mse'] / var) for r in runs.values()]
                sh = [v / var for v in shot_errors(d, m, data, copies, rng).values()]
                row[m] = {'exact': ex, 'shots': sh}
                curves[m]['exact'].append((n, ex)); curves[m]['shots'].append((n, sh))
            fit, _ = gram(data)
            gs = gram_shot_error(data, copies, rng) / var
            row['gram6_fit'] = {'exact': [fit / var], 'shots': [gs]}
            curves['gram6_fit']['exact'].append((n, [fit / var])); curves['gram6_fit']['shots'].append((n, [gs]))
            rec['sizes'][n] = row
        style = {m: dict(color=COLORS[m], marker={'qfilm16': 'o', 'qfilm2': 'D', 'qfilm1': 'D', 'qnn': 'v', 'film': 's'}[m],
                         label=NAMES[m]) for m in models}
        style['gram6_fit'] = dict(color='#1A8A6B', marker='^', label='Gram-6 (fitted, on tomography)')
        for m, c in curves.items():
            if not c['shots']:
                continue
            kw = style[m]
            xs = [n for n, _ in c['shots']]
            ax.plot(xs, [np.median(v) for _, v in c['shots']], lw=1.8, ms=5, ls='-', **kw)
            for x, v in c['shots']:
                if len(v) > 1:
                    ax.scatter([x] * len(v), v, s=9, color=kw['color'], alpha=0.4, lw=0)
            if m != 'film':
                ax.plot(xs, [np.median(v) for _, v in c['exact']], lw=1.0, ls='--', color=kw['color'], alpha=0.7)
        ax.axhline(1, color='0.5', ls='--', lw=0.9)
        ax.text(sizes[-1], 1.12, 'mean predictor', fontsize=7.5, color='0.4', ha='right')
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xticks(sizes); ax.set_xticklabels([str(n) for n in sizes])
        ax.set_xlabel('Training states (5 directions each)')
        ax.set_title(textwrap.fill(LABELS.get(name, name), 48), fontsize=8.5)
        ax.grid(alpha=0.3, which='major')
    axes[0, 0].set_ylabel('Test MSE / Var(y)')
    handles, labels = axes[0, -1].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], color='0.3', lw=1.8), Line2D([], [], color='0.3', lw=1.0, ls='--')]
    labels += [f'{copies // 2:,} shots per prediction (median)', 'exact expectations (median)']
    axes[0, -1].legend(handles, labels, fontsize=7.5, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    tag = '_'.join(names) if len(names) == 1 else 'multi'
    fig.savefig(out / f'learning_curve_{tag}.pdf')
    fig.savefig(out / f'learning_curve_{tag}.png', dpi=160)
    (out / f'learning_curve_{tag}.json').write_text(json.dumps(record, indent=1) + '\n')
    print(f'wrote results/noise_study/learning_curve_{tag}.pdf, .png, .json')


# ---------------------------------------------------------------- rotation study
def step_rotations(name, n, ks, seeds, jobs, threads, copies):
    """QFiLM with k rotations in W: exact and finite-shot test MSE and head weights per seed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_qip_finite_shot import quantum_exact_variance
    ddir, rdir = paths(n)
    data = ddir / f'{name}.npz'
    if not data.exists():
        sys.exit(f'missing {data}; run the data step first')
    root = rdir / 'rotations' / name

    def key(k):
        return 'joint_exchange' if k == 16 else f'joint_exchange_w{k}'

    def have(k):
        found = {}
        for m in [root / f'k{k}' / 'metrics.json', *sorted((root / f'k{k}').glob('seed*/metrics.json'))]:
            if m.exists():
                for r in json.loads(m.read_text())['models'][key(k)]['runs']:
                    found.setdefault(r['seed'], (m.parent, r))
        return found

    def job(task):
        k, s = task
        out = root / f'k{k}' / f'seed{s}'
        run([sys.executable, ROOT / 'scripts/train_torch.py', '--models', key(k), '--data', data, '--output', out,
             '--seeds', str(s), '--threads', str(threads)], out)
        return f'rotations k={k} seed={s}: done'

    todo = [(k, int(s)) for k in ks for s in seeds.split(',') if int(s) not in have(k)]
    print(f'rotations: {len(todo)} training runs to do', flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for msg in pool.map(job, todo):
            print(msg, flush=True)

    y = np.load(data)['test_joint_y']
    var = float(y.var())
    rows = {}
    wanted = {int(x) for x in seeds.split(',')}
    for k in ks:
        rows[k] = []
        for s_, (d, r) in sorted(have(k).items()):
            if s_ not in wanted:
                continue
            preds = np.load(d / 'predictions.npz')
            fs, s = r['architecture']['fitted_summary'], r['seed']
            pre = f'{key(k)}__seed{s}__test_joint__'
            heads, corr = preds[pre + 'heads'], preds[pre + 'correlation']
            w = np.asarray(fs['head_weights'])
            exact = float(np.mean((preds[pre + 'prediction'] - y) ** 2))
            shot = float(np.mean(quantum_exact_variance(heads, corr, w, copies // 2)))
            rows[k].append({'seed': s, 'head_weight': float(np.abs(w).max()), 'eta': fs['eta'], 'chi': fs['chi'],
                            'mse_exact_rel': exact / var, 'mse_shots_rel': (exact + shot) / var,
                            'seconds': r.get('elapsed_seconds')})

    lines = [f'QFiLM with k rotations in W, scenario {name}, {n} training states, {len(seeds.split(","))} seeds.',
             f'MSE / Var(y) on 800 held-out pairs; "shots" uses {copies} copies per prediction.', '',
             '| k | parameters | MSE exact (median) | MSE shots (median) | MSE shots (worst seed) | '
             '|head weight| per seed | seeds with |w| > 2 |', '|---|---|---|---|---|---|---|']
    for k in ks:
        r = rows[k]
        ex, sh = [x['mse_exact_rel'] for x in r], [x['mse_shots_rel'] for x in r]
        hw = [x['head_weight'] for x in r]
        lines.append(f'| {k} | {9 + k} | {np.median(ex):.3g} | {np.median(sh):.3g} | {max(sh):.3g} | '
                     + (', '.join(f'{h:.2f}' for h in hw) if len(hw) <= 6 else f'median {np.median(hw):.2f}, max {max(hw):.2f}')
                     + f' | {sum(h > 2 for h in hw)}/{len(hw)} |')
    root.mkdir(parents=True, exist_ok=True)
    stem = f"rotations_k{'-'.join(map(str, ks))}_{len(seeds.split(','))}seeds_{copies}copies"
    (root / f'{stem}.md').write_text('\n'.join(lines) + '\n')
    (root / f'{stem}.json').write_text(json.dumps({'copies': copies, 'var_y': var, 'rows': rows}, indent=1) + '\n')

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    xs = np.arange(len(ks))
    panels = [('mse_exact_rel', 'Test MSE / Var(y)\nexact expectations', True),
              ('mse_shots_rel', f'Test MSE / Var(y)\n{copies:,} copies per prediction', True),
              ('head_weight', '|head weight|', False)]
    for ax, (field, title, logy) in zip(axes, panels):
        for j, k in enumerate(ks):
            v = [x[field] for x in rows[k]]
            ax.scatter(np.full(len(v), j) + np.linspace(-0.12, 0.12, len(v)), v, s=18, color='#1F5FBF', alpha=0.7)
            ax.plot([j - 0.2, j + 0.2], [np.median(v)] * 2, color='k', lw=1.5)
        ax.set_xticks(xs, [str(k) for k in ks]); ax.set_xlabel('rotations k in W')
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale('log')
    fig.suptitle(textwrap.fill(f'{LABELS.get(name, name)}, {n} training states (dots: seeds, bar: median)', 90),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(root / f'{stem}.pdf'); fig.savefig(root / f'{stem}.png', dpi=160)
    print((root / f'{stem}.md').read_text())
    print(f'wrote {root.relative_to(ROOT)}/{stem}.md, .json, .pdf, .png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['data', 'train', 'report', 'curve', 'rotations', 'all'])
    ap.add_argument('--train-states', default='64', help='one size or a list, e.g. 8,16,32,64,128,256')
    ap.add_argument('--scenarios', default=','.join(PAPER_SCENARIOS),
                    help="default: the 8 noisy settings; 'clean' (no noise) is available as a reference")
    ap.add_argument('--custom', action='append', default=[], help='name:p=..,p_sd=..,sigma=..,offset=..,target=..')
    ap.add_argument('--models', default='qfilm16,qfilm2,film')
    ap.add_argument('--seeds', default='17,31,73')
    ap.add_argument('--jobs', type=int, default=1, help='training runs in parallel')
    ap.add_argument('--threads', type=int, default=1, help='Torch threads per quantum run')
    ap.add_argument('--rotations', default='1,2,4,8,16', help='rotation counts k for the rotations step')
    ap.add_argument('--copies', type=int, default=20000,
                    help='copies per prediction (shots = copies/2) for the finite-shot results')
    a = ap.parse_args()
    global SEEDS
    SEEDS = {int(x) for x in a.seeds.split(',')}
    scen = {s: SCENARIOS[s] for s in a.scenarios.split(',') if s}
    for c in a.custom:
        name, cfg = parse_custom(c)
        scen[name] = cfg
    models = a.models.split(',')
    sizes = [int(x) for x in a.train_states.split(',')]
    for n in sizes:
        if a.step in ('data', 'all'):
            step_data(scen, n)
        if a.step in ('train', 'all'):
            step_train(scen, n, models, a.seeds, a.jobs, a.threads)
        # 'all' over several sizes builds the learning curve only, so it does not overwrite a size's noise table
        if a.step == 'report' or (a.step == 'all' and len(sizes) == 1):
            step_report(scen, n, models)
    if a.step == 'rotations':
        for n in sizes:
            for name in scen:
                step_rotations(name, n, [int(k) for k in a.rotations.split(',')], a.seeds, a.jobs, a.threads, a.copies)
        return
    if a.step == 'curve' or (a.step == 'all' and len(sizes) > 1):
        step_curve(scen, sizes, models, a.copies)


if __name__ == '__main__':
    main()
