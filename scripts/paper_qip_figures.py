"""Figures 4, 5 and the Bell figure, plus every quoted number.

Input:  data/scaled_fixed/, data/werner/, results/quantum_scaled_fixed*/, results/film_scaled_fixed/,
        results/werner/
Output: paper/qip/fig_noise.pdf, table_noise.tex, table_curve.tex, results/paper_qip/numbers.json;
copies the realistic-noise learning curve and direction map into paper/qip/ (fig_learning_curve.pdf,
fig_direction_map.pdf). Those two come from scripts/noise_study.py and scripts/noisy_direction_map.py.
The earlier noise-free learning curve is kept in learning_curve() but no longer used by the paper.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qem.data import gram_features  # noqa: E402

PAPER = ROOT / 'paper/qip'
OUT = ROOT / 'results/paper_qip'
QUANTUM, CLASSICAL, GRAM, INK, MUTED, GRID = '#1F5FBF', '#B5470F', '#1A8A6B', '#14213D', '#6B7280', '#E5E7EB'
SIZES = [8, 16, 32, 64, 128, 256, 400]
SPLITS = ['train', 'validation', 'test_state', 'test_context', 'test_joint']
SCENARIOS = [
    ('F0.95_prepared', r'$p=0.067$ ($F_{\rm pure}=0.95$), prepared', 'Fixed preparation noise'),
    ('F0.95_clean', r'$p=0.067$ ($F_{\rm pure}=0.95$), intended', 'Fixed preparation noise'),
    ('F0.50_prepared', r'$p=0.667$ ($F_{\rm pure}=0.50$), prepared', 'Fixed preparation noise'),
    ('F0.50_clean', r'$p=0.667$ ($F_{\rm pure}=0.50$), intended', 'Fixed preparation noise'),
    ('F0.95sd_clean', r'$p\approx0.067$ ($F_{\rm pure}\approx0.95$), drift', 'Drifting preparation'),
    ('F0.50sd_clean', r'$p\approx0.667$ ($F_{\rm pure}\approx0.50$), drift', 'Drifting preparation'),
    ('F0.95_clean_r0.025', r'$p=0.067$ ($F_{\rm pure}=0.95$) + readout', 'With EPR-scale readout noise'),
    ('F0.95sd_clean_r0.025_off0.012', r'$p\approx0.067$ ($F_{\rm pure}\approx0.95$), drift + readout + offset', 'With EPR-scale readout noise'),
]

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'font.family': 'sans-serif', 'axes.titlesize': 9.5})


# ---------------------------------------------------------------- loaders
def quantum_runs(dirs):
    """{seed: {split: mse, 'pred': {split: array}}} merged over result directories."""
    out = {}
    for d in dirs:
        m, p = d / 'metrics.json', d / 'predictions.npz'
        if not m.exists():
            continue
        preds = np.load(p)
        for run in json.loads(m.read_text())['models']['joint_exchange']['runs']:
            s = run['seed']
            out[s] = {k: run['metrics'][k]['mse'] for k in SPLITS}
            out[s]['pred'] = {k: preds[f'joint_exchange__seed{s}__{k}__prediction'] for k in SPLITS + ['bell']}
    return out


def film_runs(d):
    m = json.loads((d / 'metrics.json').read_text())
    preds = np.load(d / 'predictions.npz')
    out = {}
    for run in m['selected_runs']:
        out[run['seed']] = {k: run['metrics'][k]['mse'] for k in SPLITS}
        out[run['seed']]['pred'] = {k: preds[f"{run['key']}__{k}"] for k in SPLITS + ['bell']}
    out['_config'] = m['selected_config']
    return out


def gram_fit(data):
    """Least-squares fit of y on the 6 Gram features of the model's input state, plus intercept."""
    def X(split):
        g = gram_features(data[f'{split}_rho'], data[f'{split}_context'])
        return np.column_stack([g, np.ones(len(g))])
    coef, *_ = np.linalg.lstsq(X('train'), data['train_y'], rcond=None)
    res = {k: float(np.mean((X(k) @ coef - data[f'{k}_y']) ** 2)) for k in SPLITS}
    res['coef'] = coef.tolist()
    return res


def gram_plugin(data):
    """Untrained analytic identity applied to the model's (possibly noisy) input state."""
    return {k: float(np.mean((gram_features(data[f'{k}_rho'], data[f'{k}_context']).sum(1) - data[f'{k}_y']) ** 2))
            for k in SPLITS}


def summarize(runs, split):
    v = np.array([runs[s][split] for s in runs if s != '_config'])
    return {'seeds': [int(s) for s in runs if s != '_config'], 'values': v.tolist(), 'mean': float(v.mean()),
            'geomean': float(np.exp(np.log(v).mean())), 'min': float(v.min()), 'max': float(v.max()),
            'sd': float(v.std())}


def style(ax):
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------- learning curve
def learning_curve(numbers):
    rows = {}
    for n in SIZES:
        q = quantum_runs([ROOT / f'results/quantum_scaled_fixed/n{n}', ROOT / f'results/quantum_scaled_fixed_s31_73/n{n}'])
        f = film_runs(ROOT / f'results/film_scaled_fixed/n{n}')
        data = np.load(ROOT / f'data/scaled_fixed/benchmark_n{n}.npz')
        rows[n] = {'quantum': {k: summarize(q, k) for k in SPLITS}, 'film': {k: summarize(f, k) for k in SPLITS},
                   'film_config': f['_config'], 'gram_fit': gram_fit(data),
                   'constant': float(np.mean((data['test_joint_y'] - data['train_y'].mean()) ** 2)),
                   'test_joint_var': float(data['test_joint_y'].var())}
    f1000 = film_runs(ROOT / 'results/film_scaled_fixed/n1000')
    rows[1000] = {'film': {k: summarize(f1000, k) for k in SPLITS}}
    numbers['learning_curve'] = {str(k): v for k, v in rows.items()}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, split, title in zip(axes, ['test_state', 'test_joint'], ['New states, training axes', 'New states, held-out axes']):
        for key, color, marker, label, sizes in [('quantum', QUANTUM, 'o', 'QFiLM (exchange)', SIZES),
                                                 ('film', CLASSICAL, 's', 'Classical FiLM', SIZES + [1000])]:
            x, g = [], []
            for n in sizes:
                s = rows[n][key][split]
                ax.scatter([n] * len(s['values']), s['values'], s=12, color=color, marker=marker, alpha=0.45,
                           edgecolor='none', zorder=3)
                x.append(n); g.append(s['geomean'])
            ax.plot(x, g, color=color, lw=2, marker=marker, ms=5.5, mec='white', mew=0.8, zorder=4, label=label)
        ax.axhline(rows[400]['constant'], color=MUTED, ls='--', lw=1, zorder=2)
        ax.text(9, rows[400]['constant'] * 0.012, 'training-mean\nconstant (dashed)', color=MUTED, fontsize=7.5)
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024]); ax.set_xticklabels(['8', '16', '32', '64', '128', '256', '512', '1024'])
        ax.set_xlabel('Training states (5 directions each)')
        ax.set_title(title)
        style(ax)
    axes[0].set_ylabel('Test MSE')
    axes[0].set_ylim(1e-13, 0.4)
    axes[0].legend(frameon=False, loc='center left', fontsize=8)
    fig.tight_layout()
    fig.savefig(PAPER / 'fig_learning_curve.pdf')
    plt.close(fig)


# ---------------------------------------------------------------- noise robustness
def noise_source(key):
    """400-state noise runs: scripts/noise_study.py --train-states 400 if present, else the original results/werner."""
    ns = ROOT / f'data/noise_study/n400/{key}.npz'
    if ns.exists() and (ROOT / f'results/noise_study/n400/qfilm16/{key}/metrics.json').exists():
        r = ROOT / 'results/noise_study/n400'
        return ns, [r / f'qfilm16/{key}'], r / f'film/{key}'
    return (ROOT / f'data/werner/{key}.npz',
            [ROOT / f'results/werner/quantum/{key}', ROOT / f'results/werner/quantum_s31_73/{key}'],
            ROOT / f'results/werner/film/{key}')


def noise(numbers):
    rows = []
    numbers['noise'] = {}
    for key, label, group in SCENARIOS:
        dpath, qdirs, fdir = noise_source(key)
        data = np.load(dpath)
        var = float(data['test_joint_y'].var())
        q = quantum_runs(qdirs)
        f = film_runs(fdir)
        gf, gp = gram_fit(data), gram_plugin(data)
        manifest = json.loads(dpath.with_suffix('.json').read_text())
        numbers['noise'][key] = {
            'label': label, 'group': group, 'test_joint_var': var,
            'p_prep': manifest['preparation_noise']['p_prep'], 'p_sd': manifest['preparation_noise']['p_sd'],
            'pure_state_fidelity': manifest['pure_state_fidelity'],
            'actual_fidelity': {k: manifest['preparation_noise'][k] for k in ('mean_fidelity_to_clean','std_fidelity_to_clean','min_fidelity_to_clean','max_fidelity_to_clean')},
            'readout_sigma': manifest['readout_noise']['sigma'], 'readout_offset_sd': manifest['readout_noise']['offset_sd'],
            'target_state': manifest['target_state'],
            'quantum': {k: summarize(q, k) for k in SPLITS}, 'film': {k: summarize(f, k) for k in SPLITS},
            'gram_fit': gf, 'gram_plugin': gp}
        rows.append((label, group, np.array(summarize(q, 'test_joint')['values']) / var,
                     np.array(summarize(f, 'test_joint')['values']) / var, gf['test_joint'] / var))

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    y, ypos, starts, prev = 0, [], [], None
    for _, group, *_ in rows:
        if group != prev:
            y += 0.8 if prev is not None else 0
            starts.append((y, group)); prev = group
        ypos.append(y); y += 1
    floor = 1e-13
    ax.axvspan(floor, 1e-6, color='#E4EEFF', alpha=0.6, lw=0, zorder=0)
    ax.axvline(1.0, color=MUTED, ls='--', lw=1, zorder=1)
    ax.text(2.5e-13, ypos[-1] + 0.9, 'numerically exact', color=MUTED, fontsize=7.5, va='center')
    ax.text(1.12, ypos[-1] + 0.9, 'mean predictor', color=MUTED, fontsize=7.5, va='center')
    for yy, (_, _, q, f, g) in zip(ypos, rows):
        for vals, color, marker, dy in [(q, QUANTUM, 'o', -0.2), (f, CLASSICAL, 's', 0.0)]:
            ax.plot([vals.min(), vals.max()], [yy + dy] * 2, color=color, lw=2, alpha=0.35, zorder=2)
            ax.scatter(vals, [yy + dy] * len(vals), s=16, color=color, marker=marker, edgecolor='white', lw=0.6, zorder=3)
            ax.scatter([np.exp(np.log(vals).mean())], [yy + dy], s=55, color=color, marker=marker, edgecolor='white',
                       lw=1.0, zorder=4)
        ax.scatter([max(g, floor * 1.5)], [yy + 0.2], s=50, color=GRAM, marker='^', edgecolor='white', lw=1.0, zorder=4)
        if g < floor:
            ax.annotate('', xy=(floor, yy + 0.2), xytext=(floor * 6, yy + 0.2),
                        arrowprops=dict(arrowstyle='->', color=GRAM, lw=1))
    ax.set_xscale('log'); ax.set_xlim(floor, 6)
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(ypos[-1] + 1.3, -1.0)
    for y0, group in starts:
        ax.text(-0.02, y0 - 0.62, group, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8,
                fontweight='bold', color=INK)
    ax.set_xlabel(r'Relative joint-test MSE, $\mathrm{MSE}/\mathrm{Var}(y)$')
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0); ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker='o', ls='', color=QUANTUM, ms=7, label='QFiLM (exchange)'),
               Line2D([], [], marker='s', ls='', color=CLASSICAL, ms=7, label='Classical FiLM'),
               Line2D([], [], marker='^', ls='', color=GRAM, ms=7, label='Fitted Gram-6 (physics-informed)')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PAPER / 'fig_noise.pdf')
    plt.close(fig)


# ---------------------------------------------------------------- Bell family
def bell(numbers):
    """Fig. 9 is drawn by scripts/noisy_bell.py --paper (noisy Bell inputs); only its numbers are collected here."""
    f = ROOT / 'results/figures/noisy_bell_n400.json'
    if not f.exists():
        print('Fig. 9: run python scripts/noisy_bell.py --paper')
        return
    numbers['bell'] = json.loads(f.read_text())


def sci(x, digits=2):
    if x == 0:
        return '$0$'
    e = int(np.floor(np.log10(abs(x))))
    if -2 <= e <= 0:
        return f'${x:.{digits + max(0, -e - 1)}f}$' if e < 0 else f'${x:.{digits}f}$'
    m = x / 10 ** e
    return f'${m:.{digits - 1}f}\\times10^{{{e}}}$'


CURVE_JSON = ROOT / 'results/noise_study/learning_curve_F0.95sd_clean_r0.025_off0.012.json'


def noisy_curve(numbers):
    """Table 1 and Fig. 3 from the realistic-noise learning curve of scripts/noise_study.py."""
    rec = next(iter(json.loads(CURVE_JSON.read_text()).values()))
    numbers['learning_curve_noisy'] = rec
    cols = [('qfilm16', 'QFiLM, 16'), ('qfilm2', 'QFiLM, 2'), ('qnn', 'Two-copy QNN'), ('film', 'Classical FiLM'),
            ('gram6_fit', 'Gram-6')]
    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rr' + 'l' * len(cols) + r'@{}}\toprule',
             'States & Pairs & ' + ' & '.join(c for _, c in cols) + r'\\\midrule']
    for n, row in sorted(rec['sizes'].items(), key=lambda kv: int(kv[0])):
        cells = [sci(float(np.median(row[k]['shots']))) if k in row else '--' for k, _ in cols]
        lines.append(f"{n} & {5 * int(n)} & " + ' & '.join(cells) + r'\\')
    lines.append(r'\bottomrule\end{tabular*}')
    (PAPER / 'table_curve.tex').write_text('\n'.join(lines) + '\n')
    for src, dst in [(CURVE_JSON.with_suffix('.pdf'), 'fig_learning_curve.pdf'),
                     (ROOT / 'results/figures/noisy_direction_map_n64.pdf', 'fig_direction_map.pdf')]:
        if src.exists():
            shutil.copyfile(src, PAPER / dst)
        else:
            print(f'missing {src.relative_to(ROOT)}; see README "Reproduce the paper"')


def tables(numbers):
    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lllll@{}}\toprule',
             r'Scenario & QFiLM & Classical FiLM & Fitted Gram-6 & Plug-in\\\midrule']
    group = None
    for key, label, grp in SCENARIOS:
        r = numbers['noise'][key]
        if grp != group:
            if group is not None:
                lines.append(r'\addlinespace')
            lines.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{grp}}}}}\\\\')
            group = grp
        v = r['test_joint_var']
        values = [r['quantum']['test_joint']['mean'] / v, r['film']['test_joint']['mean'] / v,
                  r['gram_fit']['test_joint'] / v, r['gram_plugin']['test_joint'] / v]
        cells = [r'$<10^{-26}$' if x < 1e-26 else sci(x) for x in values]
        lines.append(f"{label} & " + ' & '.join(cells) + r'\\')
    lines.append(r'\bottomrule\end{tabular*}')
    (PAPER / 'table_noise.tex').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {}
    noisy_curve(numbers)
    noise(numbers)
    bell(numbers)
    tables(numbers)
    (OUT / 'numbers.json').write_text(json.dumps(numbers, indent=1) + '\n')
    print('wrote figures to', PAPER, 'and numbers to', OUT / 'numbers.json')


if __name__ == '__main__':
    main()
