"""Panel (d) of Fig. 1: predicted vs exact response under realistic noise (drift + readout + offset,
intended label), 64 training states, seed 17, exact expectations; runs from scripts/noise_study.py.

Output: paper/qip/fig_overview_d.dat (columns: exact, exchange, film)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from paper_qip_figures import film_runs, quantum_runs  # noqa: E402

N, SEED, SPLIT = 64, 17, 'test_joint'
SCEN = 'F0.95sd_clean_r0.025_off0.012'


def main():
    y = np.load(ROOT / f'data/noise_study/n{N}/{SCEN}.npz')[f'{SPLIT}_y']
    q = quantum_runs([ROOT / f'results/noise_study/n{N}/qfilm16/{SCEN}'])[SEED]
    f = film_runs(ROOT / f'results/noise_study/n{N}/film/{SCEN}')[SEED]
    np.savetxt(ROOT / 'paper/qip/fig_overview_d.dat', np.column_stack([y, q['pred'][SPLIT], f['pred'][SPLIT]]),
               fmt='%.6f', header='exact exchange film', comments='')
    v = y.var()
    print(f'{SPLIT}: {len(y)} points, MSE exchange {q[SPLIT]:.2e} (rel {q[SPLIT] / v:.3f}), '
          f'FiLM {f[SPLIT]:.3f} (rel {f[SPLIT] / v:.2f})')


if __name__ == '__main__':
    main()
