"""Train QFiLM and classical FiLM on the clean data at each training size (learning curve, finite shots).

    python scripts/paper_qip_data.py                  # data/scaled_fixed/benchmark_n<n>.npz
    python scripts/train_clean_models.py              # all sizes, both models
    python scripts/train_clean_models.py --sizes 16 --models quantum --seeds 17

Run with a Python that has torch and pennylane. FiLM is also trained at n = 1000.
Output: results/clean/quantum/n<n>/, results/clean/film/n<n>/ (metrics.json, predictions.npz)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = [8, 16, 32, 64, 128, 256, 400]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default=','.join(map(str, SIZES + [1000])))
    ap.add_argument('--models', default='quantum,film')
    ap.add_argument('--seeds', default='17,31,73')
    ap.add_argument('--out', default='results/clean')
    ap.add_argument('--redo', action='store_true', help='retrain even if metrics.json exists')
    a = ap.parse_args()
    for n in map(int, a.sizes.split(',')):
        data = ROOT / f'data/scaled_fixed/benchmark_n{n}.npz'
        if not data.exists():
            sys.exit(f'missing {data}; run scripts/paper_qip_data.py first')
        for model in a.models.split(','):
            if model == 'quantum' and n not in SIZES:
                continue  # QFiLM is not trained at n = 1000
            out = ROOT / a.out / model / f'n{n}'
            if (out / 'metrics.json').exists() and not a.redo:
                print('done   ', model, n)
                continue
            if model == 'quantum':
                cmd = [sys.executable, ROOT / 'scripts/train_torch.py', '--models', 'joint_exchange']
            else:
                cmd = [sys.executable, ROOT / 'scripts/train_torch_classical.py']
            cmd += ['--data', data, '--output', out, '--seeds', a.seeds]
            print('train  ', model, n, flush=True)
            subprocess.run([str(c) for c in cmd], check=True)
