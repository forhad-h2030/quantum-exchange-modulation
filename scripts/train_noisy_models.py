"""Train QFiLM and classical FiLM on the noisy datasets (noise study, paper Sec. 6.2).

    python scripts/paper_qip_data.py --noisy          # data/werner/<scenario>.npz
    python scripts/train_noisy_models.py                 # all scenarios, both models
    python scripts/train_noisy_models.py --scenarios F0.95_prepared --models quantum --seeds 17

Run with a Python that has torch and pennylane.
Output: results/werner/quantum/<scenario>/, results/werner/film/<scenario>/ (metrics.json, predictions.npz)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from paper_qip_data import SCENARIOS  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', default=','.join(SCENARIOS))
    ap.add_argument('--models', default='quantum,film')
    ap.add_argument('--seeds', default='17,31,73')
    ap.add_argument('--out', default='results/werner')
    ap.add_argument('--redo', action='store_true', help='retrain even if metrics.json exists')
    a = ap.parse_args()
    for s in a.scenarios.split(','):
        data = ROOT / f'data/werner/{s}.npz'
        if not data.exists():
            sys.exit(f'missing {data}; run scripts/paper_qip_data.py --noisy first')
        for model in a.models.split(','):
            out = ROOT / a.out / model / s
            if (out / 'metrics.json').exists() and not a.redo:
                print('done   ', model, s)
                continue
            if model == 'quantum':
                cmd = [sys.executable, ROOT / 'scripts/train_torch.py', '--models', 'joint_exchange']
            else:
                cmd = [sys.executable, ROOT / 'scripts/train_torch_classical.py']
            cmd += ['--data', data, '--output', out, '--seeds', a.seeds]
            print('train  ', model, s, flush=True)
            subprocess.run([str(c) for c in cmd], check=True)
