"""Generate every dataset used in the QIP paper.

    python3 scripts/paper_qip_data.py            # clean and noisy data
    python3 scripts/paper_qip_data.py --noisy    # noisy data only

Clean:  data/scaled_fixed/benchmark_n{8..1000}.npz   (learning curve)
Noisy:  data/werner/<scenario>.npz, 400 training states   (noise study)
p = 0.0667 and 0.667 correspond to pure-state fidelity 0.95 and 0.50.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qem.scaled_data import make_scaled_benchmark  # noqa: E402

SIZES = [8, 16, 32, 64, 128, 256, 400, 1000]
P95, P50 = 0.0666667, 0.6666667
SCENARIOS: dict[str, dict] = {
    'F0.95_prepared': dict(p_prep=P95, target_state='prepared'),
    'F0.95_clean': dict(p_prep=P95, target_state='clean'),
    'F0.50_prepared': dict(p_prep=P50, target_state='prepared'),
    'F0.50_clean': dict(p_prep=P50, target_state='clean'),
    'F0.95sd_clean': dict(p_prep=P95, p_sd=0.03, target_state='clean'),
    'F0.50sd_clean': dict(p_prep=P50, p_sd=0.1, target_state='clean'),
    'F0.95_clean_r0.025': dict(p_prep=P95, readout_sigma=0.025, target_state='clean'),
    'F0.95sd_clean_r0.025_off0.012': dict(p_prep=P95, p_sd=0.03, readout_sigma=0.025,
                                          readout_offset_sd=0.012, target_state='clean'),
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--noisy', action='store_true', help='noisy datasets only')
    ap.add_argument('--out', default='data', help='output folder')
    a = ap.parse_args()
    out = Path(a.out)
    if not a.noisy:
        for n in SIZES:
            make_scaled_benchmark(out / f'scaled_fixed/benchmark_n{n}.npz', n, fixed_eval=True)
            print('clean', n)
    for name, noise in SCENARIOS.items():
        m = make_scaled_benchmark(out / f'werner/{name}.npz', 400, fixed_eval=True, **noise)
        print(f"{name:<32} mean fidelity {m['preparation_noise']['mean_fidelity_to_clean']:.4f}")
