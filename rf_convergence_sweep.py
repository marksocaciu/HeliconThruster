#!/usr/bin/env python3
"""One-factor mesh / degree / collision-frequency sweep for the RF plasma case.

Run from the HeliconThruster folder with the complex-arithmetic runtime, e.g.

  python3 rf_convergence_sweep.py --out runs/conv-nu20-deg1 \
      --plasma-summary runs/clean-20261022-121857/global-plasma/plasma_summary.json \
      --b-map runs/clean-20261022-121857/magnetic/field_full.npz \
      --ranks 4 --plasma-mm 16 12 8 6 4 3

Every case uses the PEC-box configuration (pml_strength=1e-9), so input power
equals chamber absorption. Sizes are absolute (mm), not scaled with --mesh-scale.
Results are appended to <out>/results.csv, one row per case.
"""
import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='new folder for this sweep')
    p.add_argument('--config', default='config/rf_nagoya.json')
    p.add_argument('--plasma-summary', required=True)
    p.add_argument('--b-map', required=True)
    p.add_argument('--ranks', type=int, default=1, help='MPI ranks for the solve')
    p.add_argument('--degree', type=int, default=None, help='element degree (config value if omitted)')
    p.add_argument('--nu', type=float, default=None, help='collision frequency [Hz] (config value if omitted)')
    p.add_argument('--freq', type=float, default=None, help='frequency [Hz] (config value if omitted)')
    p.add_argument('--plasma-mm', type=float, nargs='+', default=[16, 8, 4], help='plasma mesh sizes to sweep')
    p.add_argument('--antenna-mm', type=float, default=6.0)
    p.add_argument('--gap-mm', type=float, default=3.0)
    p.add_argument('--wall-mm', type=float, default=6.0)
    p.add_argument('--bulk-mm', type=float, default=70.0)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True)  # refuses to reuse an existing folder
    base = json.load(open(a.config))
    base['domain']['pml_strength'] = 1e-9
    if a.degree:
        base['solver']['degree'] = a.degree
    if a.nu:
        base['plasma']['collision_frequency_Hz'] = a.nu
    if a.freq:
        base['frequency_Hz'] = a.freq

    csv_path = out / 'results.csv'
    fields = ['tag', 'plasma_mm', 'degree', 'nu_Hz', 'frequency_Hz', 'dofs',
              'P_abs_W', 'R_ohm', 'X_ohm', 'power_balance_rel_error', 'checks_failed', 'seconds']
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fields).writeheader()

    for pm in a.plasma_mm:
        tag = f'plasma{pm:g}mm'
        c = copy.deepcopy(base)
        c['mesh'].update(bulk_m=a.bulk_mm / 1e3, plasma_m=pm / 1e3, wall_m=a.wall_mm / 1e3,
                         antenna_m=a.antenna_mm / 1e3, gap_m=a.gap_mm / 1e3)
        cfg = out / f'{tag}.json'
        json.dump(c, open(cfg, 'w'), indent=2)
        t0 = time.time()
        prep = out / f'{tag}-prepared'
        sol = out / tag
        r = subprocess.run([sys.executable, '-m', 'helicon.rf.run', '--config', str(cfg), '--case', 'plasma',
                            '--mesh-only', '--mesh-scale', '1', '--plasma-summary', a.plasma_summary,
                            '--b-map', a.b_map, '--out', str(prep)], capture_output=True, text=True)
        if r.returncode:
            print(tag, 'mesh preparation failed:', r.stderr[-300:])
            continue
        cmd = [sys.executable, '-m', 'helicon.rf.solve', '--build', str(prep / 'mesh'), '--out', str(sol)]
        if a.ranks > 1:
            cmd = ['mpiexec', '-n', str(a.ranks)] + cmd + ['--expected-ranks', str(a.ranks)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        summary = sol / 'rf_summary.json'
        if not summary.exists():
            print(tag, 'solve failed:', (r.stderr or r.stdout)[-300:])
            continue
        d = json.load(open(summary))
        z = d['ideal_gap_impedance_ohm']
        row = dict(tag=tag, plasma_mm=pm, degree=c['solver']['degree'],
                   nu_Hz=c['plasma']['collision_frequency_Hz'], frequency_Hz=c['frequency_Hz'],
                   dofs=d['dofs'], P_abs_W=d['material_absorption_W']['PLASMA'], R_ohm=z[0], X_ohm=z[1],
                   power_balance_rel_error=d['power_balance_relative_error'],
                   checks_failed=';'.join(k for k, v in d['numerical_checks'].items() if not v),
                   seconds=round(time.time() - t0))
        with open(csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fields).writerow(row)
        print(f"{tag}: dofs={row['dofs']} P_abs={row['P_abs_W']*1e3:.4f} mW  Z={z[0]:.3f}{z[1]:+.3f}j  "
              f"({row['seconds']} s)", flush=True)
        shutil.rmtree(prep / 'mesh', ignore_errors=True)  # keep disk use down; summaries are kept


if __name__ == '__main__':
    main()
