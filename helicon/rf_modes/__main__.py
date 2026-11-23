"""Run the independent Fourier-mode uniform-cylinder RF benchmark."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from ..config import ROOT
from ..records import digest, utc_now, write_json
from ..rf.materials import cold_epsilon
from .analysis import compare_fields, cylinder_benchmark, relative_error
from .meridian import solve_mode
from .slices import save_slices


def validate(c):
    positive = ('frequency_Hz', 'electron_density_m3', 'collision_frequency_s', 'radius_m', 'length_m')
    if set(c) != {*positive, 'z_start_m', 'Bz_T', 'nr', 'nz', 'modes'}:
        raise ValueError('Expected the keys in config/rf_modes_uniform.json')
    for name in (*positive, 'Bz_T', 'z_start_m'):
        if type(c[name]) not in (float, int) or not np.isfinite(c[name]) or (name in positive and c[name] <= 0):
            raise ValueError(f'Invalid {name}')
    for name in ('nr', 'nz'):
        if type(c[name]) is not int or c[name] < (3 if name == 'nz' else 2):
            raise ValueError(f'Invalid {name}')
    modes = {}
    for entry in c['modes']:
        if set(entry) != {'m', 'amplitude_V_m'}:
            raise ValueError('Each mode needs m and its complex amplitude_V_m pair')
        m = entry['m']
        parts = entry['amplitude_V_m']
        if (type(m) is not int or m in modes or not isinstance(parts, list) or len(parts) != 2
                or not all(type(v) in (int, float) and np.isfinite(v) for v in parts)):
            raise ValueError('Invalid mode number, duplicate mode, or invalid drive')
        modes[m] = complex(*parts)
        if modes[m] == 0:
            raise ValueError('Omit undriven modes')
    if not modes:
        raise ValueError('At least one mode is required')
    return modes


def execute(config, out, compare_3d=False, sectors=8, phi_deg=()):
    drives = validate(config)
    if compare_3d and (not set(drives) <= {-1, 1} or sectors < 3):
        raise ValueError('3D plane-wave comparison requires modes ±1 and at least three sectors')
    if not all(np.isfinite(phi_deg)):
        raise ValueError('Slice angles must be finite')
    epsilon = cold_epsilon(config['frequency_Hz'], config['electron_density_m3'],
                           config['collision_frequency_s'], [0, 0, config['Bz_T']])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = [*Path(__file__).parent.glob('*.py'), ROOT/'helicon/rf/materials.py', ROOT/'helicon/argon.py']
    record = dict(status='running', started_utc=utc_now(), model='rf_modes_uniform_cylinder_v1',
                  config=config, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                  numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(),
                  field_convention='peak SI cylindrical (r,phi,z), exp(+i m phi - i omega t)',
                  drive='prescribed transverse end field at z_start, amplitude [real,imag] V/m',
                  boundaries='PEC end, Ez=Hz=0 side, regular axis', modes={},
                  limits=dict(free_relative_residual=1e-8, reaction_balance=1e-8,
                              inlet_flux_relative_error=0.05, analytic_relative_error=0.02,
                              comparison_3d_relative_error=0.02))
    write_json(out/'run.json', record)
    start = time.monotonic()
    try:
        modal = {}
        limits = record['limits']
        checks = {}
        for m, amplitude in drives.items():
            tick = time.monotonic()
            fields, stats = solve_mode(config, epsilon, m, amplitude, config['nr'], config['nz'])
            modal[m] = fields
            pin = stats['input_reaction_W']
            stats['power_balance_relative_error'] = relative_error(stats['absorption_W'], pin)
            stats['inlet_flux_relative_error'] = relative_error(stats['input_poynting_W'], pin)
            passed = dict(passive=bool(pin > 0 and np.min(fields['cell_absorbed_W']) >= 0),
                          residual=stats['free_relative_residual'] < limits['free_relative_residual'],
                          balance=stats['power_balance_relative_error'] < limits['reaction_balance'],
                          inlet_flux=stats['inlet_flux_relative_error'] < limits['inlet_flux_relative_error'])
            if abs(m) == 1:
                theory, profile = cylinder_benchmark(config, m, amplitude, fields)
                theory['input_power_relative_error'] = relative_error(pin, theory['analytic_input_W'])
                stats['analytical_benchmark'] = theory
                for key in ('k_relative_error', 'axial_field_relative_error', 'input_power_relative_error'):
                    passed[key] = theory[key] < limits['analytic_relative_error']
                fields.update(profile)
                signal = profile['Er_numeric_V_m']
                dz = config['length_m']/config['nz']
                fields['kz_fft_per_m'] = np.fft.fftshift(2*np.pi*np.fft.fftfreq(len(signal), dz))
                fields['Er_fft_V_m'] = np.fft.fftshift(np.fft.fft(signal*np.hanning(len(signal))))/len(signal)
            filename = f'mode_{m:+d}.npz'
            np.savez_compressed(out/filename, **fields)
            stats.update(seconds=time.monotonic()-tick, checks=passed, file=filename)
            record['modes'][str(m)] = stats
            checks[str(m)] = all(passed.values())
            write_json(out/'run.json', record)
        watts = sum(f['cell_absorbed_W'] for f in modal.values())
        template = next(iter(modal.values()))
        r, z = template['r_edges_m'], template['z_edges_m']
        volumes = np.pi*np.diff(z)[:, None]*np.diff(r**2)[None, :]
        np.savez_compressed(out/'heating_rz.npz', r_faces_m=r, z_faces_m=z, cell_absorbed_W=watts,
                            cell_volume_m3=volumes, q_abs_W_m3=watts/volumes)
        record['total_absorbed_W'] = float(watts.sum())
        record['slices'] = save_slices(modal, epsilon, config['frequency_Hz'], phi_deg, out)
        if compare_3d:
            from .reference3d import solve_reference
            tick = time.monotonic()
            ref, stats = solve_reference(config, epsilon, drives, config['nz'], sectors)
            differences = compare_fields(modal, ref, stats, record['total_absorbed_W'], config)
            np.savez_compressed(out/'reference_3d.npz', **ref)
            stats.update(seconds=time.monotonic()-tick, sectors=sectors, differences=differences)
            record['reference_3d'] = stats
            checks['3d'] = (stats['free_relative_residual'] < limits['free_relative_residual']
                            and relative_error(stats['input_reaction_W'], stats['absorption_W']) < limits['reaction_balance']
                            and max(differences.values()) < limits['comparison_3d_relative_error'])
        record['checks'] = checks
        record['status'] = 'completed' if all(checks.values()) else 'checks_failed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(seconds=time.monotonic()-start, finished_utc=utc_now())
        write_json(out/'run.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/rf_modes_uniform.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compare-3d', action='store_true', help='Solve an independent prismatic 3D reference for ±1')
    parser.add_argument('--sectors', type=int, default=8, help='Angular sectors of the reference mesh')
    parser.add_argument('--phi-deg', type=float, nargs='+', default=[], help='Save reconstructed angular slices')
    args = parser.parse_args()
    record = execute(json.loads(args.config.read_text()), args.out, args.compare_3d, args.sectors, args.phi_deg)
    print(f"{record['status']} | {record['total_absorbed_W']:.8g} W | {args.out/'run.json'}")
    return 0 if record['status'] == 'completed' else 4


if __name__ == '__main__':
    raise SystemExit(main())
