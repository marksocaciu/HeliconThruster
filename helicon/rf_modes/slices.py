"""Reconstruct azimuthal slices from a completed modal run."""
import argparse
from pathlib import Path
import numpy as np

from ..records import read_json, write_json
from ..rf.materials import absorbed_density, cold_epsilon
from .analysis import reconstruct


def save_slices(modal, epsilon, frequency, angles, destination):
    grid = next(iter(modal.values()))
    r, z = grid['r_edges_m'], grid['z_edges_m']
    records = []
    for j, phi in enumerate(angles):
        e, b = reconstruct(modal, np.deg2rad(phi))
        filename = f'slice_{j:03d}.npz'
        np.savez_compressed(destination/filename, phi_deg=phi, r_m=(r[1:]+r[:-1])/2,
                            z_m=(z[1:]+z[:-1])/2, E_cyl_peak_V_m=e, B_cyl_peak_T=b,
                            q_local_W_m3=absorbed_density(e, epsilon, frequency))
        records.append(dict(file=filename, phi_deg=phi))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phi-deg', type=float, nargs='+', required=True)
    args = parser.parse_args()
    run = read_json(args.run/'run.json')
    if run.get('model') != 'rf_modes_uniform_cylinder_v1' or run['status'] != 'completed':
        parser.error('Supply a completed rf_modes uniform-cylinder run')
    if not np.isfinite(args.phi_deg).all():
        parser.error('Slice angles must be finite')
    c = run['config']
    epsilon = cold_epsilon(c['frequency_Hz'], c['electron_density_m3'],
                           c['collision_frequency_s'], [0, 0, c['Bz_T']])
    modal = {}
    for m, record in run['modes'].items():
        with np.load(args.run/record['file'], allow_pickle=False) as data:
            modal[int(m)] = {key: data[key] for key in ('r_edges_m', 'z_edges_m',
                                                        'E_cyl_peak_V_m', 'B_cyl_peak_T')}
    args.out.mkdir(parents=True, exist_ok=False)
    slices = save_slices(modal, epsilon, c['frequency_Hz'], args.phi_deg, args.out)
    write_json(args.out/'slices.json', dict(source_run=str(args.run.resolve()), slices=slices))


if __name__ == '__main__':
    main()
