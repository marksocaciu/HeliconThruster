"""Reconstruct azimuthal slices from a completed modal run."""
import argparse
from pathlib import Path
import numpy as np

from ..records import digest, read_json, write_json
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
    if run['status'] != 'completed' or not np.isfinite(args.phi_deg).all():
        parser.error('Supply a completed modal run and finite slice angles')
    if run.get('model') == 'nagoya_prescribed_closed_v1':
        c = run['rf_inputs']
        files = [(row['m'], row['file']) for row in run['modes']]
    elif run.get('model') == 'rf_modes_uniform_cylinder_v1':
        c = run['config']
        files = [(int(m), row['file']) for m, row in run['modes'].items()]
    else:
        parser.error('Unrecognized RF modal model')
    modal = {}
    for m, filename in files:
        with np.load(args.run/filename, allow_pickle=False) as data:
            modal[int(m)] = {key: data[key] for key in ('r_edges_m', 'z_edges_m',
                                                        'E_cyl_peak_V_m', 'B_cyl_peak_T')}
    if run.get('material_case') == 'rf_input_cold_plasma_with_dielectric_wall':
        material = run['material_from_rf']
        path = args.run/material['file']
        if digest(path) != material['sha256']:
            raise ValueError('Saved modal material hash mismatch')
        grid = next(iter(modal.values()))
        with np.load(path, allow_pickle=False) as data:
            epsilon = data['epsilon_r_cyl']
            if (not np.array_equal(data['r_faces_m'], grid['r_edges_m'])
                    or not np.array_equal(data['z_faces_m'], grid['z_edges_m'])
                    or epsilon.shape != (*grid['E_cyl_peak_V_m'].shape[:2], 3, 3)):
                raise ValueError('Saved modal material grid differs from field grid')
    elif run['model'] == 'nagoya_prescribed_closed_v1':
        from .nagoya import setup
        _, _, epsilon, _ = setup(c, run['radial_cells'], run['outer_radius_m'], run['current_peak_A'])
    else:
        epsilon = cold_epsilon(c['frequency_Hz'], c['electron_density_m3'],
                               c['collision_frequency_s'], [0, 0, c['Bz_T']])
    args.out.mkdir(parents=True, exist_ok=False)
    slices = save_slices(modal, epsilon, c['frequency_Hz'], args.phi_deg, args.out)
    write_json(args.out/'slices.json', dict(source_run=str(args.run.resolve()), slices=slices))


if __name__ == '__main__':
    main()
