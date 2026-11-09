"""Double radial resolution at fixed prescribed heating from an accepted state."""

import argparse
from pathlib import Path

import numpy as np

from . import spatial
from .records import digest, read_json, write_json
from .spatial_budget import reconstruct
from .spatial_seed import interpolate


def frozen_heating(source, target, heating):
    if not (np.array_equal(target.rf[::2], source.rf)
            and np.array_equal(target.zf, source.zf)):
        raise ValueError('Expected exactly two radial children per source cell')
    refined = np.repeat(heating, 2, axis=1)
    child_power = (refined * target.volumes).reshape(*source.shape, 2).sum(axis=2)
    if not np.allclose(child_power, heating * source.volumes, rtol=1e-12, atol=0):
        raise ValueError('Refinement failed to preserve parent-cell heating power')
    return refined


def budget(grid, data):
    boundary = data['boundary_ions']
    exit_flow = boundary[grid.boundary_kind == 2]
    # Integrate overlaps assuming uniform flux density on each finite-volume face.
    annuli = grid.rf**2
    outer_fraction = np.diff(np.maximum(annuli, (grid.rf[-1]/2)**2))/np.diff(annuli)
    axial_fraction = np.diff(np.maximum(grid.zf, grid.zf[-1] - .045))/grid.dz
    production = data['source'].reshape(grid.shape).sum(axis=1)
    return {
        'exit_ions_s': float(exit_flow.sum()),
        'outer_half_exit_ions_s': float(exit_flow @ outer_fraction),
        'downstream_45mm_ion_production_s': float(production @ axial_fraction),
        'side_wall_ions_s': float(boundary[grid.boundary_kind == 0].sum()),
        'back_wall_ions_s': float(boundary[grid.boundary_kind == 1].sum()),
        'exit_r_faces_m': grid.rf.tolist(),
        'exit_ions_s_by_ring': exit_flow.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Accepted spatial output or continuation step')
    parser.add_argument('--geometry', type=Path, required=True, help='Geometry JSON used by the source')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.source/'spatial_summary.json'
    fields_path = args.source/'plasma_fields.npz'
    hashes = {str(path.resolve()): digest(path) for path in (summary_path, fields_path, args.geometry)}
    geometry = spatial.load_geometry(args.geometry)
    summary = read_json(summary_path)
    config = summary['input']
    with np.load(fields_path, allow_pickle=False) as data:
        fields = dict(data)
    grid, _, checks, terms = reconstruct(geometry, config, summary, fields)
    tolerance = config['residual_tolerance']
    if summary['status'] != 'converged' or max(
        *checks['reconstructed_max_local_residuals'].values(),
        checks['relative_mass_error'], checks['relative_power_error'],
    ) >= tolerance:
        raise ValueError('Source state fails conservation checks')
    fine_config = dict(config, nr=2*config['nr'])
    fine_grid = spatial.Grid(geometry, fine_config['nr'], fine_config['nz'])
    heating = frozen_heating(grid, fine_grid, fields['q_abs_W_m3'])
    initial = interpolate(fields, grid, fine_grid, fine_config)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'geometry.json', geometry)
    report = {'source_sha256': hashes, 'source_grid': [config['nr'], config['nz']],
              'fine_grid': [fine_config['nr'], fine_config['nz']], 'source': budget(grid, terms)}
    for name, c, q, seed in (('same_grid', config, fields['q_abs_W_m3'], fields),
                              ('fine_grid', fine_config, heating, initial)):
        result, solved = spatial.solve(geometry, c, q, initial=seed, solver='sparse_newton')
        spatial.save(args.out/name, result, solved)
        solved_grid, _, _, solved_terms = reconstruct(geometry, c, result, solved)
        values = budget(solved_grid, solved_terms)
        comparison = {key: values[key]-value for key, value in report['source'].items()
                      if isinstance(value, float)}
        report[name] = {'status': result['status'], 'budget': values, 'change_s': comparison}
        passed = result['status'] == 'converged'
        if name == 'same_grid':
            # Include each ring so agreement in total exit flow cannot hide redistribution.
            keys = comparison.keys() | {'exit_ions_s_by_ring'}
            agrees = all(np.allclose(values[key], report['source'][key], rtol=1e-4, atol=0)
                         for key in keys)
            report[name]['agrees_with_source'] = agrees
            passed = passed and agrees
        if any(digest(path) != checksum for path, checksum in hashes.items()):
            raise ValueError('Source files changed during refinement')
        write_json(args.out/'comparison.json', report)
        if not passed:
            return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
