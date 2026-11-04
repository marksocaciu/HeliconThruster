"""One-way prescribed-heating experiment from saved RF cell integrals.

No Maxwell solve, voltage scaling, or RF/plasma feedback is performed.
"""
import argparse
import copy
import html
import os
from pathlib import Path
import sys

import numpy as np

from .. import spatial
from ..config import load as load_geometry
from ..records import digest, read_json, write_json, utc_now
from .config import load as load_rf
from .coupled import _validated_rf_result
from .run import snapshot
from .spatial_coupled import validate_spatial
from .spatial_maps import heating_from_rf


NOTE = ('Exploratory one-way prescribed heating from a fixed coarse RF field. '
        'Absorbed power is chosen by the user; it is not predicted RF coupling '
        'efficiency. Mesh, open-boundary and physical convergence are unverified. '
        'The plasma response is not fed back to RF. Transport uses its configured '
        'uniform axial magnetic field even when the RF source used a field map.')


def inspect_rf(directory, geometry, allow_exploratory_interface=False):
    """Require complete exports and normal RF gates; isolate the interface allowance."""
    directory = Path(directory)
    c = load_rf(directory/'rf.json')
    result = read_json(directory/'rf_summary.json')
    if c['material_case'] != 'plasma' or result.get('case') != 'plasma':
        raise ValueError('Require a completed plasma RF solution')
    for rk, gk in (('radius_m', 'radius'), ('length_m', 'length'), ('z_start_m', 'z_start')):
        if abs(c['chamber'][rk]-geometry['chamber'][gk]) > 1e-12:
            raise ValueError('RF and transport chamber geometry differ')
    _validated_rf_result(result, directory, c['solver']['power_balance_tolerance'])
    if result.get('status') != 'completed':
        raise ValueError('RF run is not completed')
    # Re-evaluate the scalar diagnostics as well as checking saved flags.
    from .solve import _power_sanity_checks
    checks = _power_sanity_checks(result['input_power_W'], result['material_absorption_W'],
        result['pml_absorption_W'], result['outward_interface_flux_W'],
        result['power_balance_relative_error'], result['free_equation_relative_residual'],
        c['solver']['power_balance_tolerance'], c['solver']['relative_residual_tolerance'])
    if not all(checks.values()):
        raise ValueError('RF scalar diagnostics fail the current numerical checks')
    pin = float(result['input_power_W'])
    roundoff = pin*1e-12
    if (result['pml_absorption_W'] < -roundoff or
            any(v < -roundoff for v in result['material_absorption_W'].values())):
        raise ValueError('Negative material/PML absorption exceeds roundoff allowance')
    flux = float(result['outward_interface_flux_W'])
    trace = float(result['pml_side_interface_flux_W'])
    if (not np.isfinite(trace) or
            abs(flux-trace)/pin >= c['solver']['power_balance_tolerance']):
        raise ValueError('RF interface traces disagree')
    exploratory = flux < -roundoff
    if exploratory and not allow_exploratory_interface:
        raise ValueError('Negative outward interface flux: use --allow-exploratory-interface '
                         'only for the provisional prescribed-heating experiment')
    ranks = result.get('mpi_ranks')
    if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1:
        raise ValueError('RF summary lacks a valid MPI rank count')
    paths = sorted(directory.glob('cells_rank*.npz'))
    expected = [directory/f'cells_rank{i:04d}.npz' for i in range(ranks)]
    if paths != expected:
        raise ValueError('RF cell exports do not match the complete MPI rank set')
    names = ['rf.json', 'rf_summary.json', 'mesh_report.json', 'rf.msh', 'run.json']
    names += [p.name for p in paths]
    if (directory/'background_B.npz').exists(): names.append('background_B.npz')
    hashes = {name: digest(directory/name) for name in names}
    report = read_json(directory/'mesh_report.json')
    if hashes['rf.json'] != report['config_sha256'] or hashes['rf.msh'] != report['mesh_sha256']:
        raise ValueError('RF mesh/config provenance mismatch')
    record = read_json(directory/'run.json')
    if record.get('status') != 'completed':
        raise ValueError('RF run record is not completed')
    manifest = record.get('artifacts_sha256', {})
    for name in names:
        if name != 'run.json' and (name not in manifest or hashes[name] != manifest[name]):
            raise ValueError('Missing or mismatched RF artifact hash: '+name)
    linear = result.get('solver_diagnostics')
    if linear is not None and linear.get('accepted') is not True:
        raise ValueError('RF linear solve was not accepted')
    return c, result, hashes, {'negative_interface_flux_used': exploratory,
        'outward_flux_relative_to_input': flux/pin,
        'interface_allowance_relative_to_input': 1e-4 if allow_exploratory_interface else 1e-12,
        'material_pml_roundoff_relative_to_input': 1e-12}


def normalized_heating(directory, grid, original_power_W, absorbed_power_W):
    if not np.isfinite(absorbed_power_W) or absorbed_power_W <= 0:
        raise ValueError('Prescribed absorbed power must be finite and positive')
    raw, info = heating_from_rf(directory, grid, original_power_W)
    if info['occupied_bin_fraction'] < 1:
        raise ValueError('RF mesh leaves empty heating bins; use a coarser transport grid')
    factor = absorbed_power_W/info['integrated_power_W']
    q = raw*factor
    integrated = float(np.sum(q*grid.volumes))
    if not np.isfinite(q).all() or abs(integrated-absorbed_power_W)/absorbed_power_W > 1e-12:
        raise ValueError('Prescribed heating normalization failed')
    info.update(original_rf_absorption_W=original_power_W,
                prescribed_absorbed_power_W=absorbed_power_W,
                normalization_factor=factor, normalized_integrated_power_W=integrated)
    return raw, q, info


def metrics(summary, fields, grid):
    out = {}
    for key in ('electron_density_m3', 'neutral_density_m3', 'electron_temperature_eV'):
        values = fields[key]
        out[key] = {'volume_mean': float(np.sum(values*grid.volumes)/grid.volumes.sum()),
                    'min': float(values.min()), 'max': float(values.max())}
    out['exit_ions_s'] = summary['balances']['exit_ions_s']
    return out


def validate_response(summary, fields):
    try:
        validate_spatial(summary)
    except RuntimeError as exc:
        te = fields['electron_temperature_eV']
        raise RuntimeError(f'{exc}; Te range {te.min():.6g}–{te.max():.6g} eV '
                           '(supported chemistry: 1–7 eV). See saved transport diagnostics.') from exc


def report(out, result, plots):
    rows = []
    for key, value in result['rf_shaped_metrics'].items():
        if isinstance(value, dict):
            uniform = result['uniform_metrics'][key]['volume_mean']
            rows.append(f'<tr><td>{html.escape(key)} (volume mean)</td>'
                        f'<td>{uniform:.6g}</td><td>{value["volume_mean"]:.6g}</td></tr>')
    links = ''.join(f'<li><a href="{case}/spatial_summary.json">{case} diagnostics</a></li>'
                    for case in ('uniform', 'rf-shaped'))
    figures = ''.join(f'<h2>{case}</h2><img src="{case}/spatial_fields.png" alt="{case} plasma fields">'
                      for case in ('uniform', 'rf-shaped')) if plots else ''
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Coarse RF prescribed-heating experiment</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}'
        'td,th{padding:8px;text-align:left}img{max-width:100%}</style>'
        '<h1>Coarse RF prescribed-heating experiment</h1><p>'+html.escape(NOTE)+'</p>'
        f'<p>Status: {result["status"]}. Prescribed plasma absorption: '
        f'{result["transfer"]["prescribed_absorbed_power_W"]:.6g} W.</p>'
        '<table><tr><th>Quantity</th><th>Uniform heating</th><th>RF-shaped heating</th></tr>'
        +''.join(rows)+'</table><ul>'+links+'</ul>'+figures)


def run(rf_solution, out, absorbed_power_W, plasma_config=None, geometry=None,
        allow_exploratory_interface=False, plots=True):
    for key in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE'):
        if int(os.environ.get(key, '1')) > 1:
            raise ValueError('Run this command once with Python, without mpiexec')
    rf_solution, out = Path(rf_solution).resolve(), Path(out)
    c, gconfig = spatial.load(plasma_config), load_geometry(geometry)
    if not np.isfinite(absorbed_power_W) or absorbed_power_W <= 0:
        raise ValueError('Prescribed absorbed power must be finite and positive')
    # Reserve before expensive input reads or transport solves. Never overwrite.
    out.mkdir(parents=True, exist_ok=False)
    record = {'model': 'rf_prescribed_heating_v1', 'status': 'running',
              'started_utc': utc_now(), 'rf_solution': str(rf_solution), 'note': NOTE}
    write_json(out/'run.json', record)
    try:
        record['source'] = snapshot(out)
        record['stage'] = 'rf_preflight'
        write_json(out/'run.json', record)
        rf, wave, hashes, acceptance = inspect_rf(rf_solution, gconfig, allow_exploratory_interface)
        record.update(input_sha256=hashes, interface_acceptance=acceptance,
                      rf_configuration=rf, rf_magnetic_map_present='background_B.npz' in hashes)
        write_json(out/'rf_summary_original.json', wave)
        write_json(out/'geometry.json', gconfig)
        c = copy.deepcopy(c); c['absorbed_power_W'] = float(absorbed_power_W)
        write_json(out/'plasma_config.json', c)
        grid = spatial.Grid(gconfig, c['nr'], c['nz'])
        raw, q, transfer = normalized_heating(rf_solution, grid,
            wave['material_absorption_W']['PLASMA'], absorbed_power_W)
        np.savez_compressed(out/'heating.npz', r_faces_m=grid.rf, z_faces_m=grid.zf,
                            q_abs_W_m3=q, original_q_abs_W_m3=raw)
        write_json(out/'transfer.json', transfer)
        record['stage'] = 'uniform_transport'; write_json(out/'run.json', record)
        uniform, ufields = spatial.solve(gconfig, c)
        spatial.save(out/'uniform', uniform, ufields)
        if plots:
            from ..spatial_report import render
            render(out/'uniform', uniform, ufields)
        validate_response(uniform, ufields)
        record['stage'] = 'rf_shaped_transport'; write_json(out/'run.json', record)
        shaped, fields = spatial.solve(gconfig, c, q, initial=ufields)
        spatial.save(out/'rf-shaped', shaped, fields)
        if plots:
            render(out/'rf-shaped', shaped, fields)
        validate_response(shaped, fields)
        # Detect changed inputs during the experiment before declaring success.
        if any(digest(rf_solution/name) != sha for name, sha in hashes.items()):
            raise ValueError('RF source artifacts changed during the experiment')
        result = {'model': record['model'], 'status': 'completed', 'note': NOTE,
                  'transfer': transfer, 'interface_acceptance': acceptance,
                  'uniform_metrics': metrics(uniform, ufields, grid),
                  'rf_shaped_metrics': metrics(shaped, fields, grid),
                  'transport_uniform_B_T': c['uniform_B_T'],
                  'rf_feedback_performed': False}
        write_json(out/'experiment_summary.json', result)
        report(out, result, plots)
        record.update(status='completed', stage='completed', finished_utc=utc_now())
        record['artifacts_sha256'] = {str(p.relative_to(out)): digest(p)
            for p in out.rglob('*') if p.is_file() and 'source' not in p.relative_to(out).parts
            and p != out/'run.json'}
        write_json(out/'run.json', record)
        return result
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}', finished_utc=utc_now())
        write_json(out/'run.json', record)
        links = ''.join(f'<li><a href="{case}/spatial_summary.json">{case} diagnostics</a></li>'
                        for case in ('uniform', 'rf-shaped')
                        if (out/case/'spatial_summary.json').exists())
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8">'
            '<h1>Prescribed-heating experiment: '+record['status']+'</h1><p>'
            +html.escape(NOTE)+'</p><p>'+html.escape(record['error'])+'</p><ul>'+links+'</ul>')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rf-solution', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--absorbed-power-W', type=float, required=True,
                        help='Prescribed plasma absorption, not accepted feed power')
    parser.add_argument('--plasma-config'); parser.add_argument('--geometry')
    parser.add_argument('--allow-exploratory-interface', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    try:
        run(args.rf_solution, args.out, args.absorbed_power_W, args.plasma_config,
            args.geometry, args.allow_exploratory_interface, not args.no_plots)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Prescribed-heating experiment stopped: {exc}. See {args.out}/run.json if created.', file=sys.stderr)
        return 2
    print(f'Completed exploratory prescribed-heating experiment: {args.out}/report.html')
    return 0


if __name__ == '__main__':
    sys.exit(main())
