"""Follow steady plasma responses from uniform to fixed RF-shaped heating.

Alpha is a numerical continuation parameter, not time or RF feedback.
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
from ..records import digest, write_json, utc_now
from .prescribed import NOTE, inspect_rf, normalized_heating, metrics, validate_response
from .run import snapshot


def mixed_heating(uniform, target, alpha, volumes, power):
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Alpha must be in [0, 1]')
    q = (1-alpha)*uniform+alpha*target
    actual = float(np.sum(q*volumes))
    if (not np.isfinite(q).all() or np.any(q < 0) or
            abs(actual-power)/power > 1e-12):
        raise ValueError('Continuation heating does not conserve prescribed power')
    return q


def render_report(out, record):
    rows = ''.join('<tr><td>'+str(row['alpha'])+'</td><td>'+str(row['accepted'])+
        '</td><td>'+f"{row['Te_max_eV']:.6g}"+'</td><td>'+
        f"{row['max_local_relative_residual']:.3g}"+'</td><td><a href="'+
        row['directory']+'/spatial_summary.json">diagnostics</a></td></tr>'
        for row in record['history'])
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Prescribed heating continuation</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px}'
        'td,th{padding:8px;text-align:left}</style>'
        '<h1>Prescribed heating continuation: '+html.escape(record['status'])+'</h1>'
        '<p>'+html.escape(NOTE)+'</p><p>Alpha changes the spatial heating shape. '
        'This is a sequence of steady solves, not physical time evolution.</p>'
        '<p>Last accepted alpha: '+str(record['last_accepted_alpha'])+'</p>'
        '<p>'+html.escape(record.get('error',''))+'</p>'
        '<table><tr><th>Alpha</th><th>Accepted</th><th>Maximum Te [eV]</th>'
        '<th>Maximum local residual</th><th>Results</th></tr>'+rows+'</table>')


def run(rf_solution, out, absorbed_power_W, plasma_config=None, geometry=None,
        steps=10, allow_exploratory_interface=False, plots=True, heating_transfer="centroid",
        transfer_samples=512, transfer_rtol=.005, transfer_mapping=None, initial_state=None):
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1000:
        raise ValueError('Steps must be an integer in [1, 1000]')
    if not np.isfinite(absorbed_power_W) or absorbed_power_W <= 0:
        raise ValueError('Prescribed absorbed power must be positive and finite')
    for key in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE'):
        if int(os.environ.get(key, '1')) > 1:
            raise ValueError('Launch one continuation command without mpiexec')
    c, gconfig = spatial.load(plasma_config), load_geometry(geometry)
    c = copy.deepcopy(c); c['absorbed_power_W'] = float(absorbed_power_W)
    rf_solution, out = Path(rf_solution).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    record = {'model':'rf_heating_continuation_v1', 'status':'running',
              'started_utc':utc_now(), 'rf_solution':str(rf_solution),
              'steps':steps, 'prescribed_absorbed_power_W':float(absorbed_power_W),
              'last_accepted_alpha':None, 'history':[], 'note':NOTE,
              'rf_feedback_performed':False, 'stage':'rf_preflight'}

    def checkpoint():
        write_json(out/'run.json', record)
        write_json(out/'continuation_summary.json', record)
        render_report(out, record)

    checkpoint()
    try:
        record['source'] = snapshot(out)
        from ..argon_chemistry import metadata
        record['transport_chemistry'] = metadata(c)
        rf, wave, hashes, acceptance = inspect_rf(rf_solution, gconfig, allow_exploratory_interface)
        record.update(input_sha256=hashes, interface_acceptance=acceptance,
                      rf_configuration=rf, transport_uniform_B_T=c['uniform_B_T'],
                      rf_magnetic_map_present='background_B.npz' in hashes)
        write_json(out/'rf_summary_original.json', wave)
        write_json(out/'plasma_config.json', c)
        write_json(out/'geometry.json', gconfig)
        grid = spatial.Grid(gconfig, c['nr'], c['nz'])
        record['stage'] = 'heating_transfer'
        checkpoint()
        raw, target, transfer = normalized_heating(rf_solution, grid,
            wave['material_absorption_W']['PLASMA'], absorbed_power_W, heating_transfer,
            transfer_samples, transfer_rtol, transfer_mapping, out/'transfer')
        uniform = np.full(grid.shape, absorbed_power_W/grid.volumes.sum())
        np.savez_compressed(out/'heating_endpoints.npz', r_faces_m=grid.rf, z_faces_m=grid.zf,
            original_rf_q_W_m3=raw, uniform_q_W_m3=uniform, target_rf_q_W_m3=target)
        write_json(out/'transfer.json', transfer)
        initial = None
        seed_info = None
        if initial_state is not None:
            from ..spatial_seed import load_seed
            record['stage'] = 'grid_initialization'
            checkpoint()
            initial, seed_info = load_seed(initial_state, gconfig, c, out/'initialization')
            record['grid_initialization'] = seed_info
            checkpoint()
        for index in range(steps+1):
            alpha = index/steps
            q = mixed_heating(uniform, target, alpha, grid.volumes, absorbed_power_W)
            folder = out/'steps'/f'step-{index:03d}'
            record.update(stage='transport', current_alpha=alpha,
                          candidate_directory=str(folder.relative_to(out)))
            checkpoint()
            summary, fields = spatial.solve(gconfig, c, q, initial=initial)
            spatial.save(folder, summary, fields)
            error = None
            try:
                validate_response(summary, fields)
            except RuntimeError as exc:
                error = str(exc)
            balances = summary['balances']
            row = {'alpha':alpha, 'accepted':error is None,
                   'initialized_from_alpha':record['last_accepted_alpha'],
                   'initialization':('coarse_grid_seed' if seed_info is not None else 'generic') if index==0 else 'previous_alpha',
                   'directory':str(folder.relative_to(out)),
                   'Te_min_eV':float(fields['electron_temperature_eV'].min()),
                   'Te_max_eV':float(fields['electron_temperature_eV'].max()),
                   'solver_nfev':summary.get('solver',{}).get('nfev'),
                   'max_local_relative_residual':float(max(
                       abs(v) for v in balances['max_local_relative_residuals'].values())),
                   'relative_mass_error':balances['relative_mass_error'],
                   'relative_power_error':balances['relative_power_error'],
                   'integrated_heating_W':float(np.sum(q*grid.volumes)),
                   'radial_power_W':np.sum(q*grid.volumes,axis=0).tolist(),
                   'metrics':metrics(summary, fields, grid)}
            record['history'].append(row)
            if error:
                # Save rejected candidate separately; never use it to seed another step.
                record.update(status='not_converged', error=error, failed_alpha=alpha)
            else:
                record.update(last_accepted_alpha=alpha,
                              last_accepted_directory=row['directory'])
                initial = fields
            checkpoint()
            if plots:
                from ..spatial_report import render
                render(folder, summary, fields)
            print(f"alpha={alpha:.3f} accepted={row['accepted']} "
                  f"Te_max={row['Te_max_eV']:.6g} eV "
                  f"max_residual={row['max_local_relative_residual']:.3g}",flush=True)
            if error:
                break
        if any(digest(rf_solution/name) != sha for name, sha in hashes.items()):
            raise ValueError('RF source artifacts changed during continuation')
        if seed_info is not None:
            from ..spatial_seed import verify_source
            verify_source(seed_info)
        if record['status'] == 'running':
            record['status'] = 'completed'
        record.update(stage='finished', finished_utc=utc_now())
        record['artifacts_sha256'] = {str(p.relative_to(out)):digest(p)
            for p in out.rglob('*') if p.is_file() and 'source' not in p.relative_to(out).parts
            and p.name not in ('run.json','continuation_summary.json','report.html')}
        checkpoint()
        return record
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}', finished_utc=utc_now())
        checkpoint()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rf-solution', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--absorbed-power-W', type=float, required=True)
    parser.add_argument('--plasma-config'); parser.add_argument('--geometry')
    parser.add_argument('--steps', type=int, default=10,
                        help='Equal alpha increments from 0 to 1; default 0.1')
    parser.add_argument('--allow-exploratory-interface', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--heating-transfer', choices=('centroid','sampled'), default='centroid')
    parser.add_argument('--transfer-samples', type=int, default=512)
    parser.add_argument('--transfer-rtol', type=float, default=.005)
    parser.add_argument('--transfer-mapping', help='Previously verified mesh_mapping.npz cache')
    parser.add_argument('--initial-state', help='Accepted alpha=0 continuation step on the same physical problem')
    args = parser.parse_args()
    try:
        result = run(args.rf_solution,args.out,args.absorbed_power_W,args.plasma_config,
                     args.geometry,args.steps,args.allow_exploratory_interface,not args.no_plots,
                     args.heating_transfer,args.transfer_samples,args.transfer_rtol,args.transfer_mapping,args.initial_state)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Continuation stopped: {exc}. See {args.out}/run.json if created.',file=sys.stderr)
        return 2
    print(f"Continuation {result['status']}; last accepted alpha="
          f"{result['last_accepted_alpha']}. See {args.out}/report.html")
    return 0 if result['status']=='completed' else 2


if __name__=='__main__':
    sys.exit(main())
