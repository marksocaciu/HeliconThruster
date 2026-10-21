"""Fixed accepted-feed-power RF / axisymmetric argon transport iteration."""
import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
import shutil

import numpy as np

from .. import spatial
from ..config import load as load_geometry
from ..records import write_json, read_json, digest, utc_now
from .config import load as load_rf
from .coupled import _write_iteration_build, _relative_change
from .spatial_maps import heating_from_rf

STATE_KEYS = ('electron_density_m3', 'neutral_density_m3', 'electron_temperature_eV',
              'collision_frequency_s')


def state_change(old, new):
    changes = {key: float(np.max(np.abs(new[key]-old[key])/
               np.maximum(np.maximum(np.abs(old[key]), np.abs(new[key])), 1e-30)))
               for key in STATE_KEYS}
    # Gauge-invariant potential comparison; its physical scale is Te in volts.
    a = old['potential_V']-old['potential_V'].flat[0]
    b = new['potential_V']-new['potential_V'].flat[0]
    changes['potential_V'] = float(np.max(np.abs(a-b))/max(
        np.max(old['electron_temperature_eV']), np.max(new['electron_temperature_eV'])))
    return changes


def validate_spatial(summary):
    b = summary.get('balances', {})
    residuals = b.get('max_local_relative_residuals', {})
    required = {'ion_res', 'neutral_res', 'current_res', 'energy_res'}
    values = list(residuals.values())+[b.get('relative_mass_error', np.inf),
                                     b.get('relative_power_error', np.inf)]
    tol = summary.get('input', {}).get('residual_tolerance', 0)
    if (summary.get('status') != 'converged' or not required <= set(residuals)
            or not all(np.isfinite(v) and abs(v) < tol for v in values)):
        raise RuntimeError('Spatial plasma failed its local/integrated conservation checks')


def relax(old, target, fraction, c):
    out = {key: np.array(value, copy=True) for key, value in old.items()}
    for key in STATE_KEYS[:3]:
        out[key] = np.exp((1-fraction)*np.log(old[key])+fraction*np.log(target[key]))
    out['potential_V'] = (1-fraction)*old['potential_V']+fraction*target['potential_V']
    out['potential_V'] -= out['potential_V'].flat[0]
    # Collision state follows relaxed thermochemistry, not an independent lagged field.
    _, nu, _ = spatial.coefficients(out['electron_temperature_eV'].ravel(),
                                    out['neutral_density_m3'].ravel(), c)
    out['collision_frequency_s'] = nu.reshape(out['potential_V'].shape)
    return out


def _mpi_command(ranks, launcher):
    return [str(launcher), '-n', str(ranks), sys.executable,
            '-m', 'helicon.rf.solve', '--expected-ranks', str(ranks)]


def _launch(command, log):
    """Launch with the controller's Python/environment and retain rank output."""
    child_env = os.environ.copy()
    # The serial transport controller may use threaded BLAS, but RF MPI ranks
    # each get one thread to avoid ranks x BLAS-thread oversubscription.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        child_env[key] = '1'
    with Path(log).open('w') as stream:
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    check=False, env=child_env)
        except OSError as exc:
            raise RuntimeError(f'Could not launch MPI RF command: {exc}; see {log}') from exc
    if result.returncode != 0:
        raise RuntimeError(f'MPI RF command exited with code {result.returncode}; see {log}')


def run_rf(build, out, state_path, tolerance, rf_ranks=1, mpi_launcher='mpiexec'):
    from .coupled import _run_rf, _validated_rf_result
    if rf_ranks == 1:
        return _run_rf(build, out, None, tolerance, plasma_map=state_path)
    out = Path(out)
    log = out.parent/(out.name+'.mpi.log')
    command = _mpi_command(rf_ranks, mpi_launcher)+[
        '--build', str(Path(build).resolve()), '--out', str(out.resolve()),
        '--plasma-map', str(Path(state_path).resolve())]
    _launch(command, log)
    summary = out/'rf_summary.json'
    if not summary.is_file():
        raise RuntimeError(f'MPI RF command produced no summary; see {log}')
    result = _validated_rf_result(read_json(summary), out, tolerance)
    if result.get('mpi_ranks') != rf_ranks:
        raise RuntimeError('RF summary rank count differs from requested MPI size; check the MPI launcher/library pairing')
    return result


def couple(build, out, feed_power_W, plasma_config=None, geometry=None,
           max_iterations=20, tolerance=.01, relaxation=.4, rf_ranks=1,
           mpi_launcher='mpiexec'):
    if not isinstance(rf_ranks, int) or isinstance(rf_ranks, bool) or rf_ranks < 1:
        raise ValueError('rf_ranks must be a positive integer')
    for key in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE'):
        if int(os.environ.get(key, '1')) > 1:
            raise RuntimeError('Launch one controller directly with --rf-ranks N; do not run the coupling driver under mpiexec')
    if rf_ranks == 1:
        from .solve import runtime
        MPI, _, _ = runtime()
        if MPI.COMM_WORLD.size != 1:
            raise RuntimeError('Launch one coupling controller, using --rf-ranks N for parallel RF')
    elif shutil.which(mpi_launcher) is None:
        raise ValueError(f'MPI launcher not found: {mpi_launcher}')
    if not np.isfinite(feed_power_W) or feed_power_W <= 0:
        raise ValueError('feed_power_W must be finite and positive')
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 2:
        raise ValueError('max_iterations must be an integer >= 2')
    if not np.isfinite(tolerance) or not 0 < tolerance < .1:
        raise ValueError('tolerance must be in (0, .1)')
    if not np.isfinite(relaxation) or not 0 < relaxation <= 1:
        raise ValueError('relaxation must be in (0, 1]')
    build, out = Path(build), Path(out)
    rf = load_rf(build/'rf.json')
    c = spatial.load(plasma_config)
    geometry = load_geometry(geometry)
    if rf['material_case'] != 'plasma':
        raise ValueError('Prepare an RF mesh with material_case=plasma')
    for rk, gk in [('radius_m', 'radius'), ('length_m', 'length'), ('z_start_m', 'z_start')]:
        if abs(rf['chamber'][rk]-geometry['chamber'][gk]) > 1e-12:
            raise ValueError('Spatial and RF chamber dimensions/origin differ')
    if not np.array_equal(rf['plasma']['uniform_B_T'], c['uniform_B_T']):
        raise ValueError('RF and spatial uniform axial magnetic fields must match')
    if (build.parent/'background_B.npz').exists():
        raise ValueError('This benchmark uses uniform axial B transport; prepare a separate RF build without a B map')
    g = spatial.Grid(geometry, c['nr'], c['nz'])
    out.mkdir(parents=True, exist_ok=False)
    from .run import snapshot
    record = {'model': 'rf_spatial_fixed_point_v1', 'status': 'running',
              'started_utc': utc_now(), 'feed_power_W': feed_power_W,
              'tolerance': tolerance, 'relaxation': relaxation, 'max_iterations': max_iterations,
              'rf_ranks': rf_ranks, 'mpi_launcher': mpi_launcher,
              'controller_thread_environment': {key: os.environ.get(key) for key in
                  ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
              'rf_threads_per_rank': 1 if rf_ranks > 1 else None,
              'mesh_sha256': digest(build/'rf.msh'), 'rf_config_sha256': digest(build/'rf.json')}
    history = []

    def checkpoint(stage):
        record.update(stage=stage, iterations_completed=len(history), updated_utc=utc_now())
        write_json(out/'run.json', record)
        write_json(out/'progress.json', {'status': record['status'], 'stage': stage, 'history': history})

    try:
        checkpoint('runtime_preflight')
        record['source'] = snapshot(out)
        if rf_ranks > 1:
            # Do not initialize MPI in the serial controller before starting mpiexec.
            _launch(_mpi_command(rf_ranks, mpi_launcher)+['--check-runtime'],
                    out/'runtime.mpi.log')
        checkpoint('seed_spatial')
        write_json(out/'plasma_config.json', c); write_json(out/'geometry.json', geometry)
        seed, state = spatial.solve(geometry, c)
        spatial.save(out/'seed', seed, state)
        validate_spatial(seed)
        previous_q = None
        for index in range(1, max_iterations+1):
            iteration = out/'iterations'/f'iteration-{index:03d}'
            iteration.mkdir(parents=True)
            state_path = iteration/'rf_input_plasma.npz'
            np.savez_compressed(state_path, **state)
            checkpoint('unit_rf')
            unit_build = iteration/'unit-build'
            _write_iteration_build(build, unit_build, rf)
            trial = run_rf(unit_build, iteration/'unit-rf', state_path, rf['solver']['power_balance_tolerance'],
                           rf_ranks=rf_ranks, mpi_launcher=mpi_launcher)
            voltage = rf['voltage_peak_V']*np.sqrt(feed_power_W/trial['input_power_W'])
            driven = copy.deepcopy(rf); driven['voltage_peak_V'] = float(voltage)
            driven_build = iteration/'driven-build'
            _write_iteration_build(build, driven_build, driven)
            checkpoint('driven_rf')
            rf_dir = iteration/'rf'
            wave = run_rf(driven_build, rf_dir, state_path, rf['solver']['power_balance_tolerance'],
                           rf_ranks=rf_ranks, mpi_launcher=mpi_launcher)
            feed_error = _relative_change(feed_power_W, wave['input_power_W'])
            if feed_error > 1e-6:
                raise RuntimeError('Scaled RF solve does not deliver the requested accepted feed power')
            absorbed = float(wave['material_absorption_W']['PLASMA'])
            q, transfer = heating_from_rf(rf_dir, g, absorbed)
            if transfer['occupied_bin_fraction'] < 1:
                raise RuntimeError('RF mesh leaves empty heating bins; refine the RF mesh or coarsen the spatial grid')
            checkpoint('spatial_transport')
            target_summary, target = spatial.solve(geometry, c, q, initial=state)
            spatial.save(iteration/'plasma', target_summary, target)
            validate_spatial(target_summary)
            changes = state_change(state, target)
            # The unrelaxed field residual must converge; small updates alone do not count.
            heating_change = None if previous_q is None else float(
                np.max(np.abs(q-previous_q))/max(np.max(q), np.max(previous_q), 1e-30))
            entry = {'iteration': index, 'accepted_feed_power_W': wave['input_power_W'],
                     'plasma_absorbed_power_W': absorbed, 'feed_power_relative_error': feed_error,
                     'voltage_peak_V': float(voltage), 'state_relative_changes': changes,
                     'heating_relative_change': heating_change, 'transfer': transfer,
                     'rf_power_balance_relative_error': wave['power_balance_relative_error'],
                     'plasma_balances': target_summary['balances'],
                     'rf_run': str(rf_dir.relative_to(out))}
            history.append(entry)
            checkpoint('iteration_complete')
            if heating_change is not None and heating_change < tolerance and max(changes.values()) < tolerance:
                shutil.copy2(state_path, out/'final_rf_input_plasma.npz')
                spatial.save(out/'final_plasma', target_summary, target)
                from ..spatial_report import render
                render(out/'final_plasma', target_summary, target)
                summary = {'model': record['model'], 'status': 'converged',
                           'history': history, 'final_rf_run': str(rf_dir.relative_to(out)),
                           'tolerance': tolerance,
                           'note': 'RF fields use final_rf_input_plasma.npz; final_plasma is the unrelaxed transport response. Their discrepancy is recorded, not assumed zero.'}
                write_json(out/'coupling_summary.json', summary)
                record.update(status='converged', finished_utc=utc_now())
                checkpoint('converged')
                return summary
            previous_q = q
            state = relax(state, target, relaxation, c)
        record['status'] = 'not_converged'
        write_json(out/'coupling_summary.json', {'status': 'not_converged', 'history': history})
        raise RuntimeError(f'Spatial RF fixed point did not converge in {max_iterations} iterations')
    except Exception as exc:
        record.update(status='failed' if record['status'] == 'running' else record['status'],
                      error=f'{type(exc).__name__}: {exc}', finished_utc=utc_now())
        checkpoint(record['stage'])
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', required=True); p.add_argument('--out', required=True)
    p.add_argument('--feed-power-W', type=float, required=True)
    p.add_argument('--plasma-config'); p.add_argument('--geometry')
    p.add_argument('--max-iterations', type=int, default=20)
    p.add_argument('--tolerance', type=float, default=.01)
    p.add_argument('--relaxation', type=float, default=.4)
    p.add_argument('--rf-ranks', type=int, default=1, help='MPI ranks for each RF solve; controller and transport remain serial')
    p.add_argument('--mpi-launcher', default='mpiexec', help='MPI launcher executable compatible with the Python MPI stack')
    a = p.parse_args()
    couple(a.build, a.out, a.feed_power_W, a.plasma_config, a.geometry,
           a.max_iterations, a.tolerance, a.relaxation, a.rf_ranks, a.mpi_launcher)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
