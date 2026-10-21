"""Iterate the RF wave solver and the volume-averaged argon model to steady state.

This is a scalar-power coupling: the RF solve returns integrated plasma
absorption, which drives the global plasma balance. The RF port voltage is
rescaled to hold accepted feed power fixed at the requested value.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys

from .. import config as geometry_config
from .. import plasma as global_plasma
from ..records import digest, read_json, utc_now, write_json
from .config import load as load_rf_config, validate as validate_rf_config
from .run import snapshot


def _relative_change(a, b):
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1e-300)


def _relax_state(old, new, fraction):
    """Log-relax positive density and temperature state variables."""
    out = {}
    for key in ('electron_density_m3', 'electron_temperature_eV', 'neutral_density_m3'):
        a, b = float(old[key]), float(new[key])
        if a <= 0 or b <= 0:
            raise ValueError(f'Cannot log-relax nonpositive plasma state {key}')
        out[key] = math.exp((1.0-fraction)*math.log(a) + fraction*math.log(b))
    return out


def _validated_global_state(result, label):
    if result.get('status') != 'steady':
        raise RuntimeError(f'{label} did not return a steady plasma state')
    residuals = result.get('balances', {}).get('relative_residuals', {})
    required = {'ion', 'neutral', 'total_mass', 'power'}
    if not required <= set(residuals) or any(
            not math.isfinite(float(residuals[key])) or abs(float(residuals[key])) > 1e-8
            for key in required):
        raise RuntimeError(f'{label} failed its particle/power conservation checks')
    try:
        metrics = result['metrics']
        state = {key: float(metrics[key]) for key in (
            'electron_density_m3', 'electron_temperature_eV', 'neutral_density_m3')}
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'{label} is missing a valid density/temperature state') from exc
    if (any(not math.isfinite(value) or value <= 0 for value in state.values())
            or not 1.0 <= state['electron_temperature_eV'] <= 7.0):
        raise RuntimeError(f'{label} returned nonphysical state values or Te outside 1–7 eV')
    return state


def _write_iteration_build(source, destination, config):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('rf.msh', 'mesh_report.json'):
        shutil.copy2(Path(source)/name, destination/name)
    report = read_json(destination/'mesh_report.json')
    write_json(destination/'rf.json', config)
    report['config_sha256'] = digest(destination/'rf.json')
    write_json(destination/'mesh_report.json', report)


def _run_rf(build, out, b_map, power_tolerance, plasma_map=None):
    from .solve import solve as solve_rf
    result = (solve_rf(build, out, b_map) if plasma_map is None else
              solve_rf(build, out, b_map, plasma_map=plasma_map))
    return _validated_rf_result(result, out, power_tolerance)


def _validated_rf_result(result, out, power_tolerance):
    """Shared acceptance gate for in-process and MPI-subprocess RF results."""
    checks = result.get('numerical_checks')
    required_checks = {
        'finite_power_values', 'positive_input_power', 'power_balance',
        'free_equation_residual', 'nonnegative_material_absorption',
        'nonnegative_pml_absorption', 'nonnegative_outward_interface_flux',
        'pml_interface_flux_agreement',
    }
    if not isinstance(checks, dict) or not required_checks <= set(checks):
        raise RuntimeError(
            f'RF run {out} lacks the required power sanity checks; update the RF solver before coupling')
    failed_checks = sorted(name for name in required_checks if checks[name] is not True)
    if not result.get('numerical_checks_passed') or failed_checks:
        raise RuntimeError(
            f"RF numerical checks failed in {out}: {', '.join(failed_checks) or 'overall check failed'}. "
            f"Inspect {out}/rf_summary.json before changing any coupling inputs.")
    pin = float(result['input_power_W'])
    absorbed = float(result['material_absorption_W']['PLASMA'])
    scale = max(abs(pin), 1e-30)
    if not math.isfinite(pin) or not math.isfinite(absorbed) or absorbed < 0:
        raise RuntimeError(f'RF run {out} returned invalid input or plasma-absorption power')
    if absorbed > pin + scale*power_tolerance:
        raise RuntimeError(
            f'RF run {out} reports plasma absorption ({absorbed:g} W) above accepted input ({pin:g} W)')
    return result


def couple(build, out, feed_power_W, plasma_config=None, geometry=None, b_map=None,
           max_iterations=12, tolerance=0.01, relaxation=0.5):
    """Run a fixed-feed-power RF/global-plasma fixed-point iteration.

    ``feed_power_W`` is accepted power at the modeled gap port, not generator
    nameplate power and not plasma absorbed power.
    """
    from .solve import runtime
    MPI, _, _ = runtime()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError('The coupling driver is serial for now; run it with one MPI rank.')
    build, out = Path(build), Path(out)
    if not math.isfinite(feed_power_W) or feed_power_W <= 0:
        raise ValueError('feed_power_W must be finite and positive')
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    if not math.isfinite(relaxation) or not 0 < relaxation <= 1:
        raise ValueError('relaxation must be in (0, 1]')

    rf_base = load_rf_config(build/'rf.json')
    if rf_base['material_case'] != 'plasma':
        raise ValueError('Coupled mode requires a prepared mesh with material_case="plasma"')
    geometry = geometry_config.load(geometry) if isinstance(geometry, (str, Path)) or geometry is None else geometry
    plasma_base = global_plasma.load(plasma_config)
    h = rf_base['chamber']
    for rf_key, plasma_key in (('radius_m', 'radius'), ('length_m', 'length')):
        if abs(h[rf_key] - geometry['chamber'][plasma_key]) > 1e-12:
            raise ValueError('RF and global plasma chamber dimensions differ')
    if b_map is None and (build.parent/'background_B.npz').exists():
        b_map = build.parent/'background_B.npz'
    if b_map is not None:
        b_map = Path(b_map)
        if not b_map.is_file():
            raise FileNotFoundError(b_map)

    out.mkdir(parents=True, exist_ok=False)
    record = {
        'model': 'rf_global_fixed_point_v1', 'status': 'running',
        'started_utc': utc_now(), 'build_mesh_sha256': digest(build/'rf.msh'),
        'build_config_sha256': digest(build/'rf.json'),
        'magnetic_map_sha256': digest(b_map) if b_map else None,
        'feed_power_W': feed_power_W, 'tolerance': tolerance,
        'relaxation': relaxation, 'max_iterations': max_iterations,
        'progress': {'stage': 'created', 'iteration': None, 'completed_iterations': 0},
    }
    write_json(out/'run.json', record)
    history = []

    def checkpoint(stage, iteration=None):
        record['progress'] = {
            'stage': stage, 'iteration': iteration,
            'completed_iterations': len(history),
        }
        record['updated_utc'] = utc_now()
        write_json(out/'progress.json', {
            'status': record['status'], 'progress': record['progress'],
            'history': history, 'updated_utc': record['updated_utc'],
        })
        write_json(out/'run.json', record)

    try:
        checkpoint('saving_inputs')
        record['source'] = snapshot(out)
        shutil.copy2(build/'rf.msh', out/'rf.msh')
        if b_map:
            shutil.copy2(b_map, out/'background_B.npz')
        write_json(out/'rf_base.json', rf_base)
        write_json(out/'plasma_base.json', plasma_base)
        write_json(out/'geometry.json', geometry)
        checkpoint('seed_global_plasma')
        seed = global_plasma.solve(geometry, copy.deepcopy(plasma_base))
        state = _validated_global_state(seed, 'Seed global plasma solve')
        previous_absorbed = None
        final_rf_dir = None
        final_global = None
        converged = False

        for index in range(1, max_iterations+1):
            iteration = out/'iterations'/f'iteration-{index:03d}'
            unit_cfg = copy.deepcopy(rf_base)
            unit_voltage = float(rf_base['voltage_peak_V'])
            unit_cfg['voltage_peak_V'] = unit_voltage
            unit_cfg['plasma'].update(state)
            validate_rf_config(unit_cfg)
            unit_build = iteration/'unit-build'
            _write_iteration_build(build, unit_build, unit_cfg)
            unit_dir = iteration/'unit-voltage'
            checkpoint('unit_voltage_rf', index)
            unit_result = _run_rf(unit_build, unit_dir, b_map,
                                  rf_base['solver']['power_balance_tolerance'])
            checkpoint('unit_voltage_rf_complete', index)
            unit_pin = float(unit_result['input_power_W'])
            if unit_pin <= 0:
                raise RuntimeError('Unit-voltage RF solve returned nonpositive accepted feed power')

            # The fixed-state Maxwell problem is linear in drive voltage, so P ~ V^2.
            checkpoint('drive_voltage_scaling', index)
            voltage = unit_voltage*math.sqrt(feed_power_W/unit_pin)
            driven_cfg = copy.deepcopy(unit_cfg)
            driven_cfg['voltage_peak_V'] = voltage
            driven_build = iteration/'driven-build'
            _write_iteration_build(build, driven_build, driven_cfg)
            rf_dir = iteration/'rf'
            checkpoint('driven_rf', index)
            rf_result = _run_rf(driven_build, rf_dir, b_map,
                                rf_base['solver']['power_balance_tolerance'])
            checkpoint('driven_rf_complete', index)
            accepted = float(rf_result['input_power_W'])
            if accepted <= 0:
                raise RuntimeError('Driven RF solve returned nonpositive accepted feed power')
            feed_error = _relative_change(feed_power_W, accepted)
            if feed_error > max(tolerance, 1e-6):
                raise RuntimeError(
                    f'Voltage-scaled port power misses requested feed power by {feed_error:.3%}')
            absorbed = float(rf_result['material_absorption_W']['PLASMA'])
            if not math.isfinite(absorbed) or absorbed <= 0:
                raise RuntimeError('RF solve returned no positive plasma-absorbed power')
            plasma_cfg = copy.deepcopy(plasma_base)
            plasma_cfg['absorbed_power_W'] = absorbed
            checkpoint('global_plasma_balance', index)
            global_result = global_plasma.solve(geometry, plasma_cfg)
            new_state = _validated_global_state(global_result, 'Global plasma balance')
            state_changes = {key: _relative_change(state[key], new_state[key]) for key in state}
            absorbed_change = (None if previous_absorbed is None else
                               _relative_change(previous_absorbed, absorbed))
            entry = {
                'iteration': index, 'plasma_state_used': dict(state),
                'voltage_peak_V': voltage, 'accepted_feed_power_W': accepted,
                'feed_power_relative_error': feed_error,
                'plasma_absorbed_power_W': absorbed,
                'pml_absorption_W': float(rf_result['pml_absorption_W']),
                'global_model_state': new_state,
                'state_relative_changes': state_changes,
                'absorbed_power_relative_change': absorbed_change,
                'rf_run': str(rf_dir.relative_to(out)),
            }
            history.append(entry)
            checkpoint('iteration_complete', index)
            final_rf_dir, final_global = rf_dir, global_result
            if (absorbed_change is not None and absorbed_change < tolerance
                    and max(state_changes.values()) < tolerance):
                converged = True
                break
            state = _relax_state(state, new_state, relaxation)
            previous_absorbed = absorbed

        result = {
            'model': 'rf_global_fixed_point_v1',
            'status': 'converged' if converged else 'not_converged',
            'input': {
                'feed_power_W': feed_power_W,
                'definition': 'accepted power at ideal antenna gap port',
                'frequency_Hz': rf_base['frequency_Hz'],
                'mass_flow_kg_s': plasma_base['mass_flow_kg_s'],
                'collision_frequency_s-1': rf_base['plasma']['collision_frequency_Hz'],
                'tolerance': tolerance, 'relaxation': relaxation,
            },
            'iterations': history,
            'iteration_count': len(history),
            'plasma_state': final_global['metrics'] if final_global else None,
            'plasma_balances': final_global['balances'] if final_global else None,
            'final_rf_run': str(final_rf_dir.relative_to(out)) if final_rf_dir else None,
            'limitations': [
                'Scalar feedback only: integrated RF absorption drives the existing uniform 0D plasma model.',
                'The global model retains prescribed wall/exit loss ratios and its existing argon rate limits.',
                'Electron momentum collision frequency remains a fixed user input during iteration.',
                'The ideal gap voltage source is not a generator, coax, or matching-network model.',
                'This steady fixed point is not a time-dependent stability or ignition calculation.',
                'Numerical fixed-point convergence does not validate mesh, PML, plasma closure or experiment.',
            ],
        }
        write_json(out/'coupling_summary.json', result)
        if converged:
            from .report import render
            render(final_rf_dir)
        record.update(status=result['status'], finished_utc=utc_now(),
                      iteration_count=len(history),
                      coupling_summary_sha256=digest(out/'coupling_summary.json'))
        checkpoint('converged' if converged else 'iteration_limit_reached')
        record['status'] = result['status']
        record['finished_utc'] = utc_now()
        write_json(out/'run.json', record)
        if not converged:
            raise RuntimeError(
                f'Coupled iteration did not converge in {max_iterations} iterations; '
                f'see {out}/coupling_summary.json')
        return result
    except Exception as exc:
        stage = record.get('progress', {}).get('stage')
        record.update(status='failed', failed_stage=stage,
                      error=f'{type(exc).__name__}: {exc}', finished_utc=utc_now())
        write_json(out/'progress.json', {
            'status': 'failed', 'failed_stage': stage,
            'progress': record.get('progress'), 'history': history,
            'error': record['error'], 'updated_utc': record['finished_utc'],
        })
        write_json(out/'run.json', record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True, help='Prepared plasma RF mesh directory')
    parser.add_argument('--out', required=True, help='New coupled-run directory')
    parser.add_argument('--feed-power-W', type=float, required=True,
                        help='Accepted RF power at the ideal antenna gap port')
    parser.add_argument('--plasma-config', help='Global argon model JSON; default config/plasma_argon.json')
    parser.add_argument('--geometry', help='Magnetostatic geometry JSON; default config/geometry.json')
    parser.add_argument('--b-map', help='Optional saved axisymmetric magnetic field NPZ')
    parser.add_argument('--max-iterations', type=int, default=12)
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='Relative tolerance for plasma state and absorbed power')
    parser.add_argument('--relaxation', type=float, default=0.5,
                        help='Log-space plasma-state update fraction in (0, 1]')
    args = parser.parse_args()
    try:
        result = couple(args.build, args.out, args.feed_power_W,
                        args.plasma_config, args.geometry, args.b_map,
                        args.max_iterations, args.tolerance, args.relaxation)
    except Exception as exc:
        print(f'Coupled RF run failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps({'status': result['status'], 'iteration_count': result['iteration_count'],
                      'plasma_state': result['plasma_state'],
                      'final_rf_run': result['final_rf_run'],
                      'summary': str((Path(args.out)/'coupling_summary.json').resolve())}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
