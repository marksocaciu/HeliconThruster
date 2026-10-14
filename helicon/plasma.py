"""Steady, volume-averaged argon source with prescribed loss coefficients.

Run with python3 -m helicon.plasma --out runs/plasma-baseline.
No FEM, MPI, Gmsh or SciPy import is needed. See docs/plasma-model.md for
the equations, the open-exit energy closure, and the limits of this model.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

from .argon import (E_CHARGE as E, K_B, M_E, M_AR, EPS_0, rates,
                    IONIZATION_EV, EXCITATION_EV, TE_MIN_EV, TE_MAX_EV, RATE_MODEL)
from .config import ROOT, load as load_geometry
from .records import config_digest, digest, source_record, utc_now, write_json

MODEL = 'argon_global_0d_v1'
LIMITATIONS = [
    'Illustrative global source model; not experimentally validated or a reproduction of the full paper.',
    'Boundary density ratios are prescribed relative to the volume mean; B does not determine them.',
    'Prescribed absorbed power; no antenna coupling, helicon mode transition, or ignition prediction.',
    'Open sonic exit: electron enthalpy plus cold-ion kinetic energy; zero conductive heat flux.',
    'Maxwellian electrons and ground-state argon; no metastable kinetics, stepwise ionization, or Ar2+.',
    'Thermal effusion into vacuum and prescribed neutral temperature; gas heating is an energy sink.',
    'No spatial profiles, nozzle acceleration, thrust, specific impulse, or stability prediction.',
]


class NoSteadyStateError(ValueError):
    """No admissible nonzero solution within the documented rate-fit interval."""


def validate(c):
    expected = {'model', 'gas', 'absorbed_power_W', 'mass_flow_kg_s',
                'neutral_temperature_K', 'losses', 'description'}
    if not isinstance(c, dict) or set(c) - expected:
        raise ValueError('Unknown plasma configuration keys')
    required = expected - {'description'}
    if not required <= set(c):
        raise ValueError(f'Missing plasma keys: {sorted(required - set(c))}')
    if c['model'] != MODEL or c['gas'] != 'argon':
        raise ValueError(f'Only model={MODEL} and gas=argon are supported')
    if 'description' in c and not isinstance(c['description'], str):
        raise ValueError('description must be text')

    def number(obj, key, low, high=None, strict=False):
        x = obj[key]
        if (isinstance(x, bool) or not isinstance(x, (int, float))
                or not math.isfinite(x) or x < low or (strict and x == low)
                or (high is not None and x > high)):
            raise ValueError(f'Invalid {key}: {x!r}')
    number(c, 'absorbed_power_W', 0)
    number(c, 'mass_flow_kg_s', 0, strict=True)
    number(c, 'neutral_temperature_K', 0, strict=True)
    if K_B*c['neutral_temperature_K']/E >= TE_MIN_EV:
        raise ValueError('Neutral temperature must be below the minimum electron temperature (1 eV)')
    loss_keys = {'side_density_ratio', 'back_density_ratio', 'exit_density_ratio',
                 'wall_recycling_fraction', 'neutral_exit_transmission'}
    loss = c['losses']
    if not isinstance(loss, dict) or set(loss) != loss_keys:
        raise ValueError(f'losses must contain exactly {sorted(loss_keys)}')
    for key in loss_keys:
        number(loss, key, 0, 1, strict=key in {'exit_density_ratio', 'neutral_exit_transmission'})


def load(path=None):
    c = json.loads(Path(path or ROOT/'config/plasma_argon.json').read_text())
    validate(c)
    return c


def geometry_terms(geometry, c):
    """Use only source dimensions; mesh and coil current are not plasma closures."""
    r, length = geometry['chamber']['radius'], geometry['chamber']['length']
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or x <= 0 for x in (r, length)):
        raise ValueError('Chamber radius and length must be finite and positive')
    area = math.pi*r*r
    side = 2*math.pi*r*length
    loss = c['losses']
    neutral_speed = math.sqrt(8*K_B*c['neutral_temperature_K']/(math.pi*M_AR))/4
    effective = {'side': side*loss['side_density_ratio'],
                 'back': area*loss['back_density_ratio'],
                 'exit': area*loss['exit_density_ratio']}
    return {'volume_m3': area*length, 'radius_m': r, 'length_m': length,
            'side_area_m2': side, 'exit_area_m2': area,
            'effective_areas_m2': effective,
            'neutral_exit_conductance_m3_s': area*neutral_speed*loss['neutral_exit_transmission'],
            'injected_atoms_s': c['mass_flow_kg_s']/M_AR}


def evaluate(ne, nn, te, c, g):
    """Independently evaluate all physical fluxes and conservation residuals."""
    k = rates(te)
    cs = math.sqrt(E*te/M_AR)
    ions = {name: ne*cs*area for name, area in g['effective_areas_m2'].items()}
    walls = ions['side']+ions['back']
    production = ne*nn*g['volume_m3']*k['ionization_m3_s']
    neutral_exit = nn*g['neutral_exit_conductance_m3_s']
    recycled = c['losses']['wall_recycling_fraction']*walls
    retained = walls-recycled
    # Floating, non-emitting planar wall: 2 Te electrons + (1/2 Te + e Vs) ions.
    wall_gamma = 2.5 + 0.5*math.log(M_AR/(2*math.pi*M_E))
    # Open source exit, NOT a floating-wall sheath: 5/2 Te enthalpy + 1/2 Te ions.
    power = {
        'ionization': E*production*IONIZATION_EV,
        'excitation': E*ne*nn*g['volume_m3']*k['excitation_m3_s']*EXCITATION_EV,
        'elastic_gas_heating': (E*ne*nn*g['volume_m3']*k['elastic_m3_s']
                                *3*M_E/M_AR*(te-K_B*c['neutral_temperature_K']/E)),
        'side_wall': E*ions['side']*wall_gamma*te,
        'back_wall': E*ions['back']*wall_gamma*te,
        'exit': E*ions['exit']*3*te,
    }
    q = g['injected_atoms_s']
    residual = {
        'ion_pairs_s': production-sum(ions.values()),
        'neutrals_s': q+recycled-production-neutral_exit,
        'total_atoms_s': q-neutral_exit-ions['exit']-retained,
        'power_W': c['absorbed_power_W']-sum(power.values()),
    }
    scaled = {
        'ion': residual['ion_pairs_s']/max(production, sum(ions.values()), q),
        'neutral': residual['neutrals_s']/max(q+recycled, production+neutral_exit),
        'total_mass': residual['total_atoms_s']/q,
        'power': residual['power_W']/max(c['absorbed_power_W'], 1e-300),
    }
    return {'rates': k, 'sound_speed_m_s': cs, 'ion_rates_s': ions,
            'ionization_events_s': production, 'neutral_exit_atoms_s': neutral_exit,
            'wall_recycled_atoms_s': recycled, 'wall_retained_atoms_s': retained,
            'power_budget_W': power, 'residuals': residual, 'relative_residuals': scaled}


def trial_state(te, c, g):
    """Analytically eliminate nn via ion balance, and ne via power balance."""
    cs = math.sqrt(E*te/M_AR)
    nn = cs*sum(g['effective_areas_m2'].values())/(g['volume_m3']*rates(te)['ionization_m3_s'])
    power_per_density = sum(evaluate(1.0, nn, te, c, g)['power_budget_W'].values())
    ne = c['absorbed_power_W']/power_per_density
    state = evaluate(ne, nn, te, c, g)
    return ne, nn, state


def solve(geometry, c):
    """Find a positive steady branch in 1--7 eV; do not claim dynamic stability."""
    validate(c)
    g = geometry_terms(geometry, c)
    base = {'model': MODEL, 'rate_model': RATE_MODEL, 'geometry': g,
            'limitations': list(LIMITATIONS), 'input': copy.deepcopy(c)}
    if c['absorbed_power_W'] == 0:
        # With no background ionization the only supported state is neutral gas.
        nn = g['injected_atoms_s']/g['neutral_exit_conductance_m3_s']
        return dict(base, status='neutral_only', metrics={
            'electron_density_m3': 0.0, 'electron_temperature_eV': None,
            'neutral_density_m3': nn, 'neutral_pressure_Pa': nn*K_B*c['neutral_temperature_K'],
            'propellant_utilization': 0.0, 'ionized_number_fraction': 0.0,
            'exit_ion_rate_s': 0.0, 'exit_equivalent_ion_current_A': 0.0},
            balances={'ion_rates_s': {'side': 0., 'back': 0., 'exit': 0.},
                      'neutral_exit_atoms_s': g['injected_atoms_s'],
                      'power_budget_W': {k: 0. for k in ('ionization', 'excitation',
                          'elastic_gas_heating', 'side_wall', 'back_wall', 'exit')},
                      'relative_residuals': {'ion': 0., 'neutral': 0., 'total_mass': 0., 'power': 0.}},
            diagnostics={'note': 'Zero absorbed power: no sustained plasma; Te is undefined.'})

    def residual(t):
        return trial_state(t, c, g)[2]['relative_residuals']['total_mass']

    # A bounded scan also detects multiple sign-changing roots rather than hiding them.
    grid = [TE_MIN_EV+(TE_MAX_EV-TE_MIN_EV)*i/256 for i in range(257)]
    values = [residual(t) for t in grid]
    roots = [t for t, f in zip(grid, values) if f == 0]
    for a, b, fa, fb in zip(grid, grid[1:], values, values[1:]):
        if fa*fb >= 0:
            continue
        for _ in range(100):
            mid = (a+b)/2
            fm = residual(mid)
            if abs(fm) < 1e-12:
                a = b = mid
                break
            if fa*fm < 0:
                b = mid
            else:
                a, fa = mid, fm
        roots.append((a+b)/2)
    if not roots:
        raise NoSteadyStateError(
            'No positive steady solution in the supported 1--7 eV rate interval. '
            f'Mass residual/feed at 1 and 7 eV: {values[0]:.5g}, {values[-1]:.5g}. '
            'This is not an ignition/extinction prediction. Check power, flow and loss '
            'assumptions, or supply a validated broader-range chemistry model; do not extrapolate these fits.')
    if len(roots) != 1:
        raise NoSteadyStateError('Multiple steady roots found; branch/stability selection is not implemented')
    te = roots[0]
    ne, nn, state = trial_state(te, c, g)
    if max(abs(x) for x in state['relative_residuals'].values()) > 1e-9:
        raise ArithmeticError('Plasma conservation checks failed')
    ion = state['ion_rates_s']
    debye = math.sqrt(EPS_0*te/(E*ne))
    warnings = []
    if debye/min(g['radius_m'], g['length_m']) > .01:
        warnings.append('Debye length exceeds 1% of the smaller chamber dimension; quasineutral bulk approximation is questionable.')
    if min(te-TE_MIN_EV, TE_MAX_EV-te) < .1:
        warnings.append('Electron temperature is within 0.1 eV of a rate-fit boundary.')
    return dict(base, status='steady', balances=state, metrics={
        'electron_density_m3': ne, 'electron_temperature_eV': te,
        'neutral_density_m3': nn, 'neutral_pressure_Pa': nn*K_B*c['neutral_temperature_K'],
        'ionized_number_fraction': ne/(ne+nn),
        'propellant_utilization': ion['exit']/g['injected_atoms_s'],
        'exit_ion_rate_s': ion['exit'], 'exit_ion_mass_flow_kg_s': ion['exit']*M_AR,
        'exit_equivalent_ion_current_A': ion['exit']*E,
        'side_ion_loss_fraction': ion['side']/sum(ion.values()),
        'back_ion_loss_fraction': ion['back']/sum(ion.values()),
        'exit_ion_loss_fraction': ion['exit']/sum(ion.values()),
        'wall_retained_feed_fraction': state['wall_retained_atoms_s']/g['injected_atoms_s'],
        'neutral_exit_feed_fraction': state['neutral_exit_atoms_s']/g['injected_atoms_s'],
        'absorbed_energy_per_exit_ion_eV': c['absorbed_power_W']/(E*ion['exit']),
    }, diagnostics={'debye_length_m': debye, 'warnings': warnings,
                    'root_count': len(roots), 'temperature_interval_eV': [TE_MIN_EV, TE_MAX_EV],
                    'solver': 'analytic elimination and bracketed bisection',
                    'magnetic_coupling': 'none: prescribed boundary-to-volume-mean density ratios'})


def run_case(geometry, config, out, plots=True):
    from .plasma_report import render
    validate(config)
    geometry_terms(geometry, config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    record = {'model': MODEL, 'status': 'running', 'started_utc': utc_now(),
              'geometry_sha256': config_digest(geometry), 'config_sha256': config_digest(config)}
    write_json(out/'run.json', record)
    try:
        write_json(out/'geometry.json', geometry)
        write_json(out/'plasma.json', config)
        record['source'] = source_record(out)
        # Include the model specification and light dependency recipe in the snapshot.
        for name in ('docs/plasma-model.md', 'requirements-plasma.txt'):
            target = out/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
            record['source']['sha256'][name] = digest(target)
        result = solve(geometry, config)
        write_json(out/'plasma_summary.json', result)
        record['plot_runtime'] = render(out, result, plots=plots)
        record['status'] = 'completed'
        record['solution_status'] = result['status']
        record['artifacts_sha256'] = {str(p.relative_to(out)): digest(p)
                                     for p in sorted(out.iterdir()) if p.is_file() and p.name != 'run.json'}
        return result
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['finished_utc'] = utc_now()
        write_json(out/'run.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Plasma inputs (default: config/plasma_argon.json)')
    parser.add_argument('--geometry', type=Path, help='Geometry JSON, optionally from a magnetic run')
    parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwritten')
    parser.add_argument('--absorbed-power', type=float, metavar='W', help='Override absorbed plasma power, not generator power')
    parser.add_argument('--mass-flow-mg-s', type=float, help='Override argon feed in mg/s')
    parser.add_argument('--no-plots', action='store_true', help='Only JSON/CSV/text/HTML; no matplotlib required')
    args = parser.parse_args()
    try:
        c = load(args.config)
        if args.absorbed_power is not None:
            c['absorbed_power_W'] = args.absorbed_power
        if args.mass_flow_mg_s is not None:
            c['mass_flow_kg_s'] = args.mass_flow_mg_s*1e-6
        result = run_case(load_geometry(args.geometry), c, args.out, plots=not args.no_plots)
    except (ValueError, OSError, ArithmeticError, ImportError) as exc:
        print(f'Plasma run failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps({'solution_status': result['status'], **result['metrics'],
                      'report': str((args.out/'report.html').resolve()),
                      'note': 'Illustrative 0D result with prescribed loss ratios; not experimental validation.'}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
