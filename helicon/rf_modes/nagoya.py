"""Compare modal truncations with a 3D prescribed-current Nagoya III reference."""
import argparse
import csv
import os
import platform
import time
from pathlib import Path
import numpy as np
import scipy

from ..config import ROOT
from ..records import write_json, digest, utc_now
from ..rf.materials import cold_epsilon
from .meridian import solve_mode
from .nagoya_prism import solve_prisms
from .nagoya_source import Sheet


def setup(c, counts, enclosure, current, *, uniform_plasma=True):
    h, a, p = c['chamber'], c['antenna'], c['plasma']
    rp = h['radius_m']
    rw = rp+h['wall_thickness_m']
    ra = a['inner_radius_m']+a['radial_thickness_m']/2
    bounds = [0., rp, rw, ra, enclosure]
    if (min(np.diff(bounds)) <= 0 or (uniform_plasma and p['uniform_B_T'][:2] != [0., 0.])
            or a['ring_width_m'] >= h['length_m']/2):
        raise ValueError('Require nested radial surfaces, an axial B field, and distinct end rings')
    edges = np.r_[0., np.concatenate([np.linspace(lo, hi, n+1)[1:]
                   for lo, hi, n in zip(bounds, bounds[1:], counts)])]
    center = (edges[:-1]+edges[1:])/2
    eps = np.tile(np.eye(3, dtype=complex), (len(center), 1, 1))
    if uniform_plasma:
        eps[center<rp] = cold_epsilon(c['frequency_Hz'], p['electron_density_m3'],
                                      p['collision_frequency_Hz'], p['uniform_B_T'])
    wall = c['dielectric']
    eps[(rp<center)&(center<rw)] *= wall['wall_epsilon_r']*(1+1j*wall['wall_loss_tangent'])
    sheet = Sheet(ra, h['length_m'], a['leg_angle_rad'], a['ring_width_m'], current)
    case = dict(radius_m=enclosure, length_m=h['length_m'], z_start_m=h['z_start_m'], frequency_Hz=c['frequency_Hz'])
    return case, edges, eps, sheet


def power_checks(stats, watts):
    work = stats['source_work_W']
    balance = abs(work-watts.sum())/max(abs(work), 1e-300)
    stats.update(volume_loss_W=float(watts.sum()), power_balance_relative_error=float(balance))
    return bool(work > 0 and np.isfinite(watts).all() and np.all(watts >= -1e-14*work)
                and balance < 1e-7 and stats['free_relative_residual'] < 1e-7)


def differences(watts, reference):
    """Integrated power and distance between unit-integral profile shapes."""
    total, target = watts.sum(), reference.sum()
    if min(total, target) <= 0:
        raise ValueError('Degenerate plasma heating map')
    difference = watts/total-reference/target
    return dict(power_relative=float(abs(total/target-1)),
                radial_shape_L1=float(abs(difference.sum(axis=0)).sum()),
                axial_shape_L1=float(abs(difference.sum(axis=1)).sum()),
                rz_shape_L1=float(abs(difference).sum()))


def run(args):
    from ..rf.config import load
    c = load(args.config)
    counts = args.radial_cells
    material_source = getattr(args, 'material_from_rf', None)
    if material_source is not None and not args.modal_only:
        raise ValueError('--material-from-rf requires --modal-only; the prism reference assumes z-invariant material and axial B')
    sectors_list = [] if args.modal_only else sorted(args.sectors)
    if (args.max_m < 1 or args.max_m % 2 != 1 or not all(n>0 for n in counts)
            or args.nz < 3 or min(args.sectors)<4 or any(n%2 for n in args.sectors)
            or len(set(args.sectors)) != len(args.sectors)
            or not np.isfinite([args.outer_radius, args.current]).all() or args.current <= 0):
        raise ValueError('Require positive cells/current, positive odd max-m, and distinct even sector counts >=4')
    case, r, eps, sheet = setup(c, counts, args.outer_radius, args.current,
                                uniform_plasma=material_source is None)
    nz = args.nz
    z = np.linspace(0, sheet.length, nz+1) + case['z_start_m']
    vol = np.pi*np.diff(z)[:, None]*np.diff(r**2)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    record = dict(model='nagoya_prescribed_closed_v1', status='running', started_utc=utc_now(), rf_inputs=c,
                  excitation='prescribed_sheet_current', material_case='uniform_cold_plasma_with_dielectric_wall',
                  boundary='PEC_on_outer_cylinder_and_both_endcaps',
                  current_peak_A=args.current, sheet_radius_m=sheet.radius, outer_radius_m=args.outer_radius,
                  nz=nz, radial_cells=counts, max_m=args.max_m, sectors=sectors_list,
                  numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                  thread_env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
                  source_hashes={p.name:digest(p) for p in [*Path(__file__).parent.glob('*.py'),
                                ROOT/'helicon/rf/materials.py', ROOT/'helicon/argon.py']},
                  modes=[], references=[], sweep=[])
    write_json(out/'run.json', record)
    try:
        if material_source is not None:
            from .nagoya_material import sample_rf_material
            record['material_case'] = 'rf_input_cold_plasma_with_dielectric_wall'
            eps, provenance = sample_rf_material(material_source, c, r, z, counts[0], eps)
            material_file = out/'material_rz.npz'
            np.savez_compressed(material_file, epsilon_r_cyl=eps, r_faces_m=r, z_faces_m=z)
            record['material_from_rf'] = dict(provenance, file=material_file.name, sha256=digest(material_file))
            record['source_hashes']['spatial_maps.py'] = digest(ROOT/'helicon/rf/spatial_maps.py')
            write_json(out/'run.json', record)
        cumulative = np.zeros((nz, len(r)-1))
        snapshots, elapsed = {}, 0.
        plasma = slice(0, counts[0])
        for high in range(1, args.max_m+1, 2):
            for m in (high, -high):
                start = time.perf_counter()
                fields, stats = solve_mode(case, eps, m, 0., len(r)-1, nz, radial_edges=r, sheet=sheet)
                dt = time.perf_counter()-start
                elapsed += dt
                watts = fields['cell_absorbed_W']
                stats.update(m=m, seconds=dt, passed=power_checks(stats, watts), plasma_W=float(watts[:, plasma].sum()))
                stats['file'] = f'mode_{m:+d}.npz'
                record['modes'].append(stats)
                np.savez_compressed(out/stats['file'], **fields)
                if not stats['passed']:
                    raise ArithmeticError(f'Mode {m} failed residual, passivity or source-work checks')
                cumulative += watts
            snapshots[high] = cumulative[:, plasma].copy()
            record['sweep'].append(dict(max_abs_m=high, modal_seconds=elapsed, plasma_W=float(snapshots[high].sum())))
        np.savez_compressed(out/'heating_rz.npz', r_faces_m=r[:counts[0]+1], z_faces_m=z,
                            cell_volume_m3=vol[:, plasma], q_abs_W_m3=cumulative[:, plasma]/vol[:, plasma],
                            cell_absorbed_W=cumulative[:, plasma])
        reference_maps = {}
        for sectors in sectors_list:
            start = time.perf_counter()
            watts, stats = solve_prisms(case, r, eps, sheet, nz, sectors)
            stats.update(seconds=time.perf_counter()-start, passed=power_checks(stats, watts),
                         plasma_W=float(watts[:, plasma].sum()))
            record['references'].append(stats)
            ref = watts[:, plasma]
            reference_maps[sectors] = ref
            np.savez_compressed(out/f'reference_{sectors}.npz', cell_loss_W=watts, radial_ring_edges_m=r, z_faces_m=z)
            if not stats['passed']:
                raise ArithmeticError(f'3D case {sectors} failed residual, passivity or source-work checks')
        for row in record['sweep']:
            high = row['max_abs_m']
            if reference_maps:
                row['vs_3d'] = differences(snapshots[high], reference_maps[sectors_list[-1]])
            row['vs_last_cutoff'] = differences(snapshots[high], snapshots[args.max_m])
        if len(reference_maps)>1:
            record['angular_refinement'] = differences(reference_maps[sectors_list[-2]], reference_maps[sectors_list[-1]])
        # Common physical bins, integrated watts. No normalization of total power.
        for axis, edges, component in [('radial', r[:counts[0]+1], 0), ('axial', z, 1)]:
            with (out/f'{axis}_profiles.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['lower_m','upper_m',*[f'through_m{m}_W' for m in snapshots],
                                 *[f'prism_{n}_W' for n in reference_maps]])
                writer.writerows(zip(edges[:-1], edges[1:], *[a.sum(axis=component) for a in
                                      (*snapshots.values(), *reference_maps.values())]))
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['finished_utc'] = utc_now()
        write_json(out/'run.json', record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'config/rf_nagoya.json', help='Geometry and material settings')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--current', type=float, default=1., help='Peak forward rail current in A')
    p.add_argument('--outer-radius', type=float, default=.060)
    p.add_argument('--radial-cells', nargs=4, type=int, default=[16, 2, 4, 8],
                   metavar=('PLASMA','WALL','GAP','EXTERIOR'))
    p.add_argument('--nz', type=int, default=80)
    p.add_argument('--max-m', type=int, default=15)
    p.add_argument('--sectors', type=int, nargs='+', default=[64, 128])
    p.add_argument('--modal-only', action='store_true', help='Skip prism reference solves')
    p.add_argument('--material-from-rf', type=Path, metavar='SOLUTION',
                   help='Use the plasma and magnetic inputs of a completed 3D RF solution (requires --modal-only)')
    args = p.parse_args()
    row = run(args)
    for ref in row['references']:
        print(f"3D {ref['sectors']} sectors: {ref['plasma_W']:.7g} W, {ref['seconds']:.2f} s")
    for entry in row['sweep']:
        print(f"|m| <= {entry['max_abs_m']}: {entry['plasma_W']:.7g} W, {entry['modal_seconds']:.2f} s")
    print(args.out/'run.json')


if __name__ == '__main__':
    main()
