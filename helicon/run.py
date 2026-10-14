"""Run mesh → FEM → plots with a unique folder, logs and a source snapshot."""
import argparse
from pathlib import Path
import subprocess
import sys
from .config import ROOT, load, validate
from .diagnostics import positive
from .records import config_digest, digest, read_json, source_record, utc_now, write_json


def stage(out, manifest, name, arguments):
    command = [sys.executable, '-m', f'helicon.{name}', *map(str, arguments)]
    entry = {'name': name, 'command': command, 'started_utc': utc_now()}
    manifest['stages'].append(entry)
    write_json(out/'run.json', manifest)
    print(f'{out.name}: {name}', flush=True)
    with (out/f'{name}.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    entry.update(returncode=result.returncode, finished_utc=utc_now())
    write_json(out/'run.json', manifest)
    if result.returncode:
        raise RuntimeError(f'{name} failed (exit {result.returncode}); see {out/name}.log')


def run_case(config, out, mesh_scale=1, target_gauss=140, mesh_only=False):
    validate(config)
    positive(mesh_scale, 'mesh scale'); positive(target_gauss, 'target gauss')
    out = Path(out).resolve()
    # An existing directory is never overwritten, even if a previous solve failed.
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'input.json', config)
    manifest = {'schema_version': 1, 'created_utc': utc_now(), 'status': 'running',
                'config_sha256': config_digest(config), 'config': config,
                'mesh_scale': mesh_scale, 'target_gauss': target_gauss,
                'effective_mesh_sizes_m': {k:v*mesh_scale for k,v in config['mesh'].items()},
                'coils': [dict(k, ampere_turns=k['turns']*k['current'],
                              current_density_A_m2=k['turns']*k['current']/(k['radial_thickness']*k['axial_width']))
                          for k in config['coils']],
                'solver': {'unknown':'A_theta','degree':2,'quadrature_degree':6,
                           'permeability':'mu0 in every region','axis_bc':'A_theta=0',
                           'outer_bc':'A_theta=0','ranks':1},
                'stages': []}
    write_json(out/'run.json', manifest)
    try:
        manifest['source'] = source_record(out)
        stage(out, manifest, 'mesh', ['--config',out/'input.json','--out',out,'--scale',mesh_scale])
        manifest['mesh'] = read_json(out/'mesh_report.json')
        manifest['mesh_sha256'] = digest(out/'helicon.msh')
        if mesh_only:
            stage(out, manifest, 'preview', ['--build',out])
            manifest['status'] = 'mesh_only'
        else:
            stage(out, manifest, 'solve', ['--build',out,'--target-gauss',target_gauss])
            manifest['metrics'] = read_json(out/'fem_summary.json')
            manifest['solver_environment'] = read_json(out/'solver_environment.json')
            stage(out, manifest, 'plot', ['--build',out])
            manifest['status'] = 'completed'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        manifest['finished_utc'] = utc_now()
        manifest['artifacts_sha256'] = {p.name:digest(p) for p in out.iterdir()
                                       if p.is_file() and p.name != 'run.json' and not p.name.endswith('.tmp')}
        write_json(out/'run.json', manifest)
    print(f'Saved {out}', flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--out', required=True, help='New run directory; must not already exist')
    p.add_argument('--scale',type=float,default=1)
    p.add_argument('--target-gauss',type=float,default=140)
    p.add_argument('--mesh-only',action='store_true',help='Mac workflow: geometry and preview, no MPI/FEM')
    a = p.parse_args()
    run_case(load(a.config), a.out, a.scale, a.target_gauss, a.mesh_only)

if __name__ == '__main__': main()
