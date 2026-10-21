"""Prepare an independent RF case, or prepare and solve in a complex environment."""
import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
from ..config import ROOT
from ..records import write_json,read_json,digest,utc_now
from .config import load,validate


def snapshot(out):
    dest=Path(out)/'source'; dest.mkdir()
    paths=list((ROOT/'helicon').rglob('*.py'))+[ROOT/'config/rf_nagoya.json']
    for name in ('docs/rf-model.md','requirements-rf.txt', 'docs/spatial-plasma.md',
                 'requirements-spatial.txt', 'config/plasma_spatial_argon.json'):
        if (ROOT/name).exists(): paths.append(ROOT/name)
    hashes={}
    for path in paths:
        relative=path.relative_to(ROOT); target=dest/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target); hashes[str(relative)]=digest(target)
    return {'sha256':hashes,'python':sys.version,'platform':platform.platform()}


def import_plasma(c,path):
    result=read_json(path)
    if result.get('model')!='argon_global_0d_v1' or result.get('status')!='steady':
        raise ValueError('Require a successful, nonzero argon global-model plasma_summary.json')
    h=c['chamber']; g=result['geometry']
    for key in ('radius_m','length_m'):
        if abs(h[key]-g[key])>1e-12: raise ValueError('RF and global-model chamber sizes differ')
    c=copy.deepcopy(c)
    for key in ('electron_density_m3','electron_temperature_eV','neutral_density_m3'):
        c['plasma'][key]=result['metrics'][key]
    validate(c)
    return c


def prepare(c,out,scale=1.,plasma_summary=None,b_map=None):
    from .mesh import generate
    from .materials import AxisymmetricField
    validate(c)
    if plasma_summary: c=import_plasma(c,plasma_summary)
    if (plasma_summary or b_map) and c['material_case']!='plasma':
        raise ValueError('Plasma/B-map inputs require --case plasma')
    if b_map: AxisymmetricField(b_map)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    record={'status':'preparing','started_utc':utc_now()}
    write_json(out/'run.json',record)
    try:
        record['source']=snapshot(out)
        if plasma_summary: shutil.copy2(plasma_summary,out/'input_plasma_summary.json')
        if b_map: shutil.copy2(b_map,out/'background_B.npz')
        record['mesh']=generate(c,out/'mesh',scale)
        record['status']='prepared'
        record['note']='Mesh preparation only: no Maxwell solve has run.'
        record['artifacts_sha256']={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and 'source' not in p.relative_to(out).parts and p!=out/'run.json'}
        return out
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        record['finished_utc']=utc_now();write_json(out/'run.json',record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config');p.add_argument('--case',choices=('vacuum','dielectric','plasma'))
    p.add_argument('--out',required=True);p.add_argument('--mesh-scale',type=float,default=1.)
    p.add_argument('--mesh-only',action='store_true');p.add_argument('--plasma-summary');p.add_argument('--b-map')
    a=p.parse_args();c=load(a.config)
    if a.case: c['material_case']=a.case
    # A full run preflights the runtime before spending time generating a mesh.
    if not a.mesh_only:
        from .solve import runtime
        MPI,_,_=runtime()
        if MPI.COMM_WORLD.size!=1: raise RuntimeError('Prepare with one process, then use mpiexec with helicon.rf.solve')
    out=prepare(c,a.out,a.mesh_scale,a.plasma_summary,a.b_map)
    if a.mesh_only:
        print(f'Prepared mesh only: {out}/mesh/rf.msh. Solve on complex Linux with helicon.rf.solve.');return 0
    from .solve import solve
    result=solve(out/'mesh',out/'solution',out/'background_B.npz' if a.b_map else None)
    from .report import render
    render(out/'solution')
    print(json.dumps(result,indent=2))
    return 0 if result['numerical_checks_passed'] else 2

if __name__=='__main__': sys.exit(main())
