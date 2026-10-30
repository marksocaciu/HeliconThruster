"""Test real AMS on exported RF P without reassembling or repartitioning."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from .ams_transfer import sha, collective_io
from .memory import memory_snapshot


def main():
    from mpi4py import MPI
    from petsc4py import PETSc
    comm=MPI.COMM_WORLD
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',required=True);p.add_argument('--out',required=True)
    p.add_argument('--require-cap-mb',type=int)
    p.add_argument('--max-it',type=int,default=100)
    args=p.parse_args()
    if not 1<=args.max_it<=1000:raise ValueError('max-it must be 1..1000')
    if np.issubdtype(PETSc.ScalarType,np.complexfloating) or not PETSc.Sys.hasExternalPackage('hypre'):
        raise RuntimeError('Use real PETSc with hypre')
    if args.require_cap_mb is not None:
        from .memory import cgroup_memory
        cap=cgroup_memory()
        ok=cap.get('max') is not None and cap['max']<=args.require_cap_mb*1000000 and cap.get('swap.max')==0
        if not all(comm.allgather(ok)):raise RuntimeError('Requested memory/swap cap not verified')
    source=Path(args.export);out=Path(args.out)
    if out.exists():raise ValueError('Refusing to overwrite report')
    manifest=collective_io(comm,lambda:json.loads((source/'manifest.json').read_text()))
    if manifest['format']!=1 or manifest['ranks']!=comm.size:raise ValueError('Export format or MPI rank count mismatch')
    def verify():
        for item in manifest['files_by_rank'][comm.rank]:
            if sha(source/item['file'])!=item['sha256']:raise ValueError('Export checksum mismatch: '+item['file'])
    collective_io(comm,verify)
    report={'status':'running','export':str(source.resolve()),'manifest_sha256':sha(source/'manifest.json'),
            'tests':[],'memory':[memory_snapshot(comm,'start')], 'boundary_policy':manifest['boundary_policy']}
    def save():collective_io(comm,lambda:out.write_text(json.dumps(report,indent=2)+'\n') if comm.rank==0 else None)
    save();objects=[]
    try:
        def load(name):
            data=collective_io(comm,lambda:dict(np.load(source/f'{name}-{comm.rank}.npz')))
            m,n=map(int,data['size']);lm,ln=map(int,data['local_size'])
            matrix=PETSc.Mat().createAIJ(size=((lm,m),(ln,n)),
                csr=(data['ia'].astype(PETSc.IntType),data['ja'].astype(PETSc.IntType),data['values']),comm=comm)
            objects.append(matrix);matrix.assemble()
            if tuple(matrix.getOwnershipRange())!=tuple(data['rows']):raise ValueError('Ownership changed')
            return matrix
        P,G,Pi=load('P'),load('G'),load('Pi')
        data=collective_io(comm,lambda:dict(np.load(source/f'rhs-{comm.rank}.npz')))
        ksp=PETSc.KSP().create(comm);objects.append(ksp)
        ksp.setOptionsPrefix('rf_offline_ams_');ksp.setOperators(P);ksp.setType('fgmres')
        ksp.setTolerances(rtol=1e-8,atol=0.,max_it=args.max_it);ksp.setErrorIfNotConverged(False)
        pc=ksp.getPC();pc.setType('hypre');pc.setHYPREType('ams')
        pc.setHYPREDiscreteGradient(G);pc.setHYPRESetInterpolations(3,ND_Pi_Full=Pi)
        opts=PETSc.Options('rf_offline_ams_')
        settings={'pc_hypre_ams_cycle_type':1,'pc_hypre_ams_max_iter':1,'pc_hypre_ams_tol':0.0}
        for key,value in settings.items():opts[key]=value
        start=time.monotonic();ksp.setFromOptions();ksp.setUp()
        report['setup_s']=time.monotonic()-start
        report['memory'].append(memory_snapshot(comm,'setup'));save()
        rhs=P.createVecLeft();x=P.createVecRight();res=rhs.duplicate();objects.extend([rhs,x,res])
        for name in manifest['rhs']:
            rows=[]
            for part in ('real','imag'):
                rhs.array[:]=data[name+'_'+part];x.set(0)
                bn=float(rhs.norm());start=time.monotonic()
                if bn: ksp.solve(rhs,x)
                P.mult(x,res);res.axpy(-1,rhs);rn=float(res.norm())
                reason=int(ksp.getConvergedReason()) if bn else 0
                # Save each owned component for optional complex-side verification.
                collective_io(comm,lambda:np.save(out.parent/f'{out.stem}-{name}-{part}-{comm.rank}.npy',x.getArray(readonly=True)))
                rows.append({'part':part,'rhs_norm':bn,'residual_norm':rn if math.isfinite(rn) else None,
                             'reason':reason,'iterations':int(ksp.getIterationNumber()) if bn else 0,
                             'solve_s':time.monotonic()-start,'zero_rhs':bn==0,
                             'passed':bool(math.isfinite(rn) and (bn==0 or reason>0) and rn<=1e-8*bn)})
            residual=math.hypot(*(r['residual_norm'] if r['residual_norm'] is not None else math.inf for r in rows))
            norm=math.hypot(*(r['rhs_norm'] for r in rows))
            report['tests'].append({'rhs':name,'components':rows,
                'combined_complex_relative_residual':residual/max(norm,1e-300) if math.isfinite(residual) else None,
                'passed':all(r['passed'] for r in rows)})
            report['memory'].append(memory_snapshot(comm,name));save()
        report['status']='passed' if all(r['passed'] for r in report['tests']) else 'not_converged'
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        raise
    finally:
        save()
        for obj in reversed(objects):obj.destroy()
    if comm.rank==0:print(json.dumps(report,indent=2))
    return 0 if report['status']=='passed' else 4

if __name__=='__main__':raise SystemExit(main())
