"""Persistent real-PETSc AMS worker. Started only by ams_bridge."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from .ams_protocol import atomic_json, atomic_npz, wait_json, cgroup_identity
from .ams_transfer import collective_io, sha
from .memory import memory_snapshot, cgroup_memory


def main():
    from mpi4py import MPI
    from petsc4py import PETSc
    comm=MPI.COMM_WORLD
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',required=True);parser.add_argument('--ipc',required=True)
    parser.add_argument('--rtol',type=float,required=True);parser.add_argument('--max-it',type=int,required=True)
    parser.add_argument('--timeout',type=float,default=600);parser.add_argument('--cap-mb',type=int)
    args=parser.parse_args(); source=Path(args.export); ipc=Path(args.ipc)
    objects=[]; settings={}; opts=None
    report={'status':'starting','requests':0}
    def root_write(name,data):collective_io(comm,lambda:atomic_json(ipc/name,data) if comm.rank==0 else None)
    try:
        if not 0<args.rtol<1 or not 1<=args.max_it<=1000 or args.timeout<=0:raise ValueError('Invalid worker bounds')
        if np.issubdtype(PETSc.ScalarType,np.complexfloating) or not PETSc.Sys.hasExternalPackage('hypre'):
            raise RuntimeError('Worker requires real PETSc with hypre')
        config=collective_io(comm,lambda:json.loads((ipc/'config.json').read_text()))
        if args.cap_mb is not None:
            cap=cgroup_memory()
            ok=cap.get('max') is not None and cap['max']<=args.cap_mb*1000000 and cap.get('swap.max')==0
            ok=ok and config['cgroup_identity']==cgroup_identity()
            if not all(comm.allgather(ok)):raise RuntimeError('Worker must share the verified parent memory cgroup')
        manifest=collective_io(comm,lambda:json.loads((source/'manifest.json').read_text()))
        if manifest['format']!=1 or manifest['ranks']!=comm.size:raise ValueError('Export MPI layout mismatch')
        def verify():
            for item in manifest['files_by_rank'][comm.rank]:
                if sha(source/item['file'])!=item['sha256']:raise ValueError('Export checksum mismatch')
        collective_io(comm,verify)
        def load(name):
            def read():
                with np.load(source/f'{name}-{comm.rank}.npz') as data:return {key:data[key] for key in data.files}
            data=collective_io(comm,read)
            m,n=map(int,data['size']);lm,ln=map(int,data['local_size'])
            matrix=PETSc.Mat().createAIJ(size=((lm,m),(ln,n)),comm=comm,
                csr=(data['ia'].astype(PETSc.IntType),data['ja'].astype(PETSc.IntType),data['values']))
            objects.append(matrix);matrix.assemble()
            ok=tuple(matrix.getOwnershipRange())==tuple(data['rows'])
            if not all(comm.allgather(ok)):raise ValueError('Export ownership changed')
            return matrix
        start=time.monotonic();P,G,Pi=load('P'),load('G'),load('Pi')
        ksp=PETSc.KSP().create(comm);objects.append(ksp)
        ksp.setOptionsPrefix('rf_bridge_ams_');ksp.setOperators(P);ksp.setType('fgmres')
        ksp.setTolerances(rtol=args.rtol,atol=0.,max_it=args.max_it)
        ksp.setInitialGuessNonzero(False);ksp.setErrorIfNotConverged(False)
        pc=ksp.getPC();pc.setType('hypre');pc.setHYPREType('ams')
        pc.setHYPREDiscreteGradient(G);pc.setHYPRESetInterpolations(3,ND_Pi_Full=Pi)
        opts=PETSc.Options('rf_bridge_ams_')
        settings={'pc_hypre_ams_cycle_type':1,'pc_hypre_ams_max_iter':1,'pc_hypre_ams_tol':0.0}
        for key,value in settings.items():opts[key]=value
        ksp.setFromOptions();ksp.setUp()
        rhs=P.createVecLeft();x=P.createVecRight();res=rhs.duplicate();objects.extend([rhs,x,res])
        report.update(status='ready',setup_and_load_s=time.monotonic()-start,
                      scalar_dtype=str(np.dtype(PETSc.ScalarType)),petsc_version=list(PETSc.Sys.getVersion()),
                      manifest_sha256=sha(source/'manifest.json'),cgroup_identity=cgroup_identity(),
                      memory=memory_snapshot(comm,'worker_ready'))
        root_write('ready.json',report)
        sequence=1
        while True:
            command=None;error=None
            if comm.rank==0:
                try:command=wait_json(ipc/'command.json',args.timeout,sequence=sequence)
                except Exception as exc:error=str(exc)
            error=comm.bcast(error,root=0)
            if error:raise RuntimeError(error)
            command=comm.bcast(command,root=0)
            if command.get('command')=='stop':break
            if command.get('command')!='solve':raise ValueError('Unknown bridge command')
            def read_rhs():
                with np.load(ipc/f'rhs-{comm.rank}.npz') as data:
                    if int(data['sequence'])!=sequence:raise ValueError('Stale RHS')
                    values=data['values'].copy()
                if values.shape!=(rhs.getLocalSize(),) or not np.all(np.isfinite(values)):raise ValueError('Invalid RHS shard')
                return values
            incoming=collective_io(comm,read_rhs);parts=[];answer=np.zeros(incoming.shape,dtype=np.complex128)
            start=time.monotonic()
            for index,part in enumerate(('real','imag')):
                rhs.array[:]=incoming.real if part=='real' else incoming.imag;x.set(0)
                bn=float(rhs.norm())
                if bn:ksp.solve(rhs,x)
                P.mult(x,res);res.axpy(-1,rhs);rn=float(res.norm())
                reason=int(ksp.getConvergedReason()) if bn else 0
                passed=math.isfinite(rn) and (bn==0 or reason>0) and rn<=args.rtol*bn
                parts.append({'part':part,'rhs_norm':bn,'residual_norm':rn if math.isfinite(rn) else None,
                    'reason':reason,'iterations':int(ksp.getIterationNumber()) if bn else 0,'passed':passed})
                if index==0:answer.real=x.getArray(readonly=True)
                else:answer.imag=x.getArray(readonly=True)
            response={'sequence':sequence,'components':parts,'passed':all(p['passed'] for p in parts),
                      'elapsed_s':time.monotonic()-start,'memory':memory_snapshot(comm,'worker_apply')}
            collective_io(comm,lambda:atomic_npz(ipc/f'answer-{comm.rank}.npz',sequence=sequence,values=answer))
            root_write('response.json',response)
            report['requests']=sequence;sequence+=1
        report.update(status='stopped',memory=memory_snapshot(comm,'worker_stop'))
        root_write('stopped.json',report)
        return 0
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        if comm.rank==0:atomic_json(ipc/'failed.json',report)
        raise
    finally:
        for obj in reversed(objects):obj.destroy()
        if opts is not None:
            for key in settings:del opts[key]

if __name__=='__main__':raise SystemExit(main())
