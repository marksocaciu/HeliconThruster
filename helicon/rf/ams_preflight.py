"""Small AMS compatibility test; never opens a user RF mesh.

Run with the same complex PETSc and MPI executable used for RF.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='ams_preflight.json')
    args=parser.parse_args()
    import numpy as np
    import dolfinx, basix, basix.ufl, ufl, petsc4py
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import mesh, fem
    from dolfinx.fem import petsc as fp
    comm=MPI.COMM_WORLD
    report={'python':sys.executable, 'petsc_dir':os.environ.get('PETSC_DIR'),
            'petsc_version':list(PETSc.Sys.getVersion()),'petsc4py':petsc4py.__version__,
            'dolfinx':dolfinx.__version__,'basix':basix.__version__,
            'scalar_dtype':str(np.dtype(PETSc.ScalarType)), 'mpi_ranks':comm.size,
            'packages':{name:bool(PETSc.Sys.hasExternalPackage(name)) for name in ('hypre','mumps')},
            'pc_interfaces':{name:hasattr(PETSc.PC,name) for name in (
                'setHYPREType','setHYPREDiscreteGradient','setHYPRESetInterpolations')},
            'tests':[], 'scope':'Tiny uniform positive curl-curl plus mass; no RF coefficients or boundary constraints.'}
    def save():
        error=None
        if comm.rank==0:
            try: Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
            except OSError as exc: error=str(exc)
        error=comm.bcast(error,root=0)
        if error: raise RuntimeError(error)
    save()
    if not report['packages']['hypre'] or not all(report['pc_interfaces'].values()):
        report['status']='ams_backend_unavailable'; save()
        if comm.rank==0: print(json.dumps(report,indent=2))
        return 2
    domain=mesh.create_unit_cube(comm,2,2,2)
    for degree in (1,2):
        row={'degree':degree}; report['tests'].append(row); save()
        objects=[]
        try:
            V=fem.functionspace(domain,basix.ufl.element('N1curl',domain.basix_cell(),degree))
            Q=fem.functionspace(domain,('Lagrange',degree))
            W=fem.functionspace(domain,('Lagrange',degree,(3,)))
            G=fp.discrete_gradient(Q,V); objects.append(G); G.assemble()
            Pi=fp.interpolation_matrix(W,V); objects.append(Pi); Pi.assemble()
            row.update(gradient_shape=list(G.getSize()),interpolation_shape=list(Pi.getSize()),
                       nodal_block_size=W.dofmap.index_map_bs)
            u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
            form=(ufl.inner(ufl.curl(u),ufl.curl(v))+ufl.inner(u,v))*ufl.dx
            A=fp.assemble_matrix(fem.form(form)); objects.append(A); A.assemble()
            exact=A.createVecRight(); objects.append(exact)
            lo,hi=exact.getOwnershipRange()
            exact.array[:]=np.sin(np.arange(lo,hi)+1)
            rhs=A.createVecLeft(); objects.append(rhs); A.mult(exact,rhs)
            x=exact.duplicate(); objects.append(x); x.set(0)
            ksp=PETSc.KSP().create(comm); objects.append(ksp)
            ksp.setOperators(A); ksp.setType('fgmres'); ksp.setTolerances(rtol=1e-8,max_it=100)
            ksp.setErrorIfNotConverged(False)
            ksp.setOptionsPrefix('ams_preflight_')
            pc=ksp.getPC(); pc.setType('hypre'); pc.setHYPREType('ams')
            pc.setHYPREDiscreteGradient(G)
            pc.setHYPRESetInterpolations(3,ND_Pi_Full=Pi)
            opts=PETSc.Options('ams_preflight_')
            settings={'pc_hypre_ams_cycle_type':1,'pc_hypre_ams_max_iter':1,'pc_hypre_ams_tol':0.0}
            try:
                for key,value in settings.items():opts[key]=value
                ksp.setFromOptions(); ksp.solve(rhs,x)
            finally:
                for key in settings: del opts[key]
            work=rhs.duplicate();objects.append(work);A.mult(x,work);work.axpy(-1,rhs)
            relative=float(work.norm()/max(rhs.norm(),1e-300))
            row.update(reason=int(ksp.getConvergedReason()),iterations=ksp.getIterationNumber(),
                       true_relative_residual=relative if np.isfinite(relative) else None,
                       passed=bool(ksp.getConvergedReason()>0 and np.isfinite(relative) and relative<=1e-8))
        except Exception as exc:
            row.update(passed=False,error=str(exc))
        finally:
            for obj in reversed(objects):obj.destroy()
        save()
    report['status']='tiny_ams_tests_passed' if all(t.get('passed') for t in report['tests']) else 'tiny_ams_tests_failed'
    report['remaining_checks']=['RF boundary constraints and gradient compatibility',
        'curl(gradient) and interpolation consistency for actual element conventions',
        'coarse RF P residual reduction and setup/peak-memory measurement under 24 GB cap',
        'symmetric cycle verification before using AMS inside CG']
    save()
    if comm.rank==0:print(json.dumps(report,indent=2))
    return 0 if report['status']=='tiny_ams_tests_passed' else 2

if __name__=='__main__':
    raise SystemExit(main())
