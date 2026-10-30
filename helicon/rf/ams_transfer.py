"""Portable per-rank real CSR export from a complex PETSc RF run."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def collective_io(comm, action):
    error=None; result=None
    try: result=action()
    except Exception as exc:error=str(exc)
    errors=comm.allgather(error)
    if any(errors):raise RuntimeError(str(errors))
    return result


def save_matrix(matrix, path, comm):
    def save():
        ia,ja,values=matrix.getValuesCSR()
        if not np.all(np.isfinite(values)) or np.any(values.imag != 0):
            raise ValueError('AMS export requires finite exactly real matrix entries')
        np.savez(path,ia=ia,ja=ja,values=values.real,
                 size=matrix.getSize(),local_size=matrix.getLocalSize(),rows=matrix.getOwnershipRange())
        return {'file':path.name,'sha256':sha(path)}
    return collective_io(comm,save)


def export_system(problem, V, degree, constrained, out, comm):
    import ufl
    from mpi4py import MPI
    from dolfinx import fem
    from dolfinx.fem import petsc as fp
    directory=out/'ams-export'
    collective_io(comm,lambda:directory.mkdir(exist_ok=True) if comm.rank==0 else None)
    if (directory/'manifest.json').exists():raise RuntimeError('Refusing to overwrite a completed AMS export')
    A,P=problem.solver.getOperators()
    files=[save_matrix(P,directory/f'P-{comm.rank}.npz',comm)]
    Q=fem.functionspace(V.mesh,('Lagrange',degree))
    W=fem.functionspace(V.mesh,('Lagrange',degree,(3,)))
    checks={}
    for name,make in [('G',lambda:fp.discrete_gradient(Q,V)),('Pi',lambda:fp.interpolation_matrix(W,V))]:
        matrix=make();matrix.assemble()
        try:
            if matrix.getSize()[0]!=P.getSize()[0]:raise ValueError('Auxiliary row layout mismatch')
            if name=='G':
                one=matrix.createVecRight();work=matrix.createVecLeft()
                one.set(1);matrix.mult(one,work)
                checks['gradient_of_constant_norm']=float(work.norm())
                one.array[:]=np.sin(np.arange(*one.getOwnershipRange())+1)
                matrix.mult(one,work)
                gradient=fem.Function(V)
                gradient.x.array[:len(work.getArray())]=work.getArray()
                gradient.x.scatter_forward()
                curl2=comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.curl(gradient),ufl.curl(gradient))*ufl.dx)),op=MPI.SUM)
                grad2=comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(gradient,gradient)*ufl.dx)),op=MPI.SUM)
                checks['sample_curl_gradient_L2_over_gradient_L2']=float(np.sqrt(abs(curl2)/max(abs(grad2),1e-300)))
                one.destroy();work.destroy()
            files.append(save_matrix(matrix,directory/f'{name}-{comm.rank}.npz',comm))
        finally:matrix.destroy()
    residual=problem.b.duplicate()
    try:
        A.mult(problem.x,residual);residual.axpy(-1,problem.b)
        n=V.dofmap.index_map.size_local*V.dofmap.index_map_bs
        local_bc=np.asarray(constrained);local_bc=local_bc[local_bc<n]
        path=directory/f'rhs-{comm.rank}.npz'
        def save_rhs():
            b=problem.b.getArray(readonly=True);r=residual.getArray(readonly=True)
            np.savez(path,b_real=b.real,b_imag=b.imag,residual_real=r.real,residual_imag=r.imag,
                     constrained_local=local_bc)
            return {'file':path.name,'sha256':sha(path)}
        files.append(collective_io(comm,save_rhs))
    finally:residual.destroy()
    manifest={'format':1,'ranks':comm.size,'degree':degree,'vector_nodal_block_size':W.dofmap.index_map_bs,
              'files_by_rank':comm.allgather(files),'checks':checks,
              'rhs':['b','residual'],'boundary_policy':'P retains original constrained rows/columns; G and Pi are unconstrained.',
              'limitations':['Gradient constant check is not a full curl-gradient identity test.',
                            'AMS performance here does not establish RF convergence.']}
    collective_io(comm,lambda:(directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n') if comm.rank==0 else None)
    if comm.rank==0:print('RF AMS export: '+str(directory),flush=True)
