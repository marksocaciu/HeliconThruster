import os
import unittest

@unittest.skipUnless(os.environ.get("HELICON_FEM_DIAGNOSTICS")=="1", "opt-in PETSc test")
class InnerCGTests(unittest.TestCase):
    def test_fixed_system(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from mpi4py import MPI
        from petsc4py import PETSc
        from helicon.rf.repeat import repeat_fixed_system
        from helicon.rf.solve import _solve_with_options
        from helicon.rf.diagnostics import linear_diagnostics
        comm=MPI.COMM_WORLD
        A=PETSc.Mat().createAIJ([20,20],nnz=5,comm=comm)
        lo,hi=A.getOwnershipRange()
        for i in range(lo,hi):
         A[i,i]=6
         if i: A[i,i-1]=-1
         if i<19: A[i,i+1]=-1
        for i in range(lo,hi):
         if i>=5: A[i,i-5]=-1
         if i<15: A[i,i+5]=-1
        A.assemble(); P=A.copy()
        b=A.createVecRight(); b.set(1)
        x=b.duplicate(); x.set(0)
        k=PETSc.KSP().create(comm); k.setOptionsPrefix('test_'); k.setOperators(A,P)
        p=SimpleNamespace(A=A,b=b,x=x,solver=k,solve=lambda:k.solve(b,x))
        opts={'ksp_type':'fgmres','pc_type':'asm','sub_pc_type':'ilu','ksp_rtol':1e-10}
        _solve_with_options(p,opts,PETSc)
        first=linear_diagnostics(p,PETSc,1e-8); before=x.copy()
        out=Path(comm.bcast(tempfile.mkdtemp() if comm.rank==0 else None,root=0))
        from helicon.rf.inner_cg import compare_inner_cg
        r=compare_inner_cg(p,opts,PETSc,comm,out,1e-8,first,0.01,100)
        assert r['A_b_P_unchanged']
        assert r['outer_result']['accepted']
        assert all(v['true_tolerance_met'] for v in r['inner_applications'])
        x.axpy(-1,before); assert x.norm()==0
        if comm.rank==0: print('PASS: sweep preserves A, b, original x and repeats scale reproducibly')
        before.destroy(); k.destroy(); x.destroy(); b.destroy(); A.destroy(); P.destroy()

    def test_limits_and_breakdown(self):
        from mpi4py import MPI
        from petsc4py import PETSc
        from helicon.rf.inner_cg import InnerCG,InnerBreakdown
        comm=MPI.COMM_WORLD
        A=PETSc.Mat().createAIJ([6,6],nnz=3,comm=comm)
        for i in range(*A.getOwnershipRange()):
         A[i,i]=4
         if i: A[i,i-1]=-1
         if i<5:A[i,i+1]=-1
        A.assemble()
        b=A.createVecRight();b.set(1);x=b.duplicate()
        c=InnerCG(A,PETSc,comm,1e-12,1)
        c.apply(None,b,x)
        assert c.rows[-1]['reason']==-3
        assert not c.rows[-1]['true_tolerance_met']
        c.destroy(None)
        A.zeroEntries()
        for i in range(*A.getOwnershipRange()):
         A[i,i]=1
         if i%2==0:A[i,i+1]=2
         else:A[i,i-1]=2
        A.assemble()
        for i in range(*b.getOwnershipRange()): b[i]=1 if i%2==0 else 0
        b.assemble()
        c=InnerCG(A,PETSc,comm,1e-12,10)
        try:
         c.apply(None,b,x)
         raise AssertionError('Expected indefinite CG failure')
        except InnerBreakdown: pass
        assert c.rows[-1]['reason']<0
        c.destroy(None); x.destroy();b.destroy();A.destroy()
        if comm.rank==0:print('PASS inner iteration limit and indefinite breakdown retained')

    def test_nested_failure(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from mpi4py import MPI
        from petsc4py import PETSc
        from helicon.rf.inner_cg import compare_inner_cg
        comm=MPI.COMM_WORLD
        A=PETSc.Mat().createAIJ([6,6],nnz=2,comm=comm)
        P=PETSc.Mat().createAIJ([6,6],nnz=2,comm=comm)
        for i in range(*A.getOwnershipRange()):
         A[i,i]=1; P[i,i]=1; P[i,i+1 if i%2==0 else i-1]=2
        A.assemble();P.assemble()
        b=A.createVecRight()
        for i in range(*b.getOwnershipRange()): b[i]=1 if i%2==0 else 0
        b.assemble(); x=b.copy()
        k=PETSc.KSP().create(comm); k.setOperators(A,P)
        p=SimpleNamespace(A=A,b=b,x=x,solver=k)
        out=Path(comm.bcast(tempfile.mkdtemp() if comm.rank==0 else None,root=0))
        r=compare_inner_cg(p,{'ksp_type':'fgmres','ksp_max_it':5,'ksp_error_if_not_converged':False},PETSc,comm,out,1e-8,{},0.01,10)
        assert r['inner_breakdown']
        assert not r['outer_result']['accepted']
        assert r['A_b_P_unchanged']
        if comm.rank==0:
         assert (out/'inner_cg_diagnostics.json').exists()
         print('PASS nested breakdown saved without accepting outer solve')
        k.destroy(); x.destroy(); b.destroy(); A.destroy(); P.destroy()

