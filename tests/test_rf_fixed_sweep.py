import os
import unittest

@unittest.skipUnless(os.environ.get("HELICON_FEM_DIAGNOSTICS")=="1", "opt-in PETSc test")
class FixedSweepTests(unittest.TestCase):
    def test_preserves_system_and_repeats_scale(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from mpi4py import MPI
        from petsc4py import PETSc
        from helicon.rf.repeat import repeat_fixed_system
        from helicon.rf.solve import _solve_with_options
        from helicon.rf.diagnostics import linear_diagnostics
        comm=MPI.COMM_WORLD
        A=PETSc.Mat().createAIJ([20,20],nnz=3,comm=comm)
        lo,hi=A.getOwnershipRange()
        for i in range(lo,hi):
         A[i,i]=4
         if i: A[i,i-1]=-1
         if i<19: A[i,i+1]=-1
        A.assemble(); P=A.copy()
        b=A.createVecRight(); b.set(1)
        x=b.duplicate(); x.set(0)
        k=PETSc.KSP().create(comm); k.setOptionsPrefix('test_'); k.setOperators(A,P)
        p=SimpleNamespace(A=A,b=b,x=x,solver=k,solve=lambda:k.solve(b,x))
        opts={'ksp_type':'fgmres','pc_type':'asm','sub_pc_type':'ilu','ksp_rtol':1e-10}
        _solve_with_options(p,opts,PETSc)
        first=linear_diagnostics(p,PETSc,1e-8); before=x.copy()
        out=Path(comm.bcast(tempfile.mkdtemp() if comm.rank==0 else None,root=0))
        def rebuild(matrix,scale):
         matrix.zeroEntries()
         for i in range(lo,hi):
          matrix[i,i]=4+scale
          if i: matrix[i,i-1]=-1
          if i<19: matrix[i,i+1]=-1
         matrix.assemble()
        r=repeat_fixed_system(p,opts,PETSc,comm,out,1e-8,3,first,scales=[2,4,2],rebuild=rebuild,initial_scale=0)
        assert r['A_and_b_unchanged']
        assert all(v['accepted'] for v in r['attempts'])
        assert r['attempts'][1]['true_relative_residual']==r['attempts'][3]['true_relative_residual']
        x.axpy(-1,before); assert x.norm()==0
        if comm.rank==0: print('PASS: sweep preserves A, b, original x and repeats scale reproducibly')
        before.destroy(); k.destroy(); x.destroy(); b.destroy(); A.destroy(); P.destroy()
