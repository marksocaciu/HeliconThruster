import os
import unittest

@unittest.skipUnless(os.environ.get('HELICON_FEM_DIAGNOSTICS')=='1', 'opt-in PETSc test')
class EquilibrationTests(unittest.TestCase):
    def test_inverse_mapping_and_preservation(self):
        import numpy as np
        from petsc4py import PETSc
        from helicon.rf.equilibrated import EquilibratedILU
        A=PETSc.Mat().createAIJ([3,3],nnz=3,comm=PETSc.COMM_SELF)
        values=np.array([[4.,-2.,0.],[-2.,400.,-20.],[0.,-20.,40000.]])
        A.setValues(range(3),range(3),values); A.assemble()
        before=A.norm()
        rhs=A.createVecRight(); rhs.set(1)
        x=rhs.duplicate(); residual=rhs.duplicate()
        context=EquilibratedILU(A,0,'natural',PETSc)
        try:
            context.apply(None,rhs,x)
            A.mult(x,residual); residual.axpy(-1,rhs)
            self.assertLess(residual.norm(),1e-12)
            d=context.matrix.getDiagonal()
            try: np.testing.assert_allclose(d.getArray(),1)
            finally: d.destroy()
            self.assertEqual(A.norm(),before)
        finally:
            context.destroy(None); rhs.destroy(); x.destroy(); residual.destroy(); A.destroy()

    def test_distributed_asm_preserves_system(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {"HELICON_RF_COMPARE_SCALED_ILU":"1"}):
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
            from helicon.rf.ilu import compare_ilu
            opts['sub_pc_factor_levels']=0
            r=compare_ilu(p,opts,PETSc,comm,out,1e-8,[1,0],first)
            assert r['attempts'][0]['pc_probe']['pc_output_norm'] != r['attempts'][1]['pc_probe']['pc_output_norm']
            assert r['A_b_P_unchanged']
            assert all(v['accepted'] and v['pc_probe']['finite'] for v in r['attempts'])
            assert r['attempts'][0]['true_relative_residual']==r['attempts'][3]['true_relative_residual']
            x.axpy(-1,before); assert x.norm()==0
            if comm.rank==0: print('PASS: sweep preserves A, b, original x and repeats scale reproducibly')
            before.destroy(); k.destroy(); x.destroy(); b.destroy(); A.destroy(); P.destroy()
