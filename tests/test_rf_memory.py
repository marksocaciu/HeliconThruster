import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from helicon.rf.memory import admission
from helicon.rf.solve import _linear_solver_options


class MemoryAdmission(unittest.TestCase):
    def test_original_failure_is_rejected(self):
        analysis={'estimated_total_mb':26017,'rank_estimated_mb':[14052,11965]}
        result=admission(analysis,3e9,24000,2000,10000)
        self.assertFalse(result['accepted'])
        self.assertEqual(len(result['reasons']),3)
        self.assertEqual(result['projected_job_mb'],31017)

    def test_total_and_per_rank_limits_are_independent(self):
        analysis={'estimated_total_mb':9000,'rank_estimated_mb':[6000,3000]}
        self.assertFalse(admission(analysis,1e9,24000,2000,5000)['accepted'])
        self.assertTrue(admission(analysis,1e9,24000,2000,7000)['accepted'])
        self.assertFalse(admission(analysis,None,24000,2000,7000)['accepted'])
        analysis['estimated_total_mb']=0
        self.assertFalse(admission(analysis,1e9,24000,2000,7000)['accepted'])

    def test_iterative_does_not_require_external_lu(self):
        petsc=SimpleNamespace(Sys=SimpleNamespace(hasExternalPackage=lambda name:False))
        with patch.dict(os.environ,{'HELICON_RF_ILU_LEVELS':'1'}):
            opts=_linear_solver_options(petsc,2,'gmres_asm')
        self.assertEqual(opts['sub_pc_factor_levels'],1)
        self.assertNotIn('pc_factor_mat_solver_type',opts)
        with self.assertRaises(ValueError):
            _linear_solver_options(petsc,2,'gmres_asm',10000)
        with patch.dict(os.environ,{'HELICON_RF_GMRES_RESTART':'0'}), self.assertRaises(ValueError):
            _linear_solver_options(petsc,2,'gmres_asm')


@unittest.skipUnless(os.environ.get('HELICON_FEM_DIAGNOSTICS')=='1','opt-in PETSc/FEM tests')
class NumericalRegression(unittest.TestCase):
    def test_reaction_action_matches_matrix_for_degrees_one_and_two(self):
        import numpy as np
        import basix.ufl
        import ufl
        from mpi4py import MPI
        from petsc4py import PETSc
        from dolfinx import mesh, fem
        from dolfinx.fem.petsc import assemble_matrix
        from helicon.rf.solve import _reaction_vector
        domain=mesh.create_unit_cube(MPI.COMM_WORLD,2,2,2,ghost_mode=mesh.GhostMode.shared_facet)
        for degree in (1,2):
            V=fem.functionspace(domain,basix.ufl.element('N1curl',domain.basix_cell(),degree))
            u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
            # Nonsymmetric anisotropic mass term catches transposition errors.
            tensor=ufl.as_matrix(((2,0.3,0),(0,3,0.2),(0.1,0,4)))
            a=(ufl.inner(ufl.curl(u),ufl.curl(v))-ufl.inner(tensor*u,v))*ufl.dx
            E=fem.Function(V)
            E.interpolate(lambda x:np.vstack((x[1]+x[2],x[0]*x[2],x[0]-x[1])).astype(PETSc.ScalarType))
            E.x.scatter_forward()
            A=assemble_matrix(fem.form(a)); A.assemble()
            expected=A.createVecLeft(); A.mult(E.x.petsc_vec,expected)
            actual=_reaction_vector(a,E,PETSc)
            actual.axpy(-1,expected)
            self.assertLess(actual.norm()/expected.norm(),1e-11)
            actual.destroy(); expected.destroy(); A.destroy()

    def test_gmres_asm_matches_known_solution_and_keeps_nested_options(self):
        import numpy as np
        import ufl
        from mpi4py import MPI
        from petsc4py import PETSc
        from dolfinx import fem, mesh
        from dolfinx.fem.petsc import LinearProblem
        from helicon.rf.solve import _solve_with_options
        domain=mesh.create_unit_cube(MPI.COMM_WORLD,3,3,3)
        V=fem.functionspace(domain,('Lagrange',1))
        u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
        a=(ufl.inner(ufl.grad(u),ufl.grad(v))+ufl.inner(u,v))*ufl.dx
        L=ufl.inner(fem.Constant(domain,PETSc.ScalarType(1)),v)*ufl.dx
        with patch.dict(os.environ,{'HELICON_RF_ILU_LEVELS':'1'}):
            opts=_linear_solver_options(PETSc,MPI.COMM_WORLD.size,'gmres_asm')
        problem=LinearProblem(a,L,petsc_options_prefix='rf_test_',petsc_options=None)
        solved=_solve_with_options(problem,opts,PETSc)
        self.assertGreater(problem.solver.getConvergedReason(),0)
        self.assertLess(np.max(np.abs(solved.x.array-1)),1e-7)
        for sub in problem.solver.getPC().getASMSubKSP():
            self.assertEqual(sub.getPC().getType(),'ilu')
        self.assertFalse(PETSc.Options('rf_test_').hasName('sub_pc_factor_levels'))
        residual=problem.b.duplicate(); problem.A.mult(problem.x,residual); residual.axpy(-1,problem.b)
        self.assertLess(residual.norm()/problem.b.norm(),1e-8)
        residual.destroy()
        del problem

    def test_native_analysis_does_not_factorize(self):
        from mpi4py import MPI
        from petsc4py import PETSc
        from helicon.rf.analysis import analyze_mumps
        comm=MPI.COMM_WORLD
        if not PETSc.Sys.hasExternalPackage('mumps'): self.skipTest('MUMPS unavailable')
        directory=comm.bcast(tempfile.mkdtemp() if comm.rank==0 else None,root=0)
        try:
            A=PETSc.Mat().createAIJ([100,100],nnz=1,comm=comm)
            lo,hi=A.getOwnershipRange()
            # All-zero numeric diagonal: symbolic analysis is possible; LU is singular.
            A.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,False)
            for i in range(lo,hi): A.setValue(i,i,0.)
            A.assemble()
            options={'mat_mumps_icntl_7':4,'mat_mumps_icntl_22':1,
                     'mat_mumps_icntl_23':1000,'mat_mumps_ooc_tmpdir':directory}
            result=analyze_mumps(A,options,comm,directory,PETSc)
            self.assertEqual(result['effective_ordering'],4)
            self.assertGreater(result['estimated_total_mb'],0)
            self.assertEqual(len(result['rank_estimated_mb']),comm.size)
            A.destroy()
        finally:
            comm.barrier()
            if comm.rank==0:
                import shutil
                shutil.rmtree(directory)


class LauncherRegression(unittest.TestCase):
    def test_reuse_iterative_flags_stale_blr_and_failure_exit(self):
        import json
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); binaries=root/'bin'; binaries.mkdir()
            trace=root/'trace.json'
            launcher=Path(__file__).resolve().parents[1]/'run.sh'
            fake_mpi=binaries/'mpiexec'
            fake_mpi.write_text('#!'+sys.executable+'\nimport os,sys,json\nfrom pathlib import Path\n'
                'Path(os.environ["TEST_TRACE"]).write_text(json.dumps({"args":sys.argv[1:], '
                '"blr":os.environ.get("HELICON_RF_MUMPS_BLR_TOL")}))\nsys.exit(17)\n')
            fake_mpi.chmod(0o755)
            fake_free=binaries/'free'; fake_free.write_text('#!/bin/sh\nexit 0\n'); fake_free.chmod(0o755)
            env=dict(os.environ,PATH=str(binaries)+os.pathsep+os.environ['PATH'],
                MEMORY_SCOPE='0',SOLVER='gmres_asm',RANKS='2',BLAS_THREADS='1',
                RUN_ROOT=str(root/'run'),BUILD=str(root/'existing-mesh'),PYTHON=sys.executable,
                TEST_TRACE=str(trace),BLR_TOL='',HELICON_RF_MUMPS_BLR_TOL='0.01',
                ANALYSIS_ONLY='0',JOB_MEMORY_MB='24000',MEM_MB='10000')
            result=subprocess.run(['bash',str(launcher)],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,17,result.stdout+result.stderr)
            recorded=json.loads(trace.read_text())
            self.assertIsNone(recorded['blr'])
            self.assertIn('gmres_asm',recorded['args'])
            self.assertIn(str(root/'existing-mesh'),recorded['args'])
            self.assertNotIn('--mumps-memory-mb',recorded['args'])
            self.assertFalse((root/'run/prepared').exists())
            status=json.loads((root/'run/launcher_status.json').read_text())
            self.assertEqual(status['exit_code'],17)
