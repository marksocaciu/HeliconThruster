import os, unittest

@unittest.skipUnless(os.environ.get("HELICON_FEM_DIAGNOSTICS")=="1", "opt-in real PETSc AMS test")
class AMSMainTests(unittest.TestCase):
    def test_field_handoff(self):
        import tempfile,sys
        from pathlib import Path
        import numpy as np
        import ufl,basix.ufl
        from mpi4py import MPI
        from petsc4py import PETSc
        from dolfinx import fem,mesh
        from dolfinx.fem.petsc import LinearProblem
        from helicon.rf.ams_main import solve_ams
        from helicon.rf.diagnostics import linear_diagnostics
        from helicon.rf.solve import _linear_solver_options
        comm=MPI.COMM_WORLD
        D=mesh.create_unit_cube(comm,2,2,2)
        V=fem.functionspace(D,basix.ufl.element('N1curl',D.basix_cell(),2))
        D.topology.create_connectivity(2,3)
        fixed=fem.locate_dofs_topological(V,2,mesh.exterior_facet_indices(D.topology))
        bc=fem.dirichletbc(fem.Function(V),fixed)
        u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
        a=(ufl.inner(ufl.curl(u),ufl.curl(v))+ufl.inner(u,v))*ufl.dx
        L=ufl.inner(fem.Constant(D,np.array([1,2,3],dtype=PETSc.ScalarType)),v)*ufl.dx
        p=LinearProblem(a,L,P=a,bcs=[bc],petsc_options_prefix='main_test_')
        out=Path(comm.bcast(tempfile.mkdtemp() if comm.rank==0 else None,root=0))
        options={'ksp_type':'fgmres','pc_type':'python','ksp_rtol':1e-9,'ksp_max_it':100,'ksp_pc_side':'right','ksp_norm_type':'unpreconditioned','ksp_error_if_not_converged':False}
        assert _linear_solver_options(PETSc,comm.size,'fgmres_ams')['pc_type']=='python'
        field,report=solve_ams(p,options,PETSc,comm,out,V,2,fixed,cap_mb=None,python=sys.executable,mpiexec=str(Path(sys.executable).parent/'mpiexec'),petsc_dir=sys.prefix)
        d=linear_diagnostics(p,PETSc,1e-8)
        assert d['accepted'],d
        assert np.linalg.norm(field.x.array)>0
        assert np.allclose(field.x.array[:p.x.getLocalSize()],p.x.array)
        assert report['worker_shutdown']['returncode']==0
        if comm.rank==0:print('PASS main LinearProblem field handoff',d)
