import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from helicon.rf.compare import compare
from helicon.rf.solve import _linear_solver_options


class ComparisonTests(unittest.TestCase):
    def test_pass_failure_input_mismatch_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            reference=root/'reference'; candidate=root/'candidate'
            summary={'status':'completed','numerical_checks_passed':True,
                'solver_diagnostics':{'true_relative_residual':1e-12},
                'ideal_gap_impedance_ohm':[1.,-2.], 'input_power_W':0.001,
                'pml_absorption_W':0.,'outward_interface_flux_W':0.,
                'material_absorption_W':{'PLASMA':0.001}}
            for path in (reference,candidate):
                path.mkdir()
                for name in ('rf.msh','rf.json'): (path/name).write_text('same input')
                (path/'rf_summary.json').write_text(json.dumps(summary))
                (path/'run.json').write_text(json.dumps({'memory':[{'cgroup':{'peak':1000000000,'swap.current':0}}]}))
            self.assertTrue(compare(reference,candidate)['passed'])
            summary['ideal_gap_impedance_ohm']=[2.,-2.]
            (candidate/'rf_summary.json').write_text(json.dumps(summary))
            self.assertFalse(compare(reference,candidate)['passed'])
            summary['ideal_gap_impedance_ohm']=[1.,-2.]
            (candidate/'rf_summary.json').write_text(json.dumps(summary))
            (candidate/'rf.msh').write_text('different mesh')
            self.assertIn('Inputs differ: rf.msh',compare(reference,candidate)['failures'])
            (candidate/'rf.msh').write_text('same input')
            self.assertFalse(compare(reference,candidate,job_limit_mb=500)['passed'])
            (candidate/'rf_summary.json').unlink()
            self.assertFalse(compare(reference,candidate)['passed'])

    def test_positive_backend_is_explicit_and_lu_free(self):
        petsc=SimpleNamespace(Sys=SimpleNamespace(hasExternalPackage=lambda _:False))
        options=_linear_solver_options(petsc,2,'fgmres_positive')
        self.assertEqual(options['ksp_type'],'fgmres')
        self.assertEqual(options['ksp_pc_side'],'right')
        self.assertEqual(options['ksp_norm_type'],'unpreconditioned')
        self.assertNotIn('pc_factor_mat_solver_type',options)


@unittest.skipUnless(os.environ.get('HELICON_FEM_DIAGNOSTICS')=='1','opt-in FEM tests')
class PositiveFormTests(unittest.TestCase):
    def test_complex_preconditioner_compiles(self):
        import basix.ufl
        import ufl
        import ffcx.compiler
        import ffcx.options
        from helicon.rf.forms import positive_preconditioner
        from helicon.rf.config import load
        domain=ufl.Mesh(basix.ufl.element('Lagrange','tetrahedron',1,shape=(3,)))
        V=ufl.FunctionSpace(domain,basix.ufl.element('N1curl','tetrahedron',2))
        T=ufl.FunctionSpace(domain,basix.ufl.element('DG','tetrahedron',0,shape=(3,3)))
        u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
        form=positive_preconditioner(u,v,ufl.Coefficient(T),ufl.dx(domain=domain),load())
        header,source=ffcx.compiler.compile_ufl_objects([form],options=ffcx.options.get_options({'scalar_type':'complex128'}))
        self.assertTrue(header); self.assertTrue(source)
        for scale in (0,-1,float('nan')):
            with self.assertRaises(ValueError):
                positive_preconditioner(u,v,ufl.Coefficient(T),ufl.dx(domain=domain),load(),scale)

    def test_separate_positive_matrix_solves_indefinite_known_field(self):
        import numpy as np
        import basix.ufl
        import ufl
        from mpi4py import MPI
        from petsc4py import PETSc
        from dolfinx import mesh,fem
        from dolfinx.fem.petsc import LinearProblem,assemble_matrix
        from helicon.rf.forms import positive_preconditioner
        from helicon.rf.config import load
        from helicon.rf.materials import MU0,EPS_0
        from helicon.rf.solve import _solve_with_options
        from helicon.rf.diagnostics import linear_diagnostics
        domain=mesh.create_unit_cube(MPI.COMM_WORLD,2,2,2)
        c=load(); c['frequency_Hz']=1/(2*np.pi*np.sqrt(MU0*EPS_0))
        for degree in (1,2):
            V=fem.functionspace(domain,basix.ufl.element('N1curl',domain.basix_cell(),degree))
            u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
            E=fem.Function(V)
            E.interpolate(lambda x:np.vstack((1+x[1],x[0]+x[2],2-x[1])).astype(PETSc.ScalarType))
            E.x.scatter_forward()
            a=(ufl.inner(ufl.curl(u),ufl.curl(v))-0.2*ufl.inner(u,v))*ufl.dx
            eps=fem.Constant(domain,np.diag([-2.,3.,4.]).astype(PETSc.ScalarType))
            p=positive_preconditioner(u,v,eps,ufl.dx(domain=domain),c)
            P=assemble_matrix(fem.form(p)); P.assemble()
            work=P.createVecLeft(); P.mult(E.x.petsc_vec,work)
            energy=E.x.petsc_vec.dot(work)
            self.assertGreater(energy.real,0)
            self.assertLess(abs(energy.imag),1e-9)
            work.destroy(); P.destroy()
            problem=LinearProblem(a,ufl.action(a,E),P=p,petsc_options_prefix='positive_test_',petsc_options=None)
            options=_linear_solver_options(PETSc,MPI.COMM_WORLD.size,'fgmres_positive')
            result=_solve_with_options(problem,options,PETSc)
            self.assertTrue(linear_diagnostics(problem,PETSc,1e-8)['accepted'])
            difference=result.x.petsc_vec.copy(); difference.axpy(-1,E.x.petsc_vec)
            self.assertLess(difference.norm()/E.x.petsc_vec.norm(),1e-6)
            difference.destroy(); del problem
