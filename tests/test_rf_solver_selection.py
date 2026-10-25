import unittest
from unittest.mock import patch
from types import SimpleNamespace
from helicon.rf.solve import _linear_solver_options

class SolverSelection(unittest.TestCase):
    def petsc(self, names):
        return SimpleNamespace(Sys=SimpleNamespace(hasExternalPackage=lambda name:name in names))

    def test_explicit_superlu_overrides_mumps_and_environment(self):
        with patch.dict('os.environ',{'HELICON_RF_LU_SOLVER':'mumps'}):
            options=_linear_solver_options(self.petsc({'mumps','superlu_dist'}),4,'superlu_dist')
        self.assertEqual(options['pc_factor_mat_solver_type'],'superlu_dist')
        self.assertEqual(options['mat_superlu_dist_colperm'],'MMD_AT_PLUS_A')
        self.assertTrue(options['ksp_error_if_not_converged'])

    def test_environment_applies_to_coupling_calls(self):
        with patch.dict('os.environ',{'HELICON_RF_LU_SOLVER':'superlu_dist'}):
            self.assertEqual(_linear_solver_options(self.petsc({'superlu_dist'}),4)['pc_factor_mat_solver_type'],'superlu_dist')

    def test_no_silent_fallback(self):
        with self.assertRaisesRegex(RuntimeError,'not available'):
            _linear_solver_options(self.petsc({'mumps'}),4,'superlu_dist')
        with self.assertRaises(ValueError):
            _linear_solver_options(self.petsc(set()),1,'typo')

    def test_auto_and_serial_fallback(self):
        self.assertEqual(_linear_solver_options(self.petsc({'mumps','superlu_dist'}),4,'auto')['pc_factor_mat_solver_type'],'mumps')
        self.assertNotIn('pc_factor_mat_solver_type',_linear_solver_options(self.petsc(set()),1,'auto'))
        with self.assertRaises(RuntimeError):_linear_solver_options(self.petsc(set()),4,'auto')

class OutOfCoreSelection(unittest.TestCase):
    petsc = SolverSelection.petsc
    def test_ooc_budget_and_scratch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            options=_linear_solver_options(self.petsc({'mumps'}),4,'mumps_ooc',2500,directory)
            self.assertEqual(options['pc_factor_mat_solver_type'],'mumps')
            self.assertEqual(options['mat_mumps_icntl_22'],1)
            self.assertEqual(options['mat_mumps_icntl_23'],2500)
            from pathlib import Path
            self.assertEqual(options['mat_mumps_ooc_tmpdir'],str(Path(directory).resolve()))
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_ooc_environment_and_explicit_precedence(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ',{
                'HELICON_RF_LU_SOLVER':'mumps_ooc','HELICON_RF_MUMPS_MEMORY_MB':'2000',
                'HELICON_RF_MUMPS_SCRATCH':directory}):
            self.assertEqual(_linear_solver_options(self.petsc({'mumps'}),4)['mat_mumps_icntl_23'],2000)
            self.assertEqual(_linear_solver_options(self.petsc({'mumps'}),4,memory_mb=3000)['mat_mumps_icntl_23'],3000)
            self.assertNotIn('mat_mumps_icntl_22',_linear_solver_options(self.petsc({'mumps'}),4,'mumps'))

    def test_ooc_invalid_inputs_fail_early(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            for budget in (0,-1,True,2.5,'bad'):
                with self.assertRaises(ValueError):
                    _linear_solver_options(self.petsc({'mumps'}),4,'mumps_ooc',budget,directory)
            with self.assertRaises(ValueError):
                _linear_solver_options(self.petsc({'mumps'}),4,'mumps_ooc',3000,directory+'/missing')
            with self.assertRaises(ValueError):
                _linear_solver_options(self.petsc({'mumps'}),4,'mumps',3000,directory)
            with self.assertRaises(RuntimeError):
                _linear_solver_options(self.petsc({'superlu_dist'}),4,'mumps_ooc',3000,directory)
