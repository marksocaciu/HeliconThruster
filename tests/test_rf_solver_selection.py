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
