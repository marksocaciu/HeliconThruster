"""MPI launch/acceptance contracts; no distributed Maxwell solve is claimed."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from helicon.records import write_json, read_json
from helicon.rf.spatial_coupled import run_rf, _launch, couple
import test_spatial_coupling as fixtures


class MPILaunchContracts(unittest.TestCase):
    def result(self):
        return {'mpi_ranks': 4, 'input_power_W': 10.,
                'material_absorption_W': {'PLASMA': 7.},
                'numerical_checks_passed': True,
                'numerical_checks': {key: True for key in (
                    'finite_power_values', 'positive_input_power', 'power_balance',
                    'free_equation_residual', 'nonnegative_material_absorption',
                    'nonnegative_pml_absorption', 'nonnegative_outward_interface_flux',
                    'pml_interface_flux_agreement', 'interface_trace_agreement')}}

    def test_mpi_results_retain_all_checks_and_validate_rank_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'rf'; out.mkdir()
            result=self.result()
            write_json(out/'rf_summary.json', result)
            with patch('helicon.rf.spatial_coupled._launch') as launch:
                actual=run_rf(root/'build',out,root/'plasma.npz',1e-3,4)
                self.assertEqual(actual,result)
                command,log=launch.call_args.args
                self.assertEqual(command[:4],['mpiexec','-n','4',sys.executable])
                self.assertIn('--expected-ranks',command)
                self.assertIn('--plasma-map',command)
                self.assertEqual(log,root/'rf.mpi.log')
                result['mpi_ranks']=1
                write_json(out/'rf_summary.json',result)
                with self.assertRaisesRegex(RuntimeError,'rank count'):
                    run_rf(root/'build',out,root/'plasma.npz',1e-3,4)
                result['mpi_ranks']=4
                result['numerical_checks']['power_balance']=False
                write_json(out/'rf_summary.json',result)
                with self.assertRaisesRegex(RuntimeError,'power_balance'):
                    run_rf(root/'build',out,root/'plasma.npz',1e-3,4)

    def test_child_failure_is_preserved_as_error_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'rf.log'
            with patch('helicon.rf.spatial_coupled.subprocess.run',return_value=SimpleNamespace(returncode=3)):
                with self.assertRaisesRegex(RuntimeError,'code 3'):
                    _launch(['mpi-launcher'],log)
            self.assertTrue(log.exists())

    def test_child_rank_threads_do_not_inherit_transport_thread_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict('os.environ', {'OPENBLAS_NUM_THREADS':'6', 'OMP_NUM_THREADS':'6'}):
                with patch('helicon.rf.spatial_coupled.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
                    _launch(['mpi-launcher'],Path(tmp)/'rf.log')
                    self.assertEqual(run.call_args.kwargs['env']['OPENBLAS_NUM_THREADS'],'1')
                    self.assertEqual(run.call_args.kwargs['env']['OMP_NUM_THREADS'],'1')
                import os
                self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'],'6')

    def test_missing_child_summary_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('helicon.rf.spatial_coupled._launch'):
                with self.assertRaisesRegex(RuntimeError,'no summary'):
                    run_rf(root,root/'rf',root/'state.npz',1e-3,4)

    def test_multi_controller_launch_is_rejected_before_mpi_import(self):
        with patch.dict('os.environ', {'OMPI_COMM_WORLD_SIZE':'4'}):
            with patch('helicon.rf.solve.runtime',side_effect=AssertionError('must not initialize parent MPI')):
                with self.assertRaisesRegex(RuntimeError,'one controller'):
                    couple('build','out',100,rf_ranks=4)


class MPIControllerTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.SpatialCoupling.setUpClass()
        cls.fixture = fixtures.SpatialCoupling()

    # Inherit the existing synthetic-RF / real-transport tests, and exercise the
    # parallel-dispatch controller without initializing MPI in the parent.
    def test_parallel_controller_preflights_children_without_parent_mpi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); build,config=self.fixture.prepare(root)
            with patch('helicon.rf.solve.runtime',side_effect=AssertionError('parent MPI initialized')), \
                 patch('helicon.rf.spatial_coupled.shutil.which',return_value='/usr/bin/mpiexec'), \
                 patch('helicon.rf.spatial_coupled._launch') as launch, \
                 patch('helicon.rf.spatial_coupled.run_rf',side_effect=self.fixture.fake_rf) as worker:
                result=couple(build,root/'out',100/.75,config,rf_ranks=4,max_iterations=3)
            self.assertEqual(result['status'],'converged')
            self.assertIn('--check-runtime',launch.call_args.args[0])
            self.assertTrue(all(call.kwargs['rf_ranks']==4 for call in worker.call_args_list))
            self.assertEqual(read_json(root/'out/run.json')['rf_ranks'],4)

    def test_later_rf_failure_retains_accepted_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);build,config=self.fixture.prepare(root)
            calls=[]
            def rf(*args,**kwargs):
                calls.append(1)
                if len(calls)==3: raise RuntimeError('rejected second iteration')
                return self.fixture.fake_rf(*args,**kwargs)
            with patch('helicon.rf.spatial_coupled.shutil.which',return_value='/usr/bin/mpiexec'), \
                 patch('helicon.rf.spatial_coupled._launch'), \
                 patch('helicon.rf.spatial_coupled.run_rf',side_effect=rf):
                with self.assertRaisesRegex(RuntimeError,'rejected second'):
                    couple(build,root/'out',100/.75,config,rf_ranks=4,max_iterations=3)
            failure=read_json(root/'out/failure.json')
            self.assertEqual(failure['last_accepted']['iteration'],1)
            self.assertEqual(len(failure['history']),1)
            self.assertEqual(failure['failed_stage'],'unit_rf')
            self.assertTrue((root/'out'/failure['last_accepted']['rf_input_plasma']).is_file())


if __name__=='__main__': unittest.main()
