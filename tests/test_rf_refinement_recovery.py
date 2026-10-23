import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from helicon.rf.config import load, validate
from helicon.rf.checkpoints import save_accepted, save_failure
from helicon.rf.coupled import couple, _write_iteration_build
from helicon.records import write_json, digest

class RefinementRecovery(unittest.TestCase):
    def test_optional_refinement_validation(self):
        c=load(); validate(c)
        c['mesh'].update(interface_m=.001, interface_band_m=.003, interface_transition_m=.009)
        validate(c)
        for key,value in [('interface_m',.01),('interface_band_m',.01),('interface_transition_m',float('nan'))]:
            bad=copy.deepcopy(c);bad['mesh'][key]=value
            with self.assertRaises(ValueError): validate(bad)
        del c['mesh']['interface_band_m']
        with self.assertRaises(ValueError): validate(c)

    def test_rejected_candidate_preserves_last_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);save_accepted(p,{'iteration':1,'rf_run':'iterations/iteration-001/rf'})
            before=(p/'last_accepted.json').read_bytes()
            result=save_failure(p,{'model':'test','status':'running'},[{'iteration':1}],RuntimeError('bad RF'),'unit_rf')
            self.assertEqual(result['last_accepted']['iteration'],1)
            self.assertEqual((p/'last_accepted.json').read_bytes(),before)
            self.assertEqual(result['status'],'failed')
            result=save_failure(p,{'model':'test','status':'not_converged'},[],RuntimeError('limit'),'limit')
            self.assertEqual(result['status'],'not_converged')

    def test_global_driver_saves_failure_before_any_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);build=root/'mesh';build.mkdir();c=load();c['material_case']='plasma'
            write_json(build/'rf.json',c);(build/'rf.msh').write_text('fixture')
            mpi=SimpleNamespace(COMM_WORLD=SimpleNamespace(size=1))
            with patch('helicon.rf.solve.runtime',return_value=(mpi,None,None)), patch('helicon.rf.coupled.snapshot',return_value={}), patch('helicon.rf.coupled.global_plasma.solve',side_effect=RuntimeError('no supported steady solution')):
                with self.assertRaisesRegex(RuntimeError,'no supported'):
                    couple(build,root/'out',100.)
            failure=json.loads((root/'out/failure.json').read_text())
            self.assertEqual(failure['failed_stage'],'seed_global_plasma')
            self.assertIsNone(failure['last_accepted'])
            self.assertEqual(failure['history'],[])

    def test_iteration_build_rejects_geometry_change_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);src=p/'src';src.mkdir();c=load()
            write_json(src/'rf.json',c);(src/'rf.msh').write_text('mesh')
            write_json(src/'mesh_report.json',{'mesh_sha256':digest(src/'rf.msh'),'config_sha256':digest(src/'rf.json')})
            target=copy.deepcopy(c);target['plasma']['electron_density_m3']*=2
            _write_iteration_build(src,p/'ok',target)
            target['mesh']['plasma_m']/=2
            with self.assertRaisesRegex(ValueError,'remeshing'): _write_iteration_build(src,p/'bad',target)
            (src/'rf.msh').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'changed'): _write_iteration_build(src,p/'tampered',c)
