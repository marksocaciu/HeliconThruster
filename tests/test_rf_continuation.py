"""Power-conserving continuation and rejection of an unbalanced intermediate state."""
import copy
import unittest
from unittest.mock import patch
import numpy as np

from helicon import spatial
from helicon.records import read_json
from helicon.rf.continuation import mixed_heating, run
from test_rf_prescribed import PrescribedHeating


class Continuation(unittest.TestCase):
    def setUp(self):
        self.fixture=PrescribedHeating()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture

    def test_every_mixture_conserves_power_with_nonuniform_cell_volumes(self):
        volumes=np.array([[1.,3.],[1.,3.]])
        uniform=np.full((2,2),12.5)
        target=np.array([[1.,1.],[1.,95/3]])
        for alpha in (0.,.1,.5,1.):
            q=mixed_heating(uniform,target,alpha,volumes,100.)
            self.assertAlmostEqual(float(np.sum(q*volumes)),100.)
        np.testing.assert_array_equal(mixed_heating(uniform,target,1.,volumes,100.),target)
        with self.assertRaises(ValueError):mixed_heating(uniform,target,1.1,volumes,100.)
        with self.assertRaises(ValueError):mixed_heating(uniform,target*2,.5,volumes,100.)

    def test_real_transport_reaches_endpoint_and_keeps_inputs_intact(self):
        before=(self.f.source/'run.json').read_bytes()
        out=self.f.root/'continuation'
        result=run(self.f.source,out,100.,self.f.config,steps=2,plots=False)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['last_accepted_alpha'],1.)
        self.assertEqual([r['initialized_from_alpha'] for r in result['history']],[None,0.,.5])
        for row in result['history']:
            self.assertTrue(row['accepted'])
            self.assertAlmostEqual(row['integrated_heating_W'],100.)
        self.assertEqual(before,(self.f.source/'run.json').read_bytes())
        self.assertTrue((out/'steps/step-002/plasma_fields.npz').exists())
        with self.assertRaises(FileExistsError):run(self.f.source,out,100.,self.f.config,steps=2,plots=False)

    def test_rejected_intermediate_candidate_stops_and_preserves_last_accepted(self):
        summary,fields=spatial.solve(self.f.gconfig,self.f.c)
        bad=copy.deepcopy(summary)
        bad['status']='not_converged'
        bad['balances']['max_local_relative_residuals']['current_res']=.47
        with patch('helicon.rf.continuation.spatial.solve',side_effect=[
                (summary,fields),(bad,fields)]) as solver:
            result=run(self.f.source,self.f.root/'rejected',100.,self.f.config,steps=10,plots=False)
        self.assertEqual(solver.call_count,2)
        self.assertEqual(result['status'],'not_converged')
        self.assertEqual(result['failed_alpha'],.1)
        self.assertEqual(result['last_accepted_alpha'],0.)
        self.assertFalse(result['history'][-1]['accepted'])
        self.assertIs(solver.call_args.kwargs['initial'],fields)
        record=read_json(self.f.root/'rejected/continuation_summary.json')
        self.assertEqual(record['last_accepted_directory'],'steps/step-000')
        self.assertTrue((self.f.root/'rejected/steps/step-001/spatial_summary.json').exists())
        self.assertFalse((self.f.root/'rejected/steps/step-002').exists())

    def test_transport_exception_retains_progress(self):
        summary,fields=spatial.solve(self.f.gconfig,self.f.c)
        out=self.f.root/'failed'
        with patch('helicon.rf.continuation.spatial.solve',side_effect=[
                (summary,fields),RuntimeError('solver exception')]):
            with self.assertRaisesRegex(RuntimeError,'solver exception'):
                run(self.f.source,out,100.,self.f.config,steps=10,plots=False)
        record=read_json(out/'continuation_summary.json')
        self.assertEqual(record['status'],'failed')
        self.assertEqual(record['last_accepted_alpha'],0.)
        self.assertEqual(record['current_alpha'],.1)


if __name__=='__main__':unittest.main()
