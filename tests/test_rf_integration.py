"""Opt-in Linux complex-FEniCS tests. Set HELICON_RF_INTEGRATION=1.

Numerical smoke/scaling tests, not mesh/PML or physics validation.
"""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from helicon.rf.config import load


@unittest.skipUnless(os.environ.get('HELICON_RF_INTEGRATION')=='1','requires opt-in complex DOLFINx environment')
class RFIntegration(unittest.TestCase):
    def test_driven_vacuum_lossy_medium_and_quadratic_scaling(self):
        from helicon.rf.solve import runtime,solve
        from helicon.rf.mesh import generate
        MPI,_,_=runtime()
        self.assertEqual(MPI.COMM_WORLD.size,1,'Run this test serially; use CLI for MPI comparisons')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=load(); results=[]
            for name,case,voltage in [('v1','vacuum',1.),('v2','vacuum',2.),('lossy','dielectric',1.),('plasma','plasma',1.)]:
                c=copy.deepcopy(base);c['material_case']=case;c['voltage_peak_V']=voltage
                build=root/(name+'-mesh');generate(c,build,scale=3.)
                result=solve(build,root/(name+'-solve'))
                self.assertTrue(result['numerical_checks_passed'],result)
                self.assertGreater(result['input_power_W'],0)
                results.append(result)
            self.assertAlmostEqual(results[1]['input_power_W']/results[0]['input_power_W'],4.,places=5)
            self.assertGreater(results[2]['material_absorption_W']['PLASMA'],0)
            self.assertGreater(results[3]['material_absorption_W']['PLASMA'],0)

if __name__=='__main__':unittest.main()
