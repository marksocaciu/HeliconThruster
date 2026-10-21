"""RF input provenance and portable report tests, without a claimed PDE solve."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from helicon.rf.config import load
from helicon.rf.coupled import _validated_global_state
from helicon.rf.run import import_plasma
from helicon.rf.report import render
from helicon.records import write_json,digest


class Workflow(unittest.TestCase):
    def test_coupled_global_state_boundary_checks(self):
        result={'status':'steady',
                'metrics':{'electron_density_m3':1e18,'electron_temperature_eV':4.,'neutral_density_m3':2e19},
                'balances':{'relative_residuals':{'ion':0.,'neutral':1e-12,'total_mass':0.,'power':1e-12}}}
        self.assertEqual(_validated_global_state(result, 'fixture')['electron_temperature_eV'], 4.)
        invalid=copy.deepcopy(result);invalid['balances']['relative_residuals']['power']=1e-3
        with self.assertRaisesRegex(RuntimeError, 'conservation checks'):
            _validated_global_state(invalid, 'fixture')
        invalid=copy.deepcopy(result);invalid['metrics']['electron_temperature_eV']=8.
        with self.assertRaisesRegex(RuntimeError, '1–7 eV'):
            _validated_global_state(invalid, 'fixture')

    def test_global_state_import_checks_geometry_and_status(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'plasma.json';c=load();before=copy.deepcopy(c)
            state={'model':'argon_global_0d_v1','status':'steady','geometry':{'radius_m':.03,'length_m':.12},
                   'metrics':{'electron_density_m3':1e18,'neutral_density_m3':2e19,'electron_temperature_eV':4.}}
            write_json(path,state);actual=import_plasma(c,path)
            self.assertEqual(actual['plasma']['electron_density_m3'],1e18);self.assertEqual(c,before)
            self.assertEqual(actual['plasma']['collision_frequency_Hz'],c['plasma']['collision_frequency_Hz'])
            state['geometry']['radius_m']=.04;write_json(path,state)
            with self.assertRaises(ValueError):import_plasma(c,path)
            state['status']='neutral_only';write_json(path,state)
            with self.assertRaises(ValueError):import_plasma(c,path)

    def test_synthetic_report_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);write_json(out/'rf.json',load());write_json(out/'mesh_report.json',{'mesh_scale':1.})
            r={'case':'synthetic fixture','frequency_Hz':27.12e6,'input_power_W':1.,'material_absorption_W':{'PLASMA':.7},
               'pml_absorption_W':.3,'outward_interface_flux_W':.3,'power_balance_relative_error':0.,'free_equation_relative_residual':1e-12,
               'numerical_checks_passed':True,'limitations':['Synthetic data; no PDE solve.']}
            write_json(out/'rf_summary.json',r)
            np.savez(out/'cells_rank0000.npz',xyz_m=np.array([[0,0,-.1],[.01,0,-.06],[-.01,0,-.03]]),
                     region=np.ones(3,dtype=int),E_peak_V_m=np.ones((3,3),dtype=complex),q_abs_W_m3=np.ones(3))
            write_json(out/'run.json',{'artifacts_sha256':{p.name:digest(p) for p in out.iterdir()}})
            render(out);self.assertGreater((out/'rf_fields.png').stat().st_size,1000)
            (out/'rf_summary.json').write_text('{}')
            with self.assertRaises((ValueError,KeyError)):render(out)

if __name__=='__main__':unittest.main()
