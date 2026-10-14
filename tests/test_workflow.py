"""Workflow tests use synthetic data; they do not claim FEM verification."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from helicon.config import load
from helicon.diagnostics import design_metrics, reconstruct_field, disk_mean
from helicon.records import config_digest, read_json, write_json, digest
from helicon.run import run_case
from helicon.study import convergence_cases, expanded_domain, sweep_cases, create_study, summarize


class DiagnosticsTests(unittest.TestCase):
    def test_regular_axis_limit_and_off_axis_field(self):
        r=np.array([0.,.01,.03]); z=np.array([-.12,-.06,0.])
        potential=r*(2+3*z)
        gradient=np.column_stack([2+3*z,3*r])
        br,bz=reconstruct_field(r,potential,gradient)
        np.testing.assert_allclose(br,-3*r)
        np.testing.assert_allclose(bz,2*(2+3*z))

    def test_exit_disk_weight(self):
        nodes,w=np.polynomial.legendre.leggauss(32); R=.03; r=(nodes+1)*R/2
        self.assertAlmostEqual(disk_mean(np.ones_like(r)*.014,r,w,R),.014)
        self.assertAlmostEqual(disk_mean(r*r,r,w,R),R*R/2)

    def test_design_objective_and_zero_denominators(self):
        m=design_metrics(.014,0,0,.014)
        self.assertEqual(m['normalized_rms_field_error'],0)
        zero=design_metrics(0,0,0,.014)
        self.assertIsNone(zero['chamber_Bz_nonuniformity_percent'])
        self.assertEqual(zero['normalized_rms_field_error'],1)
        self.assertEqual(design_metrics(-.014,0,0,.014)['normalized_rms_field_error'],2)


class PlanningTests(unittest.TestCase):
    def test_domain_expands_about_chamber_center_without_changing_local_sizes(self):
        c=load(); larger=expanded_domain(c,2)
        self.assertEqual(c['mesh'],larger['mesh'])
        self.assertEqual(c['coils'],larger['coils'])
        self.assertAlmostEqual(larger['domain']['z_min'],-1.38)
        self.assertAlmostEqual(larger['domain']['z_max'],1.26)
        self.assertAlmostEqual(larger['domain']['radius'],1.2)

    def test_sweep_ampere_turns_and_positions_cartesian(self):
        c=load(); before=copy.deepcopy(c)
        spec={'parameters':[{'coil':'coil_1','field':'ampere_turns','values':[-100,200]},
                            {'coil':'coil_2','field':'z_center','values':[-.02,.02]}]}
        cases=sweep_cases(c,spec)
        self.assertEqual(len(cases),4)
        self.assertEqual(c,before)
        self.assertEqual([x['config']['coils'][0]['current'] for x in cases],[-1,-1,2,2])
        with self.assertRaises(ValueError): sweep_cases(c,spec,max_cases=3)
        invalid=sweep_cases(c,{'parameters':[{'coil':'coil_2','field':'z_center','values':[-.12]}]})
        self.assertIn('Overlapping',invalid[0]['invalid_reason'])

    def test_reject_bad_convergence_plan(self):
        for scales,factors in [([1],[1,2]),([.5,1],[1,2]),([1,.5],[1,1]),([1,.5],[2,3])]:
            with self.assertRaises(ValueError): convergence_cases(load(),scales,factors)

    def test_failure_is_recorded_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'run'
            with patch('helicon.run.stage',side_effect=RuntimeError('deliberate test failure')):
                with self.assertRaises(RuntimeError): run_case(load(),out)
            saved=read_json(out/'run.json')
            self.assertEqual(saved['status'],'failed')
            self.assertIn('deliberate test failure',saved['error'])
            self.assertTrue((out/'source/helicon/solve.py').exists())
            with self.assertRaises(FileExistsError): run_case(load(),out)
            self.assertEqual(read_json(out/'run.json'),saved)

    def test_incomplete_study_cannot_pass_and_changes_are_measured(self):
        with tempfile.TemporaryDirectory() as temp:
            c=load(); cases=convergence_cases(c,[1,.5],[1,2])
            out=create_study(Path(temp)/'study','convergence',cases,140,1)
            first=summarize(out)
            self.assertFalse(first['complete']); self.assertFalse(first['sensitivity_criteria_met'])
            for index,case in enumerate(cases):
                folder=out/case['id']; folder.mkdir()
                (folder/'source').mkdir(); (folder/'source/test_fixture').write_text('synthetic')
                mean=.014+index*1e-6
                m={'chamber_mean_Bz_T':mean,'chamber_Bz_std_T':.00017,
                   'chamber_Br_rms_T':.00018,'axis_exit_Bz_T':.0133,
                   'exit_area_mean_Bz_T':.0134,'axis_exit_Bz_G':133.,'exit_area_mean_Bz_G':134.,
                   'axis_reference_error_percent':.2}
                m.update(design_metrics(mean,.00017,.00018,.014))
                z=np.linspace(-.12,0,121)
                np.savetxt(folder/'fem_axis.csv',np.column_stack([z,np.full_like(z,mean),np.full_like(z,.014)]),
                           delimiter=',',header='z_m,Bz_FEM_T,Bz_free_space_pack_T',comments='')
                write_json(folder/'run.json',{'status':'completed','config_sha256':config_digest(case['config']),
                    'mesh_scale':case['mesh_scale'],'target_gauss':140,'metrics':m,
                    'source':{'sha256':{'test_fixture':digest(folder/'source/test_fixture')}},'artifacts_sha256':{},
                    'mesh':{'nodes':100,'triangles':150}})
            with patch('helicon.study.plot_comparison'):
                result=summarize(out)
            self.assertTrue(result['sensitivity_criteria_met'])
            change=result['checks']['mesh']['comparisons'][0]['scalar_change_percent_of_target']['chamber_mean_Bz_T']
            self.assertAlmostEqual(change,100*1e-6/.014)
            record=read_json(out/'domain_01/run.json')
            record['metrics']['axis_reference_error_percent']=3
            write_json(out/'domain_01/run.json',record)
            with patch('helicon.study.plot_comparison'):
                self.assertFalse(summarize(out)['sensitivity_criteria_met'])


class RenderingTests(unittest.TestCase):
    def test_synthetic_field_report_and_stale_input_rejection(self):
        from helicon.plot import render
        from helicon.records import digest
        with tempfile.TemporaryDirectory(prefix='synthetic-render-') as temp:
            out=Path(temp); c=load()
            write_json(out/'geometry.json',c)
            (out/'helicon.msh').write_text('Synthetic fixture: no FEM solve')
            m=design_metrics(.014,0,0,.014)
            m.update(axis_exit_Bz_G=140,exit_area_mean_Bz_G=140,axis_reference_error_percent=0,
                     configuration_sha256=digest(out/'geometry.json'),mesh_sha256=digest(out/'helicon.msh'))
            write_json(out/'fem_summary.json',m)
            z=np.linspace(-.12,0,121); r=np.linspace(0,.03,101)
            np.savetxt(out/'fem_axis.csv',np.column_stack([z,z*0+.014,z*0+.014]),delimiter=',',
                       header='z_m,Bz_FEM_T,Bz_free_space_pack_T',comments='')
            np.savetxt(out/'fem_exit.csv',np.column_stack([r,r*0,r*0+.014,r*0+.014]),delimiter=',',
                       header='r_m,Br_T,Bz_T,B_magnitude_T',comments='')
            for name in ('device','full'):
                rr=np.linspace(0,.6,31); zz=np.linspace(-.72,.6,41); R,Z=np.meshgrid(rr,zz)
                np.savez_compressed(out/f'field_{name}.npz',r_m=rr,z_m=zz,A_theta=.007*R,
                                    Br_T=R*0,Bz_T=R*0+.014,psi_Wb_per_rad=.007*R**2)
            render(out)
            for name in ('report.html','report.md','profiles.png','field_device.png','field_full.png'):
                self.assertGreater((out/name).stat().st_size,100)
            c['coils'][0]['current']=0
            write_json(out/'geometry.json',c)
            with self.assertRaises(ValueError): render(out)


class RetryTests(unittest.TestCase):
    def test_retry_archives_failure_and_preserves_inputs(self):
        from helicon.study import execute
        with tempfile.TemporaryDirectory() as temp:
            cases=convergence_cases(load(),[1,.5],[1,2])
            out=create_study(Path(temp)/'study','convergence',cases,140,1)
            folder=out/cases[0]['id']; folder.mkdir()
            write_json(folder/'run.json',{'status':'failed','error':'synthetic failure'})
            (folder/'solve.log').write_text('preserve this diagnostic')
            with patch('helicon.study.run_case') as run, patch('helicon.study.summarize',return_value={}):
                execute(out,retry_failed=True)
            self.assertEqual(run.call_count,4)
            archived=list((out/'attempts').iterdir())
            self.assertEqual(len(archived),1)
            self.assertEqual((archived[0]/'solve.log').read_text(),'preserve this diagnostic')

if __name__=='__main__': unittest.main()
