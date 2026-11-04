"""Diagnostic reconstruction, exact face cancellation and saved-state comparison."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from helicon import spatial
from helicon.config import load as geometry
from helicon.records import write_json,digest,read_json
from helicon.spatial_budget import reconstruct,compare,ENERGY_LOSSES


class SpatialBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry=geometry();cls.c=spatial.load();cls.c.update(nr=3,nz=4)
        cls.summary,cls.fields=spatial.solve(cls.geometry,cls.c)

    def equation(self,fields):
        g=spatial.Grid(self.geometry,3,4)
        eq=spatial.Equations(g,self.c,fields['q_abs_W_m3'])
        x=np.array([np.log(fields['electron_density_m3']/1e18).ravel(),
                    np.log(fields['neutral_density_m3']/1e19).ravel(),
                    fields['electron_temperature_eV'].ravel(),fields['potential_V'].ravel()])
        return g,eq,x

    def rejected(self):
        fields=copy.deepcopy(self.fields)
        fields['electron_temperature_eV'][0,-1]=7.
        g,eq,x=self.equation(fields);a=eq.evaluate(x)
        fields['collision_frequency_s']=a['nu'].reshape(g.shape)
        summary=copy.deepcopy(self.summary);summary['status']='not_converged'
        b=summary['balances']
        b['max_local_relative_residuals']={key:float(np.max(abs(a[key]/scale))) for key,scale in (
            ('ion_res',eq.particle_scale),('neutral_res',eq.particle_scale),
            ('current_res',eq.particle_scale),('energy_res',eq.energy_scale))}
        b.update(inelastic_power_W=float(a['inelastic'].sum()),
                 electron_boundary_power_W=float(a['boundary_energy'].sum()),
                 electron_electrostatic_work_W=float(a['work'].sum()))
        exit_faces=g.boundary_kind==2
        b['relative_mass_error']=float(abs(eq.feed-a['boundary_neutrals'][exit_faces].sum()
            -a['boundary_ions'][exit_faces].sum()
            -(1-self.c['wall_recycling_fraction'])*a['boundary_ions'][~exit_faces].sum())/eq.feed)
        b['relative_power_error']=float(abs(eq.power-a['inelastic'].sum()
            -a['boundary_energy'].sum()+a['work'].sum())/eq.power)
        return summary,fields

    def test_opt_in_diagnostics_preserve_existing_equation_values(self):
        g,eq,x=self.equation(self.fields)
        normal=eq.evaluate(x);expanded=eq.evaluate(x,diagnostics=True)
        for key,value in normal.items():np.testing.assert_array_equal(value,expanded[key])
        self.assertNotIn('energy_budget',normal)

    def test_independent_face_accounting_and_nonzero_saved_residual(self):
        summary,fields=self.rejected()
        g,rows,info,a=reconstruct(self.geometry,self.c,summary,fields)
        self.assertEqual(info['saved_status'],'not_converged')
        self.assertGreater(info['reconstructed_max_local_residuals']['energy_res'],1e-3)
        faces=a['face_budget'];e=a['energy_budget']
        # Explicit incidence accumulation independent of Grid.scatter.
        for facekey,cellkey in (('conduction_W','conduction_out_W'),('enthalpy_W','enthalpy_out_W')):
            expected=np.zeros(g.size)
            for l,r,v in zip(faces['left_cell'],faces['right_cell'],faces[facekey]):
                expected[l]+=v;expected[r]-=v
            np.testing.assert_allclose(expected,e[cellkey],rtol=1e-13,atol=1e-12)
            self.assertLess(abs(expected.sum()),1e-11)
        inelastic=e['ionization_loss_W']+e['excitation_loss_W']+e['elastic_exchange_W']
        np.testing.assert_allclose(inelastic,a['inelastic'],rtol=1e-13,atol=1e-12)
        reconstructed=sum(e[k] for k in ENERGY_LOSSES)-e['rf_heating_W']-e['electrostatic_work_W']
        np.testing.assert_allclose(reconstructed,a['energy_res'],atol=1e-12)
        self.assertEqual(info['hottest_cells'][0]['cell'],2)
        self.assertTrue(rows[2]['Te_near_bound'])
        self.assertAlmostEqual(rows[2]['r_m'],g.r[-1])

    def test_rejects_changed_geometry_and_unmatched_saved_residual(self):
        geom=copy.deepcopy(self.geometry);geom['chamber']['radius']*=1.1
        with self.assertRaisesRegex(ValueError,'grid differs'):
            reconstruct(geom,self.c,self.summary,self.fields)
        wrong=copy.deepcopy(self.summary)
        wrong['balances']['max_local_relative_residuals']['current_res']=.4
        with self.assertRaisesRegex(ValueError,'residuals differ'):
            reconstruct(self.geometry,self.c,wrong,self.fields)

    def test_saved_pair_comparison_and_input_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'continuation';parent.mkdir()
            write_json(parent/'geometry.json',self.geometry)
            write_json(parent/'plasma_config.json',self.c)
            rejected,fields=self.rejected()
            spatial.save(parent/'steps/step-110',self.summary,self.fields)
            spatial.save(parent/'steps/step-111',rejected,fields)
            record={'history':[{'alpha':.220,'accepted':True,'directory':'steps/step-110'},
                               {'alpha':.222,'accepted':False,'directory':'steps/step-111'}],
                    'artifacts_sha256':{str(p.relative_to(parent)):digest(p)
                        for p in parent.rglob('*') if p.is_file()}}
            write_json(parent/'continuation_summary.json',record)
            result=compare(parent,root/'out',plots=False)
            self.assertEqual(result['cases']['accepted']['alpha'],.220)
            self.assertEqual(result['cases']['rejected']['alpha'],.222)
            self.assertTrue((root/'out/cell_changes.csv').exists())
            self.assertTrue((root/'out/rejected/face_budget.npz').exists())
            self.assertEqual(read_json(root/'out/run.json')['status'],'completed')
            with self.assertRaises(FileExistsError):compare(parent,root/'out',plots=False)
            (parent/'geometry.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'artifact hash'):compare(parent,root/'tampered',plots=False)


if __name__=='__main__':unittest.main()
