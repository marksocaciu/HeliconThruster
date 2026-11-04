"""Independent quadrature references, backend bounds and conservative transport."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from helicon import argon,argon_chemistry as chemistry,spatial
from helicon.config import load as geometry
from helicon.spatial_budget import reconstruct
from helicon.rf.spatial_maps import PlasmaMap


class ArgonChemistry(unittest.TestCase):
    def test_legacy_backend_is_exact_and_rejects_extrapolation(self):
        for t in (1.,2.4,4.,7.):self.assertEqual(chemistry.rates(t,{}),argon.rates(t))
        with self.assertRaises(ValueError):chemistry.rates(7.01,{})
        with self.assertRaises(ValueError):chemistry.bounds({'chemistry_model':'invented'})

    def test_rate_normalization_and_elastic_energy_moment_analytic_limit(self):
        # Constant cross sections have <sigma v E> = 2 Te <sigma v>.
        knots=np.linspace(0.,1000.,101);nodes,w=np.polynomial.legendre.leggauss(16)
        energy=((knots[:-1,None]+knots[1:,None])/2+np.diff(knots)[:,None]/2*nodes).ravel()
        weight=(np.diff(knots)[:,None]/2*w).ravel()
        sigma={key:np.full(len(energy),value) for key,value in (
            ('IONIZATION',1e-20),('EXCITATION',2e-20),('ELASTIC',3e-20),('EFFECTIVE',6e-20))}
        for t in (1.,7.,20.):
            k=chemistry.integrate(t,energy,weight,sigma)
            expected=3e-20*np.sqrt(8*argon.E_CHARGE*t/(np.pi*argon.M_E))
            self.assertAlmostEqual(k['elastic_m3_s']/expected,1.,places=12)
            self.assertAlmostEqual(k['elastic_energy_weighted_eVm3_s']/(2*t*expected),1.,places=12)

    def test_table_against_independent_adaptive_quadrature_reference(self):
        reference=json.loads((Path(__file__).resolve().parents[1]/'validation/argon-phelps-reference.json').read_text())
        self.assertEqual(reference['source_cross_sections_sha256'],chemistry.table()['provenance']['cross_sections_sha256'])
        for row in reference['reference']:
            k=chemistry.rates(row['temperature_eV'],{'chemistry_model':chemistry.PHELPS})
            for key in chemistry.COLUMNS:self.assertLess(abs(k[key]/row[key]-1),1e-6)
            self.assertAlmostEqual(k['ionization_energy_eVm3_s']/k['ionization_m3_s'],15.8)
            self.assertAlmostEqual(k['excitation_energy_eVm3_s']/k['excitation_m3_s'],11.5)

    def test_new_bounds_and_positive_smooth_interpolation(self):
        c={'chemistry_model':chemistry.PHELPS}
        self.assertEqual(chemistry.bounds(c),(1.,20.))
        for t in (0.99,20.001,float('nan'),True):
            with self.assertRaises(ValueError):chemistry.rates(t,c)
        for t in np.geomspace(1.,20.,101):
            values=chemistry.rates(float(t),c)
            self.assertTrue(all(np.isfinite(v) and v>0 for v in values.values()))
            self.assertGreaterEqual(values['momentum_m3_s'],values['elastic_m3_s'])

    def test_new_collision_frequency_uses_effective_momentum_once(self):
        c=spatial.load();c['chemistry_model']=chemistry.PHELPS
        te=np.array([3.,8.]);nn=np.array([1e19,2e19])
        k,nu,_=spatial.coefficients(te,nn,c)
        np.testing.assert_allclose(nu,nn*k['momentum_m3_s']+c['collision_floor_s-1'])
        self.assertTrue(np.all(k['momentum_m3_s']>k['elastic_m3_s']))

    def test_hot_saved_state_budget_and_rf_map_use_selected_backend(self):
        c=spatial.load();c.update(nr=3,nz=4,chemistry_model=chemistry.PHELPS)
        summary,fields=spatial.solve(geometry(),c)
        self.assertEqual(summary['status'],'converged')
        fields=copy.deepcopy(fields);fields['electron_temperature_eV'][:]=8.
        g=spatial.Grid(geometry(),3,4);eq=spatial.Equations(g,c,fields['q_abs_W_m3'])
        x=np.array([np.log(fields['electron_density_m3']/1e18).ravel(),
            np.log(fields['neutral_density_m3']/1e19).ravel(),np.full(g.size,8.),fields['potential_V'].ravel()])
        a=eq.evaluate(x,diagnostics=True)
        e=a['energy_budget']
        np.testing.assert_allclose(e['ionization_loss_W']+e['excitation_loss_W']+e['elastic_exchange_W'],a['inelastic'],rtol=1e-13)
        fields['collision_frequency_s']=a['nu'].reshape(g.shape)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.npz';np.savez(path,**fields)
            PlasmaMap(path) # Explicit backend accepts supported Te above 7 eV.
            fields.pop('chemistry_model');np.savez(path,**fields)
            with self.assertRaisesRegex(ValueError,'1–7'):PlasmaMap(path)
        # The actual solved state's diagnostic retains its selected bounds and table hash.
        s,f=spatial.solve(geometry(),c)
        _,rows,info,_=reconstruct(geometry(),c,s,f)
        self.assertEqual(info['chemistry']['temperature_interval_eV'],[1.,20.])
        self.assertAlmostEqual(rows[0]['Te_upper_margin_eV'],20-f['electron_temperature_eV'].ravel()[0])
        s['chemistry']['table_sha256']='tampered'
        with self.assertRaisesRegex(ValueError,'chemistry provenance'):reconstruct(geometry(),c,s,f)


if __name__=='__main__':unittest.main()
