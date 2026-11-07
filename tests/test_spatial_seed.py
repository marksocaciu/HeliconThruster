"""Accepted-state integrity, interpolation and unchanged target acceptance checks."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from helicon import spatial
from helicon.config import load as geometry
from helicon.records import write_json,digest,read_json
from helicon.spatial_seed import interpolate,load_seed
from helicon.rf.continuation import run
import test_rf_prescribed as fixtures


class SpatialSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geom=geometry();cls.c=spatial.load();cls.c.update(nr=3,nz=4)
        cls.summary,cls.fields=spatial.solve(cls.geom,cls.c)

    def source(self,root,summary=None,fields=None):
        parent=root/'seed';parent.mkdir()
        write_json(parent/'geometry.json',self.geom);write_json(parent/'plasma_config.json',self.c)
        spatial.save(parent/'steps/step-000',summary or self.summary,fields or self.fields)
        self.seal(parent)
        return parent/'steps/step-000'

    def seal(self,parent,alpha=0.,accepted=True):
        write_json(parent/'continuation_summary.json',{
            'history':[{'directory':'steps/step-000','alpha':alpha,'accepted':accepted}],
            'artifacts_sha256':{str(p.relative_to(parent)):digest(p) for p in parent.rglob('*')
                               if p.is_file() and p.name!='continuation_summary.json'}})

    def test_log_linear_profiles_gauge_and_collision_recomputation(self):
        g=spatial.Grid(self.geom,2,3);fine=spatial.Grid(self.geom,4,6)
        zz,rr=np.meshgrid(g.z,g.r,indexing='ij')
        fields={'electron_density_m3':1e17*np.exp(rr+zz),
                'neutral_density_m3':1e19*np.exp(2*rr-zz),
                'electron_temperature_eV':3*np.exp(rr+.5*zz),
                'potential_V':4*rr+2*zz+80,
                'collision_frequency_s':np.full(g.shape,-123.)}
        state=interpolate(fields,g,fine,self.c)
        z,r=np.meshgrid(np.clip(fine.z,g.z[0],g.z[-1]),np.clip(fine.r,g.r[0],g.r[-1]),indexing='ij')
        np.testing.assert_allclose(state['electron_density_m3'],1e17*np.exp(r+z),rtol=1e-13)
        np.testing.assert_allclose(state['neutral_density_m3'],1e19*np.exp(2*r-z),rtol=1e-13)
        np.testing.assert_allclose(state['electron_temperature_eV'],3*np.exp(r+.5*z),rtol=1e-13)
        expected=4*r+2*z;expected-=expected.flat[0]
        np.testing.assert_allclose(state['potential_V'],expected,atol=3e-14)
        _,nu,_=spatial.coefficients(state['electron_temperature_eV'].ravel(),state['neutral_density_m3'].ravel(),self.c)
        np.testing.assert_array_equal(state['collision_frequency_s'],nu.reshape(fine.shape))
        self.assertEqual(state['potential_V'].flat[0],0.)
        self.assertTrue(np.all(state['electron_density_m3']>0))

    def test_verified_seed_initializes_real_finer_solve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.source(root)
            c=copy.deepcopy(self.c);c['nz']=8
            before=copy.deepcopy(c)
            state,info=load_seed(source,self.geom,c,root/'initialization')
            summary,_=spatial.solve(self.geom,c,initial=state)
            self.assertEqual(summary['status'],'converged')
            self.assertEqual(c,before)
            self.assertEqual(c['max_nfev'],400)
            self.assertEqual(info['target_grid'],[3,8])
            self.assertTrue((root/'initialization/interpolated_initial_state.npz').exists())

    def test_changed_hash_incompatible_physics_or_wrong_alpha_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.source(root);parent=root/'seed'
            c=copy.deepcopy(self.c);c['ion_mobility_m2_V_s']*=2
            with self.assertRaisesRegex(ValueError,'beyond grid'):
                load_seed(source,self.geom,c,root/'out')
            self.seal(parent,alpha=1.)
            with self.assertRaisesRegex(ValueError,'accepted alpha=0'):
                load_seed(source,self.geom,self.c,root/'out')
            self.seal(parent,accepted=False)
            with self.assertRaisesRegex(ValueError,'accepted alpha=0'):
                load_seed(source,self.geom,self.c,root/'out')
            self.seal(parent)
            (source/'plasma_fields.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_seed(source,self.geom,self.c,root/'out')

    def test_nonuniform_source_is_not_accepted_as_uniform_seed(self):
        g=spatial.Grid(self.geom,3,4)
        q=np.broadcast_to(np.linspace(.8,1.2,4)[:,None],g.shape).copy()
        q*=100/np.sum(q*g.volumes)
        summary,fields=spatial.solve(self.geom,self.c,q,initial=self.fields)
        self.assertEqual(summary['status'],'converged')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.source(root,summary,fields)
            with self.assertRaisesRegex(ValueError,'uniform heating'):
                load_seed(source,self.geom,self.c,root/'out')

    def test_continuation_records_external_seed_without_accepting_it_as_target_solution(self):
        f=fixtures.PrescribedHeating();f.setUp();self.addCleanup(f.doCleanups)
        seed=self.source(f.root)
        result=run(f.source,f.root/'out',100.,f.config,steps=2,plots=False,initial_state=seed)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['history'][0]['initialization'],'coarse_grid_seed')
        self.assertIsNone(result['history'][0]['initialized_from_alpha'])
        self.assertEqual(result['history'][1]['initialization'],'previous_alpha')
        self.assertTrue(result['history'][0]['accepted'])
        self.assertEqual(read_json(f.root/'out/plasma_config.json')['max_nfev'],400)


if __name__=='__main__':unittest.main()
