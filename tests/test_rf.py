"""RF material physics, interpolation, configuration and mesh topology checks."""
from collections import Counter
import copy
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from helicon.rf.config import load,validate
from helicon.rf.materials import cold_epsilon,dissipation_tensor,absorbed_density,AxisymmetricField
from helicon.rf.solve import _power_sanity_checks
from helicon.argon import E_CHARGE,M_E,EPS_0
from helicon.rf.mesh import generate,FACE
from helicon.mesh import gmsh_module


class Materials(unittest.TestCase):
    def test_power_sanity_checks_reject_negative_pml_and_flux(self):
        good = _power_sanity_checks(10., {'PLASMA': 8.}, 2., 2., 0., 1e-10, .001, 1e-8)
        self.assertTrue(all(good.values()))
        bad = _power_sanity_checks(1062.5, {'PLASMA': 1134.75}, -72.25, -75.05,
                                   0., 1e-14, .001, 1e-8)
        self.assertFalse(bad['nonnegative_pml_absorption'])
        self.assertFalse(bad['nonnegative_outward_interface_flux'])
        self.assertFalse(bad['pml_interface_flux_agreement'])
        bad_material = _power_sanity_checks(10., {'PLASMA': -0.1}, 10.1, 10.1,
                                            0., 1e-10, .001, 1e-8)
        self.assertFalse(bad_material['nonnegative_material_absorption'])

    def test_unmagnetized_drude_limit(self):
        f=27.12e6;n=2e18;nu=2e7;w=2*np.pi*f
        expected=1-n*E_CHARGE**2/(M_E*EPS_0*w*(w+1j*nu))
        np.testing.assert_allclose(cold_epsilon(f,n,nu,[0,0,0]),np.eye(3)*expected,rtol=1e-13)

    def test_rotation_covariance_and_B_reversal(self):
        a=.7;Q=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        b=np.array([0,0,.014]);eps=cold_epsilon(27.12e6,2e18,2e7,b)
        np.testing.assert_allclose(cold_epsilon(27.12e6,2e18,2e7,Q@b),Q@eps@Q.T,rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(cold_epsilon(27.12e6,2e18,2e7,-b),eps.T,rtol=1e-12,atol=1e-10)

    def test_positive_absorption_and_lossless_limit(self):
        rng=np.random.default_rng(12);fields=rng.normal(size=(40,3))+1j*rng.normal(size=(40,3))
        for b in ([0,0,0],[.002,.003,.014]):
            eps=cold_epsilon(27.12e6,2e18,2e7,b)
            self.assertTrue(np.all(np.linalg.eigvalsh(dissipation_tensor(eps))>0))
            self.assertTrue(np.all(absorbed_density(fields,eps,27.12e6)>0))
            lossless=cold_epsilon(27.12e6,2e18,0,b)
            np.testing.assert_allclose(dissipation_tensor(lossless),0,atol=1e-10)
        np.testing.assert_allclose(cold_epsilon(27.12e6,0,2e7,[0,0,.014]),np.eye(3))

    def test_peak_phasor_factor_and_amplitude_scaling(self):
        sigma=3.;f=27.12e6;w=2*np.pi*f;eps=np.eye(3)*(1+1j*sigma/(EPS_0*w));field=np.array([2+3j,0,0])
        self.assertAlmostEqual(absorbed_density(field,eps,f),.5*sigma*13)
        self.assertAlmostEqual(absorbed_density(2*field,eps,f),4*absorbed_density(field,eps,f))

    def test_circular_mode_eigenvalues(self):
        f=27.12e6;n=2e18;nu=2e7;b=.014;w=2*np.pi*f;omega=E_CHARGE*b/M_E
        eps=cold_epsilon(f,n,nu,[0,0,b]);v=np.array([1,1j,0])
        eigen=1+1j*n*E_CHARGE**2/(M_E*EPS_0*w*(nu-1j*(w-omega)))
        np.testing.assert_allclose(eps@v,eigen*v,rtol=1e-12)

    def test_map_preserves_cartesian_direction_and_rejects_extrapolation(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'b.npz';r=np.linspace(0,.1,5);z=np.linspace(-.2,.2,7);R,Z=np.meshgrid(r,z)
            np.savez(path,r_m=r,z_m=z,Br_T=2*R,Bz_T=3+Z)
            field=AxisymmetricField(path);points=np.array([[0,0,0],[.03,.04,.05]])
            np.testing.assert_allclose(field(points),[[0,0,3],[.06,.08,3.05]],atol=1e-14)
            with self.assertRaises(ValueError):field([[.2,0,0]])

    def test_bad_config(self):
        for section,key,value in [('antenna','inner_radius_m',.01),('antenna','gap_m',.2),('plasma','collision_frequency_Hz',-1),('solver','degree',3)]:
            c=load();c[section][key]=value
            with self.assertRaises(ValueError):validate(c)
        c=load();c['frequency_Hz']=float('nan')
        with self.assertRaises(ValueError):validate(c)


class Mesh(unittest.TestCase):
    def test_mesh_is_conforming_and_feed_is_internal(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'mesh';c=load();report=generate(c,out,scale=3)
            self.assertGreater(report['tetrahedra'],0)
            self.assertAlmostEqual(report['volumes_m3']['PLASMA'],math.pi*.03**2*.12,places=12)
            with self.assertRaises(FileExistsError):generate(c,out,scale=3)
            g=gmsh_module();g.initialize()
            try:
                g.open(str(out/'rf.msh'));counts=Counter();cellcount=0
                for _,volume in g.model.getEntities(3):
                    if not len(g.model.getPhysicalGroupsForEntity(3,volume)):continue
                    self.assertEqual(len(g.model.getPhysicalGroupsForEntity(3,volume)),1)
                    types,_,connections=g.model.mesh.getElements(3,volume)
                    for typ,conn in zip(types,connections):
                        self.assertEqual(typ,4)
                        for tet in np.asarray(conn).reshape(-1,4):
                            cellcount+=1
                            for inds in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)):
                                counts[tuple(sorted(int(tet[i]) for i in inds))]+=1
                tagged_boundary=set()
                for name,tag in FACE.items():
                    faces=set()
                    for surf in g.model.getEntitiesForPhysicalGroup(2,tag):
                        _,_,conns=g.model.mesh.getElements(2,int(surf))
                        for conn in conns:
                            for tri in np.asarray(conn).reshape(-1,3):faces.add(tuple(sorted(map(int,tri))))
                    self.assertTrue(faces)
                    expected=2 if name in ('FEED_SHEET','PML_INTERFACE') else 1
                    self.assertTrue(all(counts[f]==expected for f in faces),name)
                    if expected==1:tagged_boundary |= faces
                self.assertEqual(tagged_boundary,{f for f,count in counts.items() if count==1})
                self.assertTrue(all(v in (1,2) for v in counts.values()))
                self.assertEqual(cellcount,report['tetrahedra'])
            finally:g.finalize()

if __name__=='__main__':unittest.main()
