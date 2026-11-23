"""Checks on the closed-current source and an independently assembled full 3D system."""
import unittest
import numpy as np
from helicon.rf.config import load
from helicon.rf.materials import MU0
from helicon.rf_modes.fem import assemble, local_operator
from helicon.rf_modes.nagoya import setup
from helicon.rf_modes.nagoya_prism import disk, solve_prisms
from helicon.rf_modes.nagoya_source import solve_current
from helicon.rf_modes.reference3d import prism_basis


class NagoyaCurrent(unittest.TestCase):
    def setUp(self):
        self.c, self.r, self.eps, self.sheet = setup(load(), [2, 1, 1, 1], .06, 1.)

    def test_rail_current_fourier_and_discrete_surface_divergence(self):
        sheet = self.sheet
        z = np.linspace(0, sheet.length, 10)
        axial, returns = sheet.axial_integrals(z)
        dz = np.diff(z)
        np.testing.assert_allclose(returns, np.r_[axial/dz, 0]-np.r_[0, axial/dz], atol=1e-13)
        n = 360
        phi = np.arange(n)*2*np.pi/n
        vertical, circum = np.zeros(n), np.zeros(n)
        for j in range(n):
            pair, circum[j] = sheet.segment_integrals(phi[j], (j+1)*2*np.pi/n)
            vertical[[j,(j+1)%n]] += pair
        self.assertAlmostEqual(vertical[vertical>0].sum(), 1.)
        np.testing.assert_allclose(vertical, circum-np.roll(circum, 1), atol=2e-13)
        for m in [0, 1, -3, 4, 7]:
            ring, axial_mode, return_mode = sheet.modal_load(m, self.r, z)
            approx = np.exp(-1j*m*phi) @ vertical
            np.testing.assert_allclose(axial_mode, approx*axial, rtol=.002, atol=1e-12)
            self.assertAlmostEqual(self.r[ring], sheet.radius)
            np.testing.assert_allclose(1j*m*return_mode, -approx*returns, rtol=.002, atol=1e-12)

    def test_axial_transform_against_monolithic_prisms(self):
        nz, sectors = 5, 6
        omega = 2*np.pi*self.c['frequency_Hz']
        dz = self.sheet.length/nz
        xy, rings, tri, bands = disk(self.r, sectors)
        pairs = [(0,1),(1,2),(2,0)]
        edges, ids = np.unique(np.sort(tri[:,pairs],axis=-1).reshape(-1,2), axis=0, return_inverse=True)
        ids=ids.reshape(-1,3)
        he = np.arange((nz+1)*len(edges)).reshape(nz+1,-1)
        ve = he.size + np.arange(nz*len(xy)).reshape(nz,-1)
        cells=np.concatenate((he[:-1,ids],he[1:,ids],ve[:,tri]),axis=2)
        blocks = []
        for t, band in zip(tri,bands):
            jac=(xy[t[1:]]-xy[t[0]]).T
            grad=np.array([[-1,-1],[1,0],[0,1]]) @ np.linalg.inv(jac)
            oriented=[(a,b) if t[a]<t[b] else (b,a) for a,b in pairs]
            mat=np.zeros((9,9),complex)
            for bary in np.eye(3)*.5+1/6:
                for v in [.5-.5/np.sqrt(3), .5+.5/np.sqrt(3)]:
                    basis,curl=prism_basis(grad,oriented,bary,v,dz)
                    mat += np.linalg.det(jac)*dz/12*local_operator(basis,curl,self.eps[band],omega)
            blocks.append(mat)
        size=he.size+ve.size
        matrix=assemble(cells.reshape(-1,9),np.tile(blocks,(nz,1,1)),size)
        fixed=np.zeros(size,bool)
        fixed[he[[0,-1]]]=True
        fixed[he[:,np.isin(edges,rings[-1]).all(axis=1)]]=True
        fixed[ve[:,rings[-1]]]=True
        load_vector=np.zeros(size)
        rails,returns=self.sheet.axial_integrals(np.linspace(0,self.sheet.length,nz+1))
        ids_by_edge={tuple(e):i for i,e in enumerate(edges)}
        ring=rings[np.flatnonzero(np.isclose(self.r[1:],self.sheet.radius)).item()]
        for j,a in enumerate(ring):
            b=ring[(j+1)%sectors]
            weights,g=self.sheet.segment_integrals(j*2*np.pi/sectors,(j+1)*2*np.pi/sectors)
            load_vector[ve[:,[a,b]]] += rails[:,None]*weights
            load_vector[he[:,ids_by_edge[tuple(sorted((a,b)))]]] -= np.sign(b-a)*g*returns
        _, work, residual=solve_current(matrix,fixed,load_vector,omega)
        watts, stats=solve_prisms(self.c,self.r,self.eps,self.sheet,nz,sectors)
        self.assertLess(residual,1e-7)
        np.testing.assert_allclose([watts.sum(),stats['source_work_W']], work, rtol=1e-8)
