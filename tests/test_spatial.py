"""Spatial PDE conservation, heating response, transport and transfer checks."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from helicon.config import load as geometry
from helicon.spatial import Grid, Equations, load, solve
from helicon.rf.materials import cold_epsilon, absorbed_density
from helicon.rf.spatial_maps import PlasmaMap, heating_from_rf
from helicon.rf.mesh import CELL


class SpatialPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load(); cls.c.update(nr=3, nz=4)
        cls.summary, cls.fields = solve(geometry(), cls.c)

    def test_actual_steady_pde_and_independent_integrated_balances(self):
        s, f = self.summary, self.fields
        self.assertEqual(s['status'], 'converged', s)
        self.assertLess(max(s['balances']['max_local_relative_residuals'].values()), 1e-7)
        from helicon.argon import M_AR
        b = s['balances']
        feed = self.c['mass_flow_kg_s']/M_AR
        self.assertAlmostEqual((b['neutral_exit_atoms_s']+b['exit_ions_s']+b['wall_retained_atoms_s'])/feed, 1., places=8)
        self.assertAlmostEqual((b['inelastic_power_W']+b['electron_boundary_power_W']-b['electron_electrostatic_work_W'])/b['absorbed_power_W'], 1., places=8)
        self.assertGreater(np.ptp(f['electron_density_m3'])/f['electron_density_m3'].mean(), .1)
        self.assertGreater(np.ptp(f['potential_V']), .1)
        self.assertGreater(np.ptp(f['electron_temperature_eV']), .1)
        self.assertAlmostEqual(f['cell_volume_m3'].sum(), np.pi*.03**2*.12)

    def test_nonuniform_prescribed_heating_changes_resolved_solution(self):
        f = self.fields
        q = np.ones((4, 3))*np.linspace(.8, 1.2, 4)[:, None]
        q *= self.c['absorbed_power_W']/np.sum(q*f['cell_volume_m3'])
        s, heated = solve(geometry(), self.c, q, initial=f)
        self.assertEqual(s['status'], 'converged', s)
        self.assertGreater(np.max(np.abs(heated['electron_temperature_eV']-f['electron_temperature_eV'])), .01)
        self.assertAlmostEqual(s['balances']['absorbed_power_W'], 100., places=9)

    def test_iteration_limit_is_not_reported_as_converged(self):
        c = copy.deepcopy(self.c); c['max_nfev'] = 1
        s, _ = solve(geometry(), c)
        self.assertEqual(s['status'], 'not_converged')

    def test_jacobian_coloring_matches_independent_perturbations(self):
        g = Grid(geometry(), 3, 4)
        eq = Equations(g, self.c, self.fields['q_abs_W_m3'])
        f = self.fields
        x = np.array([np.log(f['electron_density_m3']/1e18).ravel(),
                      np.log(f['neutral_density_m3']/1e19).ravel(),
                      f['electron_temperature_eV'].ravel(), f['potential_V'].ravel()]).ravel()
        jac = eq.jacobian()(x)
        for j in (0, 11, 16, 29, 37, 47):
            step = np.sqrt(np.finfo(float).eps)*max(abs(x[j]), 1.)
            point = x.copy(); point[j] += step
            expected = (eq.residual(point)-eq.residual(x))/step
            np.testing.assert_allclose(jac[:, j], expected, rtol=1e-6, atol=1e-6)

    def test_heating_and_transport_input_rejection(self):
        c = copy.deepcopy(self.c); c['uniform_B_T'] = [.01, 0, .01]
        with self.assertRaises(ValueError): solve(geometry(), c)
        with self.assertRaises(ValueError): solve(geometry(), self.c, np.full((4, 3), -1.))
        with self.assertRaises(ValueError): solve(geometry(), self.c, np.ones((3, 4)))

    def test_map_covers_axis_and_walls_and_rejects_extrapolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.npz'; np.savez(path, **self.fields)
            profile = PlasmaMap(path)
            values = profile(np.array([[0, 0, -.12], [.03, 0, 0]]))
            self.assertEqual(values['electron_density_m3'][0], self.fields['electron_density_m3'][0, 0])
            with self.assertRaises(ValueError): profile([[.031, 0, -.06]])

    def test_variable_dielectric_passivity_and_scalar_equivalence(self):
        n = np.array([1e17, 1e18, 2e18]); nu = np.array([1e6, 2e6, 3e6])
        b = np.tile([0., 0., .014], (3, 1))
        eps = cold_epsilon(27.12e6, n, nu, b)
        for j in range(3):
            np.testing.assert_allclose(eps[j], cold_epsilon(27.12e6, n[j], nu[j], b[j]))
        q = absorbed_density(np.ones((3, 3), complex), eps, 27.12e6)
        self.assertTrue(np.all(q > 0))

    def test_integrated_power_transfer_ignores_visualization_samples(self):
        g = Grid(geometry(), 3, 4)
        zz, rr = np.meshgrid(g.z, g.r, indexing='ij')
        xyz = np.column_stack([rr.ravel(), np.zeros(g.size), zz.ravel()])
        # Two azimuthal samples per bin, nonuniform integrated powers.
        xyz = np.r_[xyz, xyz*np.array([-1., 1., 1.])]
        powers = np.arange(1, len(xyz)+1, dtype=float)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cells_rank0000.npz'
            data = dict(xyz_m=xyz, region=np.full(len(xyz), CELL['PLASMA']),
                        cell_absorbed_power_W=powers, cell_volume_m3=np.tile(g.volumes.ravel()/2, 2),
                        q_abs_W_m3=np.full(len(xyz), 1e100))
            np.savez(path, **data)
            q, info = heating_from_rf(tmp, g, powers.sum())
            self.assertAlmostEqual(np.sum(q*g.volumes), powers.sum())
            self.assertEqual(info['occupied_bin_fraction'], 1.)
            with self.assertRaises(ValueError): heating_from_rf(tmp, g, powers.sum()*2)
            del data['cell_absorbed_power_W']; np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, 'integrated cell'): heating_from_rf(tmp, g, powers.sum())


if __name__ == '__main__': unittest.main()
