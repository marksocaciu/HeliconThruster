"""Uniform cylinder waves, complex modal reconstruction, and the axis treatment."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helicon.config import ROOT
from helicon.rf.materials import absorbed_density, cold_epsilon
from helicon.rf_modes.__main__ import execute, validate
from helicon.rf_modes.analysis import reconstruct
from helicon.rf_modes.meridian import basis_at


class TestModes(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'config/rf_modes_uniform.json').read_text())

    def test_signed_dispersion_and_independent_three_dimensional_field(self):
        self.config['nr'] = 3
        self.config['modes'][1]['amplitude_V_m'] = [0.2, -0.3]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'waves'
            result = execute(self.config, dest, compare_3d=True, phi_deg=[37])
            self.assertEqual(result['status'], 'completed', result['checks'])
            self.assertLess(result['reference_3d']['differences']['E_relative_error'], 1e-7)
            self.assertLess(result['reference_3d']['differences']['B_relative_error'], 1e-7)
            for mode in result['modes'].values():
                self.assertLess(mode['analytical_benchmark']['k_relative_error'], 1e-3)
            with np.load(dest/'heating_rz.npz') as heating:
                power = (heating['q_abs_W_m3']*heating['cell_volume_m3']).sum()
                self.assertAlmostEqual(power, result['total_absorbed_W'], places=12)
            with self.assertRaises(FileExistsError):
                execute(self.config, dest)

    def test_axis_and_nontrivial_azimuthal_heating(self):
        self.config['nr'] = 5
        self.config['modes'] = [dict(m=0, amplitude_V_m=[1., 0.]), dict(m=2, amplitude_V_m=[0., .7])]
        epsilon = cold_epsilon(self.config['frequency_Hz'], self.config['electron_density_m3'],
                               self.config['collision_frequency_s'], [0, 0, self.config['Bz_T']])
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'waves'
            result = execute(self.config, dest)
            self.assertEqual(result['status'], 'completed')
            modes = {}
            for m in [0, 2]:
                with np.load(dest/result['modes'][str(m)]['file']) as archive:
                    modes[m] = {k: archive[k] for k in ('E_cyl_peak_V_m', 'B_cyl_peak_T')}
            angles = np.arange(12)*2*np.pi/12
            heat = [absorbed_density(reconstruct(modes, angle)[0], epsilon, self.config['frequency_Hz'])
                    for angle in angles]
            diagonal = sum(absorbed_density(f['E_cyl_peak_V_m'], epsilon, self.config['frequency_Hz'])
                           for f in modes.values())
            np.testing.assert_allclose(np.mean(heat, axis=0), diagonal, rtol=1e-11)
            self.assertGreater(np.linalg.norm(heat[0]-diagonal), 0.01*np.linalg.norm(diagonal))
        self.config['modes'].append(self.config['modes'][0])
        with self.assertRaises(ValueError):
            validate(self.config)

    def test_radial_curl_for_regular_axis_fields(self):
        dr, dz = .03, .1
        for m in (0, 1, -1, 2):
            basis, curl = basis_at(0, dr, dz, m, .2, .6)
            if m:
                # grad(r exp(i m phi)) is curl free; analytic at the axis for |m|=1.
                dofs = np.array([1, 1, 0, 0, 0, 1j*m*dr, 0, 1j*m*dr])
                np.testing.assert_allclose(dofs @ curl, 0, atol=1e-11)
                np.testing.assert_allclose(dofs @ basis, [1, 1j*m, 0])
            else:
                dofs = np.array([0, 0, 0, 0, 0, dr**2, 0, dr**2])
                np.testing.assert_allclose(dofs @ curl, [0, 0, 2], atol=1e-11)
