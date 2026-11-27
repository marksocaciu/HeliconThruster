"""Regression and independent element checks for imported RF material."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial.legendre import leggauss

from helicon.records import digest, read_json, write_json
from helicon.rf.config import load
from helicon.rf.materials import EPS_0, MU0, absorbed_density, cold_epsilon
from helicon.rf_modes import nagoya, slices
from helicon.rf_modes.analysis import reconstruct
from helicon.rf_modes.fem import assemble
from helicon.rf_modes.meridian import basis_at


class MappedNagoya(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'rf'
        self.source.mkdir()
        self.config = load()
        self.plasma = self.config['plasma']
        self.args = SimpleNamespace(config=None, radial_cells=[2, 1, 1, 1], outer_radius=.06,
                                    current=1., nz=4, max_m=3, modal_only=True, sectors=[8],
                                    material_from_rf=self.source, out=self.root/'mapped')

    def write_source(self, *, bands=False, magnetic_map=False):
        directory = self.source
        config = dict(self.config, material_case='plasma')
        write_json(directory/'rf.json', config)
        h, p = config['chamber'], self.plasma
        r = np.linspace(0, h['radius_m'], 3)
        z = h['z_start_m'] + np.linspace(0, h['length_m'], 3)
        density = np.full((2, 2), p['electron_density_m3'])
        collision = np.full_like(density, p['collision_frequency_Hz'])
        if bands:
            density[1] *= .5
            collision[1] *= 3
        np.savez(directory/'input_plasma_fields.npz', r_faces_m=r, z_faces_m=z,
                 electron_density_m3=density, collision_frequency_s=collision,
                 electron_temperature_eV=np.full_like(density, p['electron_temperature_eV']),
                 neutral_density_m3=np.full_like(density, p['neutral_density_m3']))
        if magnetic_map:
            zz, rr = np.meshgrid(z, r, indexing='ij')
            np.savez(directory/'background_B.npz', r_m=r, z_m=z, Br_T=.1*rr,
                     Bz_T=.014 + .01*(zz-z[0]))
        hashes = {path.name: digest(path) for path in directory.iterdir() if path.name != 'run.json'}
        write_json(directory/'run.json', dict(status='completed', config_sha256=hashes['rf.json'],
                   plasma_map_sha256=hashes['input_plasma_fields.npz'],
                   magnetic_map_sha256=hashes.get('background_B.npz'), artifacts_sha256=hashes))

    def test_constant_map_recovers_each_signed_mode_and_transport_heating(self):
        self.write_source()
        mapped = nagoya.run(self.args)
        uniform_dir = self.root / 'uniform'
        uniform = nagoya.run(SimpleNamespace(**{**vars(self.args), 'out': uniform_dir, 'material_from_rf': None}))
        self.assertEqual(mapped['status'], 'completed')
        for actual, reference in zip(mapped['modes'], uniform['modes'], strict=True):
            self.assertEqual(actual['m'], reference['m'])
            np.testing.assert_allclose(actual['plasma_W'], reference['plasma_W'], rtol=1e-10, atol=0)
            with np.load(self.args.out/actual['file']) as a, np.load(uniform_dir/reference['file']) as b:
                np.testing.assert_allclose(a['cell_absorbed_W'], b['cell_absorbed_W'], rtol=1e-10, atol=1e-18)
        with np.load(self.args.out/'heating_rz.npz') as a, np.load(uniform_dir/'heating_rz.npz') as b:
            self.assertEqual(set(a.files), set(b.files))
            for key in b.files:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-18)

    def test_axial_bands_with_radial_field_and_offline_slices(self):
        self.write_source(bands=True, magnetic_map=True)
        self.args.max_m = 1
        with patch('helicon.rf_modes.meridian.assemble', wraps=assemble) as assembly:
            result = nagoya.run(self.args)
        self.assertTrue(all(row['passed'] for row in result['modes']))
        with np.load(self.args.out/'material_rz.npz') as data:
            r, z, epsilon = data['r_faces_m'], data['z_faces_m'], data['epsilon_r_cyl']
        _, nr = epsilon.shape[:2]
        blocks = assembly.call_args_list[0].args[1]  # +1 mode, (z,r) cell ordering
        nodes, weights = leggauss(4)
        omega = 2*np.pi*self.config['frequency_Hz']
        modes = {}
        for row in result['modes']:
            with np.load(self.args.out/row['file']) as data:
                modes[row['m']] = dict(data)
        fields = modes[1]
        for j in (0, 3):
            i = 1
            dr, dz = r[i+1]-r[i], z[j+1]-z[j]
            br = .1*(r[i]+dr/2)
            bz = .014+.01*(z[j]+dz/2-z[0])
            tensor = cold_epsilon(self.config['frequency_Hz'], self.plasma['electron_density_m3']*(1 if j==0 else .5),
                                  self.plasma['collision_frequency_Hz']*(1 if j==0 else 3), [br, 0., bz])
            np.testing.assert_allclose(epsilon[j, i], tensor, rtol=1e-12)
            coeff = np.r_[fields['Er_edge_V_m'][j:j+2, i], fields['Ez_edge_V_m'][j, i:i+2],
                          fields['rEphi_vertex_V'][j:j+2, i:i+2].ravel()]
            block = np.zeros((8, 8), complex)
            watts = 0.
            for u, wu in zip((nodes+1)/2, weights/2):
                for v, wv in zip((nodes+1)/2, weights/2):
                    basis, curl = basis_at(r[i], dr, dz, 1, u, v)
                    dv = 2*np.pi*(r[i]+dr*u)*dr*dz*wu*wv
                    block += dv*(curl.conj() @ curl.T - omega**2*MU0*EPS_0*basis.conj() @ tensor @ basis.T)
                    e = coeff @ basis
                    watts += dv*.5*omega*EPS_0*np.vdot(e, tensor @ e).imag
            np.testing.assert_allclose(blocks[j*nr+i], block, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(fields['cell_absorbed_W'][j,i], watts, rtol=1e-10, atol=0)

        self.source.rename(self.root/'archived_source')
        dest = self.root/'slices'
        with patch('sys.argv', ['slices', '--run', str(self.args.out), '--out', str(dest), '--phi-deg', '37']):
            slices.main()
        e, _ = reconstruct(modes, np.deg2rad(37))
        with np.load(dest/'slice_000.npz') as data:
            np.testing.assert_allclose(data['q_local_W_m3'], absorbed_density(e, epsilon, omega/(2*np.pi)),
                                       rtol=1e-12, atol=0)

    def test_uniform_3d_input_fallback_and_chamber_check(self):
        self.write_source()
        record = read_json(self.source/'run.json')
        (self.source/'input_plasma_fields.npz').unlink()
        del record['artifacts_sha256']['input_plasma_fields.npz']
        record['plasma_map_sha256'] = None
        other = dict(self.config, material_case='plasma', plasma=dict(self.plasma, uniform_B_T=[0., 0., -.02]))
        write_json(self.source/'rf.json', other)
        record['artifacts_sha256']['rf.json'] = record['config_sha256'] = digest(self.source/'rf.json')
        write_json(self.source/'run.json', record)
        nagoya.run(self.args)
        with np.load(self.args.out/'material_rz.npz') as data:
            expected = cold_epsilon(self.config['frequency_Hz'], self.plasma['electron_density_m3'],
                                    self.plasma['collision_frequency_Hz'], [0., 0., -.02])
            np.testing.assert_allclose(data['epsilon_r_cyl'][0, 0], expected)

        shifted = dict(self.config, chamber=dict(self.config['chamber'], z_start_m=-.11))
        self.args.config = self.root/'shifted.json'
        write_json(self.args.config, shifted)
        self.args.out = self.root/'shifted'
        with self.assertRaisesRegex(ValueError, 'z_start_m'):
            nagoya.run(self.args)

    def test_provenance_and_prism_guards(self):
        self.write_source()
        self.args.modal_only = False
        with self.assertRaisesRegex(ValueError, 'modal-only'):
            nagoya.run(self.args)
        self.args.modal_only = True
        with (self.source/'input_plasma_fields.npz').open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'RF input hash mismatch'):
            nagoya.run(self.args)
        self.assertEqual(read_json(self.args.out/'run.json')['status'], 'failed')
