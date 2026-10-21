"""Driver contract tests with synthetic RF exports; no Maxwell solve is claimed."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from helicon import spatial
from helicon.config import load as geometry
from helicon.records import write_json, read_json
from helicon.rf.config import load as rf_config
from helicon.rf.mesh import CELL
from helicon.rf.spatial_coupled import couple, state_change, validate_spatial


class SpatialCoupling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = spatial.load(); cls.c.update(nr=3, nz=4)
        cls.summary, cls.fields = spatial.solve(geometry(), cls.c)

    def prepare(self, root):
        build = root/'build'; build.mkdir()
        rf = rf_config(); rf['material_case'] = 'plasma'
        write_json(build/'rf.json', rf)
        (build/'rf.msh').write_text('Synthetic mesh placeholder; never passed to FEM')
        write_json(build/'mesh_report.json', {})
        config = root/'spatial.json'; write_json(config, self.c)
        return build, config

    def fake_rf(self, build, out, state_path, tolerance, rf_ranks=1, mpi_launcher='mpiexec'):
        c = read_json(Path(build)/'rf.json')
        pin = 2*c['voltage_peak_V']**2
        out = Path(out); out.mkdir()
        g = spatial.Grid(geometry(), self.c['nr'], self.c['nz'])
        z, r = np.meshgrid(g.z, g.r, indexing='ij')
        xyz = np.column_stack((r.ravel(), np.zeros(g.size), z.ravel()))
        np.savez(out/'cells_rank0000.npz', xyz_m=xyz, region=np.full(g.size, CELL['PLASMA']),
                 cell_volume_m3=g.volumes.ravel(),
                 cell_absorbed_power_W=.75*pin*g.volumes.ravel()/g.volumes.sum())
        return {'input_power_W': pin, 'material_absorption_W': {'PLASMA': .75*pin},
                'power_balance_relative_error': 0.}

    def test_actual_transport_closes_with_conservative_synthetic_rf_heating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); build, config = self.prepare(root)
            with patch('helicon.rf.solve.runtime', return_value=(SimpleNamespace(COMM_WORLD=SimpleNamespace(size=1)), None, None)), \
                 patch('helicon.rf.spatial_coupled.run_rf', side_effect=self.fake_rf):
                result = couple(build, root/'out', 100/.75, config, max_iterations=3)
            self.assertEqual(result['status'], 'converged')
            self.assertEqual(len(result['history']), 2)
            for entry in result['history']:
                self.assertLess(entry['transfer']['relative_power_error'], 1e-12)
                self.assertLess(entry['feed_power_relative_error'], 1e-12)
            self.assertTrue((root/'out/final_rf_input_plasma.npz').exists())
            self.assertTrue((root/'out/final_plasma/report.html').exists())

    def test_tiny_relaxation_cannot_hide_unrelaxed_fixed_point_error(self):
        calls = 0
        def divergent_response(*args, **kwargs):
            nonlocal calls
            calls += 1
            fields = copy.deepcopy(self.fields if calls == 1 else kwargs['initial'])
            if calls != 1:
                fields['electron_density_m3'] *= 2
            return copy.deepcopy(self.summary), fields
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); build, config = self.prepare(root)
            with patch('helicon.rf.solve.runtime', return_value=(SimpleNamespace(COMM_WORLD=SimpleNamespace(size=1)), None, None)), \
                 patch('helicon.rf.spatial_coupled.run_rf', side_effect=self.fake_rf), \
                 patch('helicon.spatial.solve', side_effect=divergent_response):
                with self.assertRaisesRegex(RuntimeError, 'did not converge'):
                    couple(build, root/'out', 100/.75, config, max_iterations=2, relaxation=1e-8)
            record = read_json(root/'out/run.json')
            self.assertEqual(record['status'], 'not_converged')
            history = read_json(root/'out/coupling_summary.json')['history']
            self.assertGreater(history[-1]['state_relative_changes']['electron_density_m3'], .4)

    def test_failed_spatial_balance_rejected_even_if_status_says_converged(self):
        summary = copy.deepcopy(self.summary)
        summary['balances']['relative_power_error'] = .1
        with self.assertRaises(RuntimeError): validate_spatial(summary)

    def test_potential_change_is_gauge_invariant(self):
        shifted = copy.deepcopy(self.fields)
        shifted['potential_V'] += 15
        self.assertLess(state_change(self.fields, shifted)['potential_V'], 1e-12)


if __name__ == '__main__': unittest.main()
