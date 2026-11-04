"""One-way transfer gates and a real transport solve with synthetic RF powers."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from helicon import spatial
from helicon.config import load as geometry
from helicon.records import read_json, write_json, digest
from helicon.rf.config import load as rf_config
from helicon.rf.mesh import CELL
from helicon.rf.prescribed import inspect_rf, normalized_heating, run


class PrescribedHeating(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'rf'; self.source.mkdir()
        self.gconfig = geometry()
        self.c = spatial.load(); self.c.update(nr=3, nz=4)
        self.grid = spatial.Grid(self.gconfig, 3, 4)
        self.config = self.root/'spatial.json'; write_json(self.config, self.c)
        rf = rf_config(); rf['material_case'] = 'plasma'
        write_json(self.source/'rf.json', rf)
        (self.source/'rf.msh').write_text('Synthetic fixture; no Maxwell solve')
        write_json(self.source/'mesh_report.json', {
            'mesh_sha256':digest(self.source/'rf.msh'), 'config_sha256':digest(self.source/'rf.json')})
        self.wave = {'status':'completed', 'case':'plasma', 'mpi_ranks':2,
            'input_power_W':2., 'material_absorption_W':{'PLASMA':2., 'WALL':0.},
            'pml_absorption_W':0., 'outward_interface_flux_W':0., 'pml_side_interface_flux_W':0.,
            'power_balance_relative_error':0., 'free_equation_relative_residual':1e-12,
            'numerical_checks_passed':True,
            'numerical_checks':dict.fromkeys(('finite_power_values', 'positive_input_power',
                'power_balance', 'free_equation_residual', 'nonnegative_material_absorption',
                'nonnegative_pml_absorption', 'nonnegative_outward_interface_flux',
                'pml_interface_flux_agreement', 'interface_trace_agreement'), True)}
        z, r = np.meshgrid(self.grid.z, self.grid.r, indexing='ij')
        xyz = np.column_stack((r.ravel(), np.zeros(self.grid.size), z.ravel()))
        powers = 2*self.grid.volumes.ravel()/self.grid.volumes.sum()
        for i, ids in enumerate(np.array_split(np.arange(self.grid.size),2)):
            np.savez(self.source/f'cells_rank{i:04d}.npz', xyz_m=xyz[ids],
                region=np.full(len(ids), CELL['PLASMA']), cell_volume_m3=self.grid.volumes.ravel()[ids],
                cell_absorbed_power_W=powers[ids])
        self.seal()

    def seal(self):
        write_json(self.source/'rf_summary.json', self.wave)
        write_json(self.source/'run.json', {'status':'completed', 'artifacts_sha256':{
            p.name:digest(p) for p in self.source.iterdir() if p.is_file() and p.name != 'run.json'}})

    def test_normalization_preserves_shape_and_raw_power(self):
        # Unequal heating per volume must survive transfer and normalization.
        weights = np.arange(1,self.grid.size+1,dtype=float)
        powers = weights*self.grid.volumes.ravel()
        powers *= 2/powers.sum()
        for i, ids in enumerate(np.array_split(np.arange(self.grid.size),2)):
            path=self.source/f'cells_rank{i:04d}.npz'
            with np.load(path) as data: arrays={key:data[key].copy() for key in data.files}
            arrays['cell_absorbed_power_W']=powers[ids]
            np.savez(path,**arrays)
        raw, q, info = normalized_heating(self.source, self.grid, 2., 100.)
        self.assertAlmostEqual(np.sum(raw*self.grid.volumes),2.)
        self.assertAlmostEqual(np.sum(q*self.grid.volumes),100.)
        np.testing.assert_allclose(q,raw*50)
        self.assertAlmostEqual(q.ravel()[-1]/q.ravel()[0],self.grid.size)
        self.assertEqual(info['occupied_bin_fraction'],1.)

    def test_conservation_checked_before_normalization(self):
        with self.assertRaisesRegex(ValueError,'conserve power'):
            normalized_heating(self.source,self.grid,3.,100.)

    def test_missing_rank_and_changed_export_rejected(self):
        path = self.source/'cells_rank0001.npz'
        data = path.read_bytes(); path.unlink()
        with self.assertRaisesRegex(ValueError,'rank set'): inspect_rf(self.source,self.gconfig)
        path.write_bytes(data+b'changed')
        with self.assertRaisesRegex(ValueError,'artifact hash'): inspect_rf(self.source,self.gconfig)

    def test_negative_interface_requires_explicit_provisional_option(self):
        self.wave.update(outward_interface_flux_W=-1e-4,pml_side_interface_flux_W=-1e-4)
        self.seal()
        with self.assertRaisesRegex(ValueError,'exploratory-interface'):
            inspect_rf(self.source,self.gconfig)
        _,_,_,info = inspect_rf(self.source,self.gconfig,True)
        self.assertTrue(info['negative_interface_flux_used'])

    def test_interface_option_does_not_allow_negative_pml_or_large_flux(self):
        self.wave['pml_absorption_W']=-1e-5;self.seal()
        with self.assertRaisesRegex(ValueError,'Negative material/PML'):
            inspect_rf(self.source,self.gconfig,True)
        self.wave.update(pml_absorption_W=0.,outward_interface_flux_W=-.1);self.seal()
        with self.assertRaisesRegex(ValueError,'scalar diagnostics'):
            inspect_rf(self.source,self.gconfig,True)

    def test_false_checks_and_geometry_mismatch_rejected(self):
        self.wave['numerical_checks']['power_balance']=False;self.seal()
        with self.assertRaisesRegex(RuntimeError,'numerical checks'): inspect_rf(self.source,self.gconfig)
        other=copy.deepcopy(self.gconfig);other['chamber']['radius']*=2
        with self.assertRaisesRegex(ValueError,'geometry differ'): inspect_rf(self.source,other)

    def test_empty_bins_and_nonpositive_power_rejected(self):
        fine=spatial.Grid(self.gconfig,6,8)
        with self.assertRaisesRegex(ValueError,'empty heating bins'):
            normalized_heating(self.source,fine,2.,100.)
        for power in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError): normalized_heating(self.source,self.grid,2.,power)

    def test_real_transport_with_synthetic_uniform_rf_shape(self):
        out=self.root/'out'
        result=run(self.source,out,100.,self.config,plots=False)
        self.assertEqual(result['status'],'completed')
        self.assertFalse(result['rf_feedback_performed'])
        self.assertTrue((out/'report.html').exists())
        self.assertEqual(read_json(out/'run.json')['status'],'completed')
        for case in ('uniform','rf-shaped'):
            s=read_json(out/case/'spatial_summary.json')
            self.assertLess(s['balances']['relative_power_error'],1e-5)
            self.assertEqual(s['status'],'converged')
        before=(out/'run.json').read_bytes()
        with self.assertRaises(FileExistsError):run(self.source,out,100.,self.config,plots=False)
        self.assertEqual(before,(out/'run.json').read_bytes())

    def test_failure_retains_diagnostics_and_does_not_normalize_bad_input(self):
        out=self.root/'failed'
        with patch('helicon.spatial.solve',side_effect=RuntimeError('transport failed')):
            with self.assertRaisesRegex(RuntimeError,'transport failed'):
                run(self.source,out,100.,self.config,plots=False)
        record=read_json(out/'run.json')
        self.assertEqual(record['status'],'failed')
        self.assertEqual(record['stage'],'uniform_transport')
        self.assertTrue((out/'transfer.json').exists())
        self.assertFalse((out/'experiment_summary.json').exists())


if __name__=='__main__':unittest.main()
