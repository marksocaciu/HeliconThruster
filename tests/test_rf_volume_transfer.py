"""Independent geometric fixtures for cell matching and volume-power deposition."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.stats import qmc

from helicon import spatial
from helicon.records import write_json
from helicon.rf.volume_transfer import read_tetrahedra,map_cells,deposit,heating


def write_mesh(path,tets):
    n=len(tets);vertices=np.asarray(tets).reshape(-1,3);tags=np.arange(len(vertices))*3+101
    text='$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Entities\n0 0 0 1\n10 0 0 0 1 1 1 1 1 0\n$EndEntities\n'
    text+=f'$Nodes\n1 {len(vertices)} {tags.min()} {tags.max()}\n3 10 0 {len(vertices)}\n'
    text+='\n'.join(map(str,tags))+'\n'
    text+='\n'.join(' '.join(map(str,v)) for v in vertices)+'\n$EndNodes\n'
    text+=f'$Elements\n1 {n} 50 {50+n-1}\n3 10 4 {n}\n'
    for i in range(n):text+=f'{50+i} '+' '.join(map(str,tags[i*4:i*4+4]))+'\n'
    path.write_text(text+'$EndElements\n')


class VolumeTransfer(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.tets=np.array([[[0,0,0],[.01,0,0],[0,.01,0],[0,0,1]],
                            [[0,0,1],[.01,0,1],[0,.01,1],[0,0,0]]],float)
        write_mesh(self.root/'rf.msh',self.tets)
        write_json(self.root/'rf_summary.json',{'mpi_ranks':2})
        self.vol=np.abs(np.linalg.det(self.tets[:,1:]-self.tets[:,:1]))/6
        # Exports deliberately reverse mesh order across ranks.
        for rank,index in enumerate((1,0)):
            np.savez(self.root/f'cells_rank{rank:04d}.npz',xyz_m=self.tets[index:index+1].mean(axis=1),
                region=np.ones(1,dtype=int),cell_volume_m3=self.vol[index:index+1],
                cell_absorbed_power_W=np.array([float(index+1)]))
        self.grid=spatial.Grid({'chamber':{'radius':.02,'length':1.,'z_start':0.}},1,2)

    def test_reads_noncontiguous_nodes_and_matches_reversed_mpi_exports(self):
        ids,vertices,region,volume=read_tetrahedra(self.root/'rf.msh')
        np.testing.assert_array_equal(ids,[50,51])
        np.testing.assert_allclose(volume,self.vol)
        arrays,mapped,info=map_cells(self.root,self.root/'mapping.npz')
        np.testing.assert_array_equal(mapped,self.tets[::-1])
        self.assertEqual(info['matched_cells'],2)
        arrays,cached,again=map_cells(self.root,self.root/'cached.npz',self.root/'mapping.npz')
        np.testing.assert_array_equal(cached,mapped)
        self.assertTrue(again['loaded_from_cache'])

    def test_ambiguous_geometry_and_duplicate_exports_are_rejected(self):
        write_mesh(self.root/'rf.msh',np.repeat(self.tets[:1],2,axis=0))
        with self.assertRaisesRegex(ValueError,'ambiguous or missing'):
            map_cells(self.root,self.root/'bad.npz')
        write_mesh(self.root/'rf.msh',self.tets)
        first=self.root/'cells_rank0000.npz';second=self.root/'cells_rank0001.npz'
        second.write_bytes(first.read_bytes())
        with self.assertRaisesRegex(ValueError,'bijection'):
            map_cells(self.root,self.root/'bad2.npz')

    def test_changed_volume_and_cache_identity_rejected(self):
        map_cells(self.root,self.root/'map.npz')
        path=self.root/'cells_rank0000.npz'
        with np.load(path) as d:a={k:d[k].copy() for k in d.files}
        a['cell_volume_m3']*=1.01;np.savez(path,**a)
        with self.assertRaisesRegex(ValueError,'identity/content'):
            map_cells(self.root,self.root/'cache.npz',self.root/'map.npz')
        with self.assertRaisesRegex(ValueError,'ambiguous or missing'):
            map_cells(self.root,self.root/'fresh.npz')

    def test_analytic_tetrahedron_plane_fraction_and_per_cell_power_conservation(self):
        u=qmc.Sobol(d=4,scramble=True,seed=20261105).random_base2(15)
        b=-np.log(u);b/=b.sum(axis=1,keepdims=True)
        # For this tetrahedron, volume at z > .5 is exactly (1-.5)^3 = 1/8.
        p,v=deposit(self.tets[:1],np.array([8.]),self.vol[:1],self.grid,b)
        self.assertAlmostEqual(float(p.sum()),8.,places=11)
        self.assertAlmostEqual(float(v.sum()),self.vol[0],places=14)
        self.assertLess(abs(p[1,0]-1.),.015)
        # A zero-power tetrahedron still deposits geometric coverage.
        p,v=deposit(self.tets[:1],np.zeros(1),self.vol[:1],self.grid,b)
        self.assertEqual(float(p.sum()),0.)
        self.assertTrue(np.all(v>0))

    def test_two_level_gate_retains_report_and_rejects_uncovered_bins(self):
        q,info=heating(self.root,self.grid,3.,self.root/'good',samples=2048,rtol=.01)
        self.assertTrue(info['sampling_checks_passed'])
        self.assertAlmostEqual(float(np.sum(q*self.grid.volumes)),3.,places=12)
        missing=spatial.Grid({'chamber':{'radius':.04,'length':1.,'z_start':0.}},2,2)
        with self.assertRaisesRegex(ValueError,'checks failed'):
            heating(self.root,missing,3.,self.root/'bad',samples=256)
        report=json.loads((self.root/'bad/transfer_diagnostics.json').read_text())
        self.assertFalse(report['sampling_checks_passed'])
        self.assertLess(report['occupied_bin_fraction'],1.)
        self.assertTrue((self.root/'bad/transfer_samples.npz').exists())

    def test_unstable_sampling_is_rejected_even_when_all_bins_are_covered(self):
        thirds=spatial.Grid({'chamber':{'radius':.02,'length':1.,'z_start':0.}},1,3)
        with self.assertRaisesRegex(ValueError,'checks failed'):
            heating(self.root,thirds,3.,self.root/'unstable',samples=16,rtol=1e-12)
        report=json.loads((self.root/'unstable/transfer_diagnostics.json').read_text())
        self.assertEqual(report['occupied_bin_fraction'],1.)
        self.assertFalse(report['sampling_checks_passed'])
        self.assertGreater(report['power_distribution_L1_relative_change'],1e-12)


if __name__=='__main__':unittest.main()
