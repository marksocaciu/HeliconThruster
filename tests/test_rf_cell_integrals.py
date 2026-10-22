"""Regression for DOLFINx 0.10/0.11 cell-integral vector return types."""
import os
from types import SimpleNamespace
import unittest
import numpy as np
from helicon.rf.solve import _owned_cell_integrals


class CellIntegralCompatibility(unittest.TestCase):
    def test_array_and_vector_results_map_owned_cells_without_ghosts(self):
        # Deliberately reorder DOFs and include a ghost entry with a large value.
        dofs = [2, 0, 1, 3]
        space = SimpleNamespace(dofmap=SimpleNamespace(cell_dofs=lambda i: [dofs[i]]))
        values = np.array([2+0j, 3+0j, 5+0j, 999+0j])
        for vector in (values, SimpleNamespace(array=values)):
            with self.subTest(vector_type=type(vector).__name__):
                result = _owned_cell_integrals(vector, space, 3)
                np.testing.assert_array_equal(result, [5., 2., 3.])
                self.assertAlmostEqual(result.sum(), 10.)
                self.assertTrue(np.isrealobj(result))
                self.assertFalse(np.shares_memory(result, values))

    def test_zero_owned_cells(self):
        space = SimpleNamespace(dofmap=SimpleNamespace(cell_dofs=lambda i: [i]))
        result = _owned_cell_integrals(SimpleNamespace(array=np.array([123.])), space, 0)
        self.assertEqual(result.shape, (0,))


@unittest.skipUnless(os.environ.get('HELICON_FEM_DIAGNOSTICS') == '1',
                     'opt-in DOLFINx cell-assembly check (real or complex runtime)')
class ActualCellAssembly(unittest.TestCase):
    def test_owned_cell_integrals_match_global_volume_and_heating(self):
        from mpi4py import MPI
        from dolfinx import fem, mesh
        import dolfinx
        import ufl
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 3, 3, 3,
                                       ghost_mode=mesh.GhostMode.shared_facet)
        space = fem.functionspace(domain, ('DG', 0))
        test = ufl.TestFunction(space)
        dx = ufl.Measure('dx', domain=domain)
        x = ufl.SpatialCoordinate(domain)
        density = 2+x[0]+3*x[2]
        jit = {'cache_dir': os.environ['HELICON_FEM_CACHE_DIR']} if os.environ.get('HELICON_FEM_CACHE_DIR') else {}
        volume = fem.assemble_vector(fem.form(ufl.conj(test)*dx, jit_options=jit))
        power = fem.assemble_vector(fem.form(density*ufl.conj(test)*dx, jit_options=jit))
        owned = domain.topology.index_map(domain.topology.dim).size_local
        cells_v = _owned_cell_integrals(volume, space, owned)
        cells_p = _owned_cell_integrals(power, space, owned)
        self.assertTrue(np.all(cells_v > 0))
        self.assertAlmostEqual(comm.allreduce(float(cells_v.sum()), op=MPI.SUM), 1., places=11)
        self.assertAlmostEqual(comm.allreduce(float(cells_p.sum()), op=MPI.SUM), 4., places=11)
        global_power = comm.allreduce(float(fem.assemble_scalar(fem.form(density*dx, jit_options=jit))), op=MPI.SUM)
        self.assertAlmostEqual(comm.allreduce(float(cells_p.sum()), op=MPI.SUM), global_power, places=11)
        if comm.rank == 0:
            print(f'Cell exports verified with DOLFINx {dolfinx.__version__}, {comm.size} MPI rank(s).')


if __name__ == '__main__': unittest.main()
