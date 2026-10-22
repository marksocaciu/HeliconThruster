"""Manufactured two-sided power traces; no Maxwell solve or PML fitting."""
import os
import unittest

@unittest.skipUnless(os.environ.get('HELICON_FEM_DIAGNOSTICS') == '1',
                     'opt-in DOLFINx interface-assembly check')
class InterfaceFluxTests(unittest.TestCase):
    def test_material_selected_traces_and_reversed_orientation(self):
        import numpy as np
        import ufl
        from mpi4py import MPI
        from dolfinx import mesh, fem
        from helicon.rf.forms import interface_flux, interface_flux_traces
        from helicon.rf.materials import MU0
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 1, 1, 2,
                                       ghost_mode=mesh.GhostMode.shared_facet)
        domain.topology.create_connectivity(2, 3)
        facets = mesh.locate_entities(domain, 2, lambda x: np.isclose(x[2], .5))
        tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 14, np.int32))
        ds = ufl.Measure('dS', domain=domain, subdomain_data=tags)
        V = fem.functionspace(domain, ('DG', 1, (3,)))
        E = fem.Function(V, dtype=np.complex128)
        frequency = 1.
        w = 2*np.pi*frequency
        # At z=.5, E=(1,0,0). H_y is 2 below and 6 above.
        def field(x):
            a = np.where(x[2] < .5, 2., 6.)*w*MU0
            values = np.zeros(x.shape, dtype=np.complex128)
            values[0] = 1+1j*a*(x[2]-.5)
            return values
        E.interpolate(field)
        E.x.scatter_forward()
        Q = fem.functionspace(domain, ('DG', 0))
        indicator = fem.Function(Q, dtype=np.complex128)
        jit = {'cache_dir': os.environ['HELICON_FEM_CACHE_DIR']} if os.environ.get('HELICON_FEM_CACHE_DIR') else {}
        def integrate(expr):
            f = fem.form(expr*ds(14), dtype=np.complex128, jit_options=jit)
            return float(np.real(comm.allreduce(fem.assemble_scalar(f), op=MPI.SUM)))
        indicator.interpolate(lambda x: (x[2] < .5).astype(np.complex128))
        indicator.x.scatter_forward()
        a, b = interface_flux_traces(domain, E, frequency, indicator)
        self.assertAlmostEqual(integrate(a), 1., places=9)
        self.assertAlmostEqual(integrate(b), 3., places=9)
        self.assertAlmostEqual(integrate(interface_flux(domain, E, frequency, (0,0,0))), 2., places=9)
        indicator.x.array[:] = 1-indicator.x.array
        indicator.x.scatter_forward()
        a, b = interface_flux_traces(domain, E, frequency, indicator)
        self.assertAlmostEqual(integrate(a), -3., places=9)
        self.assertAlmostEqual(integrate(b), -1., places=9)

if __name__ == '__main__':
    unittest.main()
