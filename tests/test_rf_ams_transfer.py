import os
import unittest

@unittest.skipUnless(os.environ.get("HELICON_FEM_DIAGNOSTICS")=="1", "opt-in real PETSc/hypre test")
class AMSTransferTests(unittest.TestCase):
    def test_boundary_system_round_trip(self):
        from pathlib import Path
        from types import SimpleNamespace
        import numpy as np
        import ufl,basix.ufl
        from mpi4py import MPI
        from petsc4py import PETSc
        from dolfinx import fem,mesh
        from dolfinx.fem import petsc as fp
        from helicon.rf.ams_transfer import export_system
        comm=MPI.COMM_WORLD
        D=mesh.create_unit_cube(comm,2,2,2)
        V=fem.functionspace(D,basix.ufl.element('N1curl',D.basix_cell(),2))
        D.topology.create_connectivity(2,3)
        fixed=fem.locate_dofs_topological(V,2,mesh.exterior_facet_indices(D.topology))
        bc=fem.dirichletbc(fem.Function(V),fixed)
        u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
        P=fp.assemble_matrix(fem.form((ufl.inner(ufl.curl(u),ufl.curl(v))+ufl.inner(u,v))*ufl.dx),bcs=[bc]);P.assemble()
        x=P.createVecRight();x.array[:]=np.sin(np.arange(*x.getOwnershipRange())+1)
        b=P.createVecLeft();P.mult(x,b);x.set(0)
        k=PETSc.KSP().create(comm);k.setOperators(P,P)
        import tempfile
        root=Path(comm.bcast(tempfile.mkdtemp() if comm.rank==0 else None,root=0))
        export_system(SimpleNamespace(A=P,b=b,x=x,solver=k),V,2,fixed,root,comm)
        k.destroy();x.destroy();b.destroy();P.destroy()
        
        from unittest.mock import patch
        from helicon.rf.ams_offline import main
        with patch('sys.argv',['ams_offline','--export',str(root/'ams-export'),'--out',str(root/'result.json')]):
            assert main()==0
