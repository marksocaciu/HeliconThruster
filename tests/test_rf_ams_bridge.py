import os,unittest

@unittest.skipUnless(os.environ.get("HELICON_FEM_DIAGNOSTICS")=="1", "opt-in real PETSc/hypre MPI bridge test")
class AMSBridgeTests(unittest.TestCase):
 def test_persistent_worker(self):
  import tempfile,shutil,sys
  from pathlib import Path
  from types import SimpleNamespace
  import numpy as np
  from mpi4py import MPI
  from petsc4py import PETSc
  from helicon.rf.ams_bridge import compare_ams_bridge
  from helicon.rf.solve import _solve_with_options
  from helicon.rf.diagnostics import linear_diagnostics
  comm=MPI.COMM_WORLD
  root=Path(comm.bcast(tempfile.mkdtemp(prefix='rf-bridge-test-') if comm.rank==0 else None,root=0))
  import ufl,basix.ufl
  from dolfinx import fem,mesh
  from dolfinx.fem import petsc as fp
  from helicon.rf.ams_transfer import export_system
  D=mesh.create_unit_cube(comm,2,2,2)
  V=fem.functionspace(D,basix.ufl.element('N1curl',D.basix_cell(),2))
  D.topology.create_connectivity(2,3)
  fixed=fem.locate_dofs_topological(V,2,mesh.exterior_facet_indices(D.topology))
  bc=fem.dirichletbc(fem.Function(V),fixed)
  u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
  P=fp.assemble_matrix(fem.form((ufl.inner(ufl.curl(u),ufl.curl(v))+ufl.inner(u,v))*ufl.dx),bcs=[bc])
  P.assemble();A=P.copy()
  b=P.createVecLeft();exact=P.createVecRight();exact.array[:]=np.sin(np.arange(*exact.getOwnershipRange())+1)
  P.mult(exact,b);x=exact.duplicate();x.set(0)
  k=PETSc.KSP().create(comm);k.setOptionsPrefix('baseline_');k.setOperators(A,P)
  p=SimpleNamespace(A=A,b=b,x=x,solver=k,solve=lambda:k.solve(b,x))
  opts={'ksp_type':'fgmres','pc_type':'asm','sub_pc_type':'ilu','ksp_rtol':1e-8,'ksp_max_it':30,'ksp_error_if_not_converged':False}
  export_system(p,V,2,fixed,root,comm)
  _solve_with_options(p,opts,PETSc);first=linear_diagnostics(p,PETSc,1e-8);saved=x.copy()
  r=compare_ams_bridge(p,opts,PETSc,comm,root,1e-8,first,cap_mb=None,rtol=.01,max_it=100,timeout=30,
   python=sys.executable,mpiexec=str(Path(sys.executable).parent/'mpiexec'),petsc_dir=sys.prefix)
  assert r['outer_result']['accepted'],r
  assert r['A_b_P_unchanged']
  assert r['worker_shutdown']['returncode']==0
  assert len(r['inner_applications'])>1
  x.axpy(-1,saved);assert x.norm()==0
  if comm.rank==0:print('PASS persistent worker, outer solve, fixed matrices, preserved original x, clean shutdown',root)
  from helicon.rf.ams_bridge import AMSBridge
  from helicon.rf.ams_protocol import atomic_npz,atomic_json,wait_json
  bridge=AMSBridge(P,comm,root/'ams-export',root/'complex-ipc',rtol=1e-8,timeout=30,
   cap_mb=None,python=sys.executable,mpiexec=str(Path(sys.executable).parent/'mpiexec'),petsc_dir=sys.prefix)
  try:
   bridge.start()
   atomic_npz(bridge.directory/f'rhs-{comm.rank}.npz',sequence=1,values=b.array*(1+2j))
   comm.barrier()
   bridge.root_action(lambda:atomic_json(bridge.directory/'command.json',{'sequence':1,'command':'solve'}))
   reply=bridge.root_action(lambda:wait_json(bridge.directory/'response.json',30,failed=bridge.directory/'failed.json',process=bridge.proc,sequence=1))
   bridge.sequence=2
   assert reply['passed'],reply
   with np.load(bridge.directory/f'answer-{comm.rank}.npz') as data: answer=data['values'].copy()
   z=P.createVecRight();res=b.duplicate()
   for part,factor in [(answer.real,1),(answer.imag,2)]:
    z.array[:]=part;P.mult(z,res);res.axpy(-factor,b)
    assert res.norm()/(factor*b.norm())<1e-8
   z.destroy();res.destroy()
  finally:bridge.close()
  assert bridge.shutdown['returncode']==0
  # An impossible launch must become a retained failure report, not hang MPI.
  failed=compare_ams_bridge
  k.setOperators(A,P)
  missing=root/'missing';missing.mkdir(exist_ok=True) if comm.rank==0 else None
  comm.barrier()
  r=failed(p,opts,PETSc,comm,missing,1e-8,first,cap_mb=None,mpiexec='/nonexistent/rf-mpiexec')
  assert r['status']=='bridge_failed',r
  if comm.rank==0:print('PASS complex transport and failed-launch cleanup')
  from helicon.rf.ams_bridge import compare_ams_tolerances
  k.setOperators(A,P)
  sweep=compare_ams_tolerances(p,opts,PETSc,comm,root,1e-8,first,rtols='0.01 0.001',
   cap_mb=None,python=sys.executable,mpiexec=str(Path(sys.executable).parent/'mpiexec'),petsc_dir=sys.prefix)
  assert sweep['status']=='completed',sweep
  assert len(sweep['attempts'])==2
  assert all(row['outer_result']['accepted'] and row['A_b_P_unchanged'] for row in sweep['attempts'])
  from helicon.rf.ams_bridge import compare_ams_masses
  coefficient=fem.Constant(D,PETSc.ScalarType(1))
  variable=fem.form((ufl.inner(ufl.curl(u),ufl.curl(v))+coefficient*ufl.inner(u,v))*ufl.dx)
  def rebuild(matrix,scale):
   coefficient.value=PETSc.ScalarType(scale)
   matrix.zeroEntries();fp.assemble_matrix(matrix,variable,bcs=[bc]);matrix.assemble()
  preserved=x.copy()
  masses=compare_ams_masses(p,opts,PETSc,comm,root,1e-8,first,[1,3,10],rebuild,
   rtol=.01,cap_mb=None,python=sys.executable,mpiexec=str(Path(sys.executable).parent/'mpiexec'),petsc_dir=sys.prefix)
  assert masses['status']=='completed',masses
  assert masses['original_A_b_P_unchanged']
  assert len(masses['attempts'])==3
  assert all(row['outer_result']['accepted'] and row['A_b_unchanged'] for row in masses['attempts'])
  x.axpy(-1,preserved);assert x.norm()==0;preserved.destroy()
  import json
  fingerprints=[]
  for index in range(1,4):
   trial=json.loads((root/f'ams-mass-{index}'/'ams_bridge_diagnostics.json').read_text())
   fingerprints.append(trial['P_before'])
  assert fingerprints[0]!=fingerprints[1]!=fingerprints[2]
  saved.destroy();k.destroy();exact.destroy();b.destroy();x.destroy();A.destroy();P.destroy()
