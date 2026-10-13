"""Serial FEniCSx (0.10) P2 axisymmetric A_theta starter solver."""
import argparse
import json
from pathlib import Path
import numpy as np
from .mesh import gmsh_module
from .config import load
from .reference import MU0, pack_bz
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, geometry, io
from dolfinx.io import gmsh
from dolfinx.fem.petsc import LinearProblem


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',default='build',help='mesh directory; uses its saved configuration')
    a=p.parse_args(); out=Path(a.build); c=load(out/'geometry.json')
    gmsh_module()  # Also makes Homebrew's Python API importable by DOLFINx.
    comm=MPI.COMM_WORLD
    if comm.size != 1:
        raise SystemExit('This initial solver and point sampling are serial; run without mpirun.')
    data=gmsh.read_from_msh(out/'helicon.msh',comm,rank=0,gdim=2)
    mesh,cells,facets=data.mesh,data.cell_tags,data.facet_tags
    if cells is None or facets is None:
        raise ValueError('Mesh lacks physical region/boundary tags')
    V=fem.functionspace(mesh,('Lagrange',2))
    A,v=ufl.TrialFunction(V),ufl.TestFunction(V)
    r=ufl.SpatialCoordinate(mesh)[0]
    dx=ufl.Measure('dx',domain=mesh,subdomain_data=cells,metadata={'quadrature_degree':6})
    # B_r=-dA/dz, B_z=dA/dr+A/r; common 2*pi factor cancels.
    form=(A.dx(1)*v.dx(1)+(A.dx(0)+A/r)*(v.dx(0)+v/r))*r/MU0*dx
    rhs=0*v*dx
    for i,k in enumerate(c['coils']):
        J=k['turns']*k['current']/(k['radial_thickness']*k['axial_width'])
        rhs += J*v*r*dx(101+i)
    ids=np.unique(np.concatenate([facets.find(11),facets.find(12)]))
    dofs=fem.locate_dofs_topological(V,1,ids)
    bc=fem.dirichletbc(PETSc.ScalarType(0),dofs,V)
    problem=LinearProblem(form,rhs,bcs=[bc],petsc_options_prefix='helicon_',
                          petsc_options={'ksp_type':'preonly','pc_type':'lu'})
    solution=problem.solve(); solution.name='A_theta'
    if problem.solver.getConvergedReason() <= 0:
        raise RuntimeError('PETSc solve did not converge')
    solution.x.scatter_forward()
    # Quadrature points are strictly inside cells: no r=0 division during assembly.
    Br=-solution.dx(1)
    Bz=solution.dx(0)+solution/r
    integrate=lambda expr: float(np.real(fem.assemble_scalar(fem.form(expr))))
    volume=integrate(2*np.pi*r*dx(1))
    mean=integrate(Bz*2*np.pi*r*dx(1))/volume
    std=np.sqrt(max(0,integrate((Bz-mean)**2*2*np.pi*r*dx(1))/volume))
    radial_rms=np.sqrt(max(0,integrate(Br**2*2*np.pi*r*dx(1))/volume))
    # Regularity gives B_z(0,z)=2*dA/dr. Avoid pointwise A/r on the axis.
    Baxis=fem.Function(fem.functionspace(mesh,('DG',1)))
    Baxis.interpolate(fem.Expression(2*solution.dx(0),Baxis.function_space.element.interpolation_points))
    h=c['chamber']; z=np.linspace(h['z_start'],h['z_start']+h['length'],121)
    points=np.column_stack([np.zeros_like(z),z,np.zeros_like(z)])
    tree=geometry.bb_tree(mesh,2)
    candidates=geometry.compute_collisions_points(tree,points)
    hits=geometry.compute_colliding_cells(mesh,candidates,points)
    if any(len(hits.links(i))==0 for i in range(len(z))):
        raise RuntimeError('On-axis sample outside mesh')
    owners=np.array([hits.links(i)[0] for i in range(len(z))],dtype=np.int32)
    sampled=Baxis.eval(points,owners).real.ravel()
    reference=pack_bz(z,c['coils'])
    error=float(np.linalg.norm(sampled-reference)/np.linalg.norm(reference)) if np.linalg.norm(reference)>0 else None
    np.savetxt(out/'fem_axis.csv',np.column_stack([z,sampled,reference]),delimiter=',',
               header='z_m,Bz_FEM_T,Bz_free_space_pack_T',comments='')
    # DG0 cell-center field output avoids evaluating 1/r at axis vertices.
    W=fem.functionspace(mesh,('DG',0,(2,)))
    B=fem.Function(W); B.name='B_r_B_z'
    B.interpolate(fem.Expression(ufl.as_vector([Br,Bz]),W.element.interpolation_points))
    with io.XDMFFile(comm,out/'field.xdmf','w') as f:
        f.write_mesh(mesh); f.write_function(B)
    # P2 potential cannot be written directly on P1 geometry through XDMF.
    P1=fem.Function(fem.functionspace(mesh,('Lagrange',1))); P1.name='A_theta_visualization'
    P1.interpolate(solution)
    with io.XDMFFile(comm,out/'potential.xdmf','w') as f:
        f.write_mesh(mesh); f.write_function(P1)
    metrics={'chamber_volume_m3':volume,'chamber_mean_Bz_T':mean,
             'chamber_Bz_std_T':std,'chamber_Br_rms_T':radial_rms,
             'axis_relative_l2_error_vs_free_space_pack':error,
             'note':'Error includes finite outer-boundary truncation and discretization. P2 A_theta; mu=mu0 everywhere.'}
    (out/'fem_summary.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
