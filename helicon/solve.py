"""Serial FEniCSx (0.10) P2 axisymmetric A_theta starter solver."""
import argparse
import json
from pathlib import Path
import numpy as np
from .mesh import gmsh_module
from .config import load
from .reference import MU0, pack_bz
from .diagnostics import design_metrics, positive, reconstruct_field, disk_mean
from .records import digest, write_json
import importlib.metadata
import platform
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',default='build',help='mesh directory; uses its saved configuration')
    p.add_argument('--target-gauss',type=float,default=140)
    a=p.parse_args(); out=Path(a.build); c=load(out/'geometry.json')
    positive(a.target_gauss, 'target gauss')
    gmsh_module()  # Also makes Homebrew's Python API importable by DOLFINx.
    from mpi4py import MPI
    from petsc4py import PETSc
    import ufl
    import dolfinx
    from dolfinx import fem, geometry, io
    from dolfinx.io import gmsh
    from dolfinx.fem.petsc import LinearProblem
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
    # P2 derivatives are exactly representable in DG1 on these affine triangles.
    G=fem.Function(fem.functionspace(mesh,('DG',1,(2,))))
    G.interpolate(fem.Expression(ufl.grad(solution),G.function_space.element.interpolation_points))
    tree=geometry.bb_tree(mesh,2,padding=1e-12)
    def sample(radii, axial):
        radii=np.asarray(radii); axial=np.asarray(axial)
        points=np.column_stack([radii,axial,np.zeros_like(radii)])
        candidates=geometry.compute_collisions_points(tree,points)
        hits=geometry.compute_colliding_cells(mesh,candidates,points)
        if any(len(hits.links(i))==0 for i in range(len(points))):
            raise RuntimeError('Field sample outside mesh')
        # Derivatives at interfaces use one colliding cell; converge the mesh.
        owners=np.array([hits.links(i)[0] for i in range(len(points))],dtype=np.int32)
        av=solution.eval(points,owners).real.ravel()
        grad=G.eval(points,owners).real.reshape(-1,2)
        br,bz=reconstruct_field(radii,av,grad)
        return av,br,bz
    h=c['chamber']; end=h['z_start']+h['length']
    z=np.linspace(h['z_start'],end,121)
    _,_,sampled=sample(np.zeros_like(z),z)
    reference=pack_bz(z,c['coils'])
    error=float(np.linalg.norm(sampled-reference)/np.linalg.norm(reference)) if np.linalg.norm(reference)>0 else None
    np.savetxt(out/'fem_axis.csv',np.column_stack([z,sampled,reference]),delimiter=',',
               header='z_m,Bz_FEM_T,Bz_free_space_pack_T',comments='')
    radii=np.linspace(0,h['radius'],101)
    exit_a,exit_br,exit_bz=sample(radii,np.full_like(radii,end))
    np.savetxt(out/'fem_exit.csv',np.column_stack([radii,exit_br,exit_bz,np.hypot(exit_br,exit_bz)]),
               delimiter=',',header='r_m,Br_T,Bz_T,B_magnitude_T',comments='')
    # Area mean = integral(B * 2*pi*r dr)/(pi*R^2), with interior radial quadrature.
    nodes,weights=np.polynomial.legendre.leggauss(64)
    qr=(nodes+1)*h['radius']/2
    _,qbr,qbz=sample(qr,np.full_like(qr,end))
    # Stokes: flux through disk = 2*pi*R*A_theta(R,z). Exact for continuous A.
    exit_mean=float(2*exit_a[-1]/h['radius'])
    exit_magnitude_mean=disk_mean(np.hypot(qbr,qbz),qr,weights,h['radius'])
    d=c['domain']
    focus_r=min(d['radius']*.98,max([h['radius']+h['wall_thickness'],c['downstream']['radius']]+
                  [k['radius']+k['radial_thickness']/2 for k in c['coils']])*1.2)
    margin=(d['z_max']-d['z_min'])*1e-8
    focus_min=max(d['z_min']+margin,(min([h['z_start']]+[k['z_center']-k['axial_width']/2 for k in c['coils']])-.03))
    focus_max=min(d['z_max']-margin,max([end+c['downstream']['length']]+
                   [k['z_center']+k['axial_width']/2 for k in c['coils']])+.03)
    # Grids are for visualization only, never for convergence metrics or objectives.
    for name,bounds in [('full',(d['radius'],d['z_min'],d['z_max'])),
                        ('device',(focus_r,focus_min,focus_max))]:
        rr=np.linspace(0,bounds[0],151); zz=np.linspace(bounds[1],bounds[2],241)
        R,Z=np.meshgrid(rr,zz)
        av,br,bz=sample(R.ravel(),Z.ravel())
        np.savez_compressed(out/f'field_{name}.npz',r_m=rr,z_m=zz,
                            A_theta=av.reshape(R.shape),Br_T=br.reshape(R.shape),
                            Bz_T=bz.reshape(R.shape),psi_Wb_per_rad=R*av.reshape(R.shape))
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
    metrics.update(design_metrics(mean,std,radial_rms,a.target_gauss/1e4))
    metrics.update(axis_exit_Bz_T=float(sampled[-1]),axis_exit_Bz_G=float(sampled[-1]*1e4),
                   exit_area_mean_Bz_T=exit_mean,exit_area_mean_Bz_G=exit_mean*1e4,
                   exit_area_mean_magnitude_T=exit_magnitude_mean,
                   exit_area_mean_Bz_method="2*A_theta(R,z_exit)/R from Stokes theorem",
                   exit_magnitude_quadrature_points=64,
                   axis_reference_error_percent=None if error is None else 100*error,
                   mesh_sha256=digest(out/'helicon.msh'),
                   configuration_sha256=digest(out/'geometry.json'),
                   field_sampling='P2 potential and exact DG1 derivatives; regular axis limit; one cell at interfaces',
                   objective_definition='sqrt(volume_mean((Bz-target)^2 + Br^2))/abs(target)')
    versions={}
    for package in ('numpy','fenics-dolfinx','fenics-ufl','fenics-basix','petsc4py','mpi4py','gmsh'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='not available as package metadata'
    versions.update(dolfinx=dolfinx.__version__,petsc=str(PETSc.Sys.getVersion()),
                    mpi=MPI.Get_library_version(),python=sys.version,platform=platform.platform())
    write_json(out/'solver_environment.json',versions)
    write_json(out/'fem_summary.json',metrics)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
