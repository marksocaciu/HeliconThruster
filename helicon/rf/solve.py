"""Complex Nedelec RF solve. Run on a compatible complex DOLFINx/PETSc build.

python -m helicon.rf.solve --build runs/rf-prepared/mesh --out runs/rf-solved
MPI is supported by distributed assembly, reductions and rank-local output.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from ..records import write_json, read_json, digest, utc_now
from ..mesh import gmsh_module
from .config import load
from .materials import cold_epsilon, AxisymmetricField, MU0, EPS_0
from .mesh import CELL, FACE


def runtime():
    from mpi4py import MPI
    from petsc4py import PETSc
    import dolfinx
    if not np.issubdtype(PETSc.ScalarType,np.complexfloating) or not np.issubdtype(dolfinx.default_scalar_type,np.complexfloating):
        raise RuntimeError('RF requires complex PETSc AND complex DOLFINx. Use a separate complex environment; the existing DC solver remains unchanged.')
    return MPI,PETSc,dolfinx


def solve(build,out,b_map=None):
    MPI,PETSc,dolfinx=runtime()
    from dolfinx import fem,mesh as dmesh,io
    from dolfinx.fem.petsc import LinearProblem, assemble_matrix
    import basix.ufl
    import ufl
    from .forms import bilinear,pml_tensors,interface_flux
    from .run import snapshot
    comm=MPI.COMM_WORLD; build=Path(build); out=Path(out)
    if b_map is None and (build.parent/'background_B.npz').exists():
        b_map=build.parent/'background_B.npz'
    # Fail before collective mesh import if any rank lacks an input.
    error=None
    try:
        c=load(build/'rf.json'); report=read_json(build/'mesh_report.json')
        if digest(build/'rf.msh') != report['mesh_sha256'] or digest(build/'rf.json') != report['config_sha256']:
            raise ValueError('Prepared RF mesh/config changed after generation')
        if b_map and c['material_case']!='plasma': raise ValueError('A B map requires the plasma material case')
        field=AxisymmetricField(b_map) if b_map else None
    except Exception as exc: error=str(exc)
    errors=comm.allgather(error)
    if any(errors): raise RuntimeError(str(errors))
    error=None
    if comm.rank==0:
        try:
            out.mkdir(parents=True,exist_ok=False)
            shutil.copy2(build/'rf.json',out/'rf.json')
            shutil.copy2(build/'mesh_report.json',out/'mesh_report.json')
            shutil.copy2(build/'rf.msh',out/'rf.msh')
            if b_map: shutil.copy2(b_map,out/'background_B.npz')
            if (build.parent/'input_plasma_summary.json').exists():
                shutil.copy2(build.parent/'input_plasma_summary.json',out/'input_plasma_summary.json')
            record={'status':'running','started_utc':utc_now(),'source':snapshot(out),
                    'mesh_sha256':digest(out/'rf.msh'),'config_sha256':digest(out/'rf.json'),
                    'magnetic_map_sha256':digest(out/'background_B.npz') if b_map else None}
            write_json(out/'run.json',record)
        except Exception as exc: error=str(exc)
    error=comm.bcast(error,root=0)
    if error: raise RuntimeError(error)
    try:
        try: gmsh_module()
        except SystemExit as exc: raise RuntimeError(str(exc)) from exc
        from dolfinx.io import gmsh as gmshio
        data=gmshio.read_from_msh(out/'rf.msh',comm,rank=0,gdim=3,partitioner=dmesh.create_cell_partitioner(dmesh.GhostMode.shared_facet))
        domain,ct,ft=data.mesh,data.cell_tags,data.facet_tags
        if ct is None or ft is None: raise ValueError('RF mesh must contain cell and face tags')
        tdim=domain.topology.dim; domain.topology.create_connectivity(tdim-1,tdim)
        for name,tag in CELL.items():
            if comm.allreduce(len(ct.find(tag)),op=MPI.SUM)==0: raise ValueError('Missing '+name)
        for name,tag in FACE.items():
            if comm.allreduce(len(ft.find(tag)),op=MPI.SUM)==0: raise ValueError('Missing '+name)
        degree=int(c['solver']['degree'])
        V=fem.functionspace(domain,basix.ufl.element('N1curl',domain.basix_cell(),degree))
        T=fem.functionspace(domain,('DG',0,(3,3)))
        eps=fem.Function(T); eps.name='epsilon_r'
        tensors=eps.x.array.reshape(-1,3,3); tensors[:]=np.eye(3)
        owned=domain.topology.index_map(tdim).size_local
        nc=owned+domain.topology.index_map(tdim).num_ghosts
        centers=dmesh.compute_midpoints(domain,tdim,np.arange(nc,dtype=np.int32))
        def assign(ids,values):
            blocks=np.array([T.dofmap.cell_dofs(int(i))[0] for i in ids],dtype=int)
            tensors[blocks]=values
        case=c['material_case']; dc=c['dielectric']; p=c['plasma']; w=2*np.pi*c['frequency_Hz']; k02=w*w*MU0*EPS_0
        if case!='vacuum':
            assign(ct.find(CELL['WALL']),np.eye(3)*dc['wall_epsilon_r']*(1+1j*dc['wall_loss_tangent']))
        plasma_ids=ct.find(CELL['PLASMA'])
        material_error=None
        try:
            if case=='dielectric': assign(plasma_ids,np.eye(3)*dc['test_epsilon_r']*(1+1j*dc['test_loss_tangent']))
            if case=='plasma':
                b=field(centers[plasma_ids]) if field else np.broadcast_to(p['uniform_B_T'],(len(plasma_ids),3))
                assign(plasma_ids,cold_epsilon(c['frequency_Hz'],p['electron_density_m3'],p['collision_frequency_Hz'],b))
        except Exception as exc: material_error=str(exc)
        errors=comm.allgather(material_error)
        if any(errors): raise ValueError(str(errors))
        eps.x.scatter_forward()
        d=c['domain']
        epml,invmu=pml_tensors(ufl.SpatialCoordinate(domain),c)
        dx=ufl.Measure('dx',domain=domain,subdomain_data=ct,metadata={'quadrature_degree':max(6,2*degree+2)})
        ds=ufl.Measure('dS',domain=domain,subdomain_data=ft)
        u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
        form=bilinear(domain,u,v,eps,dx,c)
        rhs=ufl.inner(fem.Constant(domain,np.zeros(3,dtype=PETSc.ScalarType)),v)*dx
        fixed=fem.locate_dofs_topological(V,2,np.concatenate((ft.find(FACE['PEC_ANTENNA']),ft.find(FACE['OUTER']))))
        port=fem.locate_dofs_topological(V,2,ft.find(FACE['FEED_SHEET']))
        unit=fem.Function(V)
        unit.interpolate(lambda points: np.vstack((np.zeros(points.shape[1]),np.zeros(points.shape[1]),np.full(points.shape[1],1/c['antenna']['gap_m']))).astype(PETSc.ScalarType))
        common=np.intersect1d(fixed,port)
        if comm.allreduce(float(np.max(np.abs(unit.x.array[common]))) if len(common) else 0.,op=MPI.MAX)>1e-9:
            raise RuntimeError('Feed trace conflicts with PEC trace')
        port=np.setdiff1d(port,fixed).astype(np.int32)
        lift=fem.Function(V); lift.x.array[port]=unit.x.array[port]; lift.x.scatter_forward()
        drive=fem.Function(V); drive.x.array[:]=c['voltage_peak_V']*lift.x.array
        zero=fem.Function(V)
        bcs=[fem.dirichletbc(zero,fixed),fem.dirichletbc(drive,port)]
        options={'ksp_type':'preonly','pc_type':'lu','ksp_error_if_not_converged':True}
        if PETSc.Sys.hasExternalPackage('mumps'): options['pc_factor_mat_solver_type']='mumps'
        elif PETSc.Sys.hasExternalPackage('superlu_dist'): options['pc_factor_mat_solver_type']='superlu_dist'
        elif comm.size>1: raise RuntimeError('MPI RF solve needs a parallel LU backend (MUMPS or SuperLU_DIST)')
        problem=LinearProblem(form,rhs,bcs=bcs,petsc_options_prefix='helicon_rf_',petsc_options=options)
        solution=problem.solve(); solution.name='E_peak'; solution.x.scatter_forward()
        if problem.solver.getConvergedReason()<=0: raise RuntimeError('RF linear solve failed')
        # Recover source current from the unconstrained weak-form reaction.
        matrix=assemble_matrix(fem.form(form)); matrix.assemble()
        reaction=matrix.createVecLeft(); matrix.mult(solution.x.petsc_vec,reaction)
        nowned=V.dofmap.index_map.size_local*V.dofmap.index_map_bs
        rlocal=reaction.array
        reaction_scalar=comm.allreduce(np.vdot(lift.x.array[:nowned],rlocal),op=MPI.SUM)
        current=1j*reaction_scalar/(w*MU0)
        voltage=c['voltage_peak_V']; pin=.5*float(np.real(voltage*np.conj(current)))
        impedance=voltage/current if abs(current)>0 else None
        free=np.ones(nowned,dtype=bool)
        constrained=np.union1d(fixed,port); free[constrained[constrained<nowned]]=False
        free_norm=np.sqrt(comm.allreduce(float(np.vdot(rlocal[free],rlocal[free]).real),op=MPI.SUM))
        full_norm=reaction.norm()
        relative_residual=free_norm/max(full_norm,1e-300)
        def integral(expr):
            return comm.allreduce(fem.assemble_scalar(fem.form(expr)),op=MPI.SUM)
        powers={}
        for name,tag in CELL.items():
            if tag==CELL['PML']: continue
            powers[name]=float(np.real(integral(.5*w*EPS_0*ufl.imag(ufl.inner(eps*solution,solution))*dx(tag))))
        pml_form=(ufl.inner(invmu*ufl.curl(solution),ufl.curl(solution))-k02*ufl.inner(epml*solution,solution))*dx(CELL['PML'])
        ppml=-float(np.imag(integral(pml_form)))/(2*w*MU0)
        flux=interface_flux(domain,solution,c['frequency_Hz'],(0,0,(d['z_min_m']+d['z_max_m'])/2))
        outgoing=float(np.real(integral(flux*ds(FACE['PML_INTERFACE']))))
        total=sum(powers.values())+ppml
        balance=abs(pin-total)/max(abs(pin),abs(total),1e-30)
        passed=pin>0 and balance<c['solver']['power_balance_tolerance'] and relative_residual<c['solver']['relative_residual_tolerance']
        if any(val < -max(abs(pin),1e-30)*1e-8 for val in powers.values()): passed=False
        # Portable per-rank cell data: DG interpolation avoids treating edge DOFs as point values.
        def sample(expr,shape):
            space=fem.functionspace(domain,('DG',0,shape)) if shape else fem.functionspace(domain,('DG',0))
            f=fem.Function(space); f.interpolate(fem.Expression(expr,space.element.interpolation_points))
            count=int(np.prod(shape)) if shape else 1
            indices=np.array([space.dofmap.cell_dofs(i)[0] for i in range(owned)])
            return f.x.array.reshape(-1,count)[indices]
        electric=sample(solution,(3,)); magnetic=sample(ufl.curl(solution)/(1j*w),(3,))
        heating=sample(.5*w*EPS_0*ufl.imag(ufl.inner(eps*solution,solution)),()).real.ravel()
        tags=np.zeros(nc,dtype=np.int32); tags[ct.indices]=ct.values
        heating[tags[:owned]==CELL['PML']]=0 # PML attenuation is not plasma heating.
        np.savez_compressed(out/f'cells_rank{comm.rank:04d}.npz',xyz_m=centers[:owned],region=tags[:owned],E_peak_V_m=electric,B_peak_T=magnetic,q_abs_W_m3=heating)
        # XDMF DG0 fields are cell samples, not the high-order solution or restart data.
        vector_space=fem.functionspace(domain,('DG',0,(3,)))
        er=fem.Function(vector_space); ei=fem.Function(vector_space)
        er.name='E_real_cell_sample'; ei.name='E_imag_cell_sample'
        er.interpolate(fem.Expression(ufl.real(solution),vector_space.element.interpolation_points))
        ei.interpolate(fem.Expression(ufl.imag(solution),vector_space.element.interpolation_points))
        with io.XDMFFile(comm,str(out/'fields.xdmf'),'w') as writer:
            writer.write_mesh(domain); writer.write_function(er); writer.write_function(ei)
        result={'status':'completed' if passed else 'checks_failed','frequency_Hz':c['frequency_Hz'],'case':case,
                'voltage_peak_V':voltage,'port_current_peak_A':[float(current.real),float(current.imag)],
                'ideal_gap_impedance_ohm':None if impedance is None else [float(impedance.real),float(impedance.imag)],
                'input_power_W':pin,'material_absorption_W':powers,'pml_absorption_W':ppml,'outward_interface_flux_W':outgoing,
                'power_balance_relative_error':balance,'free_equation_relative_residual':relative_residual,
                'interface_flux_vs_pml_relative_difference':abs(outgoing-ppml)/max(abs(pin),1e-30),
                'numerical_checks_passed':passed,'dofs':V.dofmap.index_map.size_global*V.dofmap.index_map_bs,
                'mpi_ranks':comm.size,'dolfinx':dolfinx.__version__,'petsc':str(PETSc.Sys.getVersion()),
                'scalar_type':str(PETSc.ScalarType),'linear_solver_options':options,
                'limitations':['Ideal gap-sheet source and PEC antenna; impedance is source-model dependent, not calibrated terminal impedance.',
                    'No coax, matching network, RF coil shielding, reflector, sheath, or plume plasma.',
                    'Fixed uniform plasma state; effective collision rate is a user assumption.',
                    'Passing algebraic/power checks does not establish mesh, PML or physical convergence.']}
        matrix.destroy(); reaction.destroy()
        comm.barrier()
        if comm.rank==0:
            write_json(out/'rf_summary.json',result)
            record.update(status=result['status'],finished_utc=utc_now())
            record['artifacts_sha256']={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='run.json'}
            write_json(out/'run.json',record)
        return result
    except Exception as exc:
        if comm.rank==0:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_utc=utc_now()); write_json(out/'run.json',record)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build');p.add_argument('--out');p.add_argument('--b-map');p.add_argument('--check-runtime',action='store_true')
    a=p.parse_args()
    if a.check_runtime:
        MPI,PETSc,dolfinx=runtime()
        if MPI.COMM_WORLD.rank==0: print(f'DOLFINx {dolfinx.__version__}, PETSc {PETSc.ScalarType}, MPI ranks={MPI.COMM_WORLD.size}')
        return 0
    if not a.build or not a.out: p.error('--build and --out are required for a solve')
    result=solve(a.build,a.out,a.b_map)
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank==0: print(json.dumps(result,indent=2))
    return 0 if result['numerical_checks_passed'] else 2

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception:
        # Prevent a rank-local exception leaving other ranks in a collective.
        import traceback
        traceback.print_exc()
        if 'mpi4py.MPI' in sys.modules:
            from mpi4py import MPI
            if MPI.COMM_WORLD.size>1: MPI.COMM_WORLD.Abort(1)
        sys.exit(1)
