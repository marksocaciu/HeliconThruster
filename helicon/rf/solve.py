"""Complex Nedelec RF solve. Run on a compatible complex DOLFINx/PETSc build.

python -m helicon.rf.solve --build runs/rf-prepared/mesh --out runs/rf-solved
MPI is supported by distributed assembly, reductions and rank-local output.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
from ..records import write_json, read_json, digest, utc_now
from ..mesh import gmsh_module
from .config import load
from .materials import cold_epsilon, AxisymmetricField, MU0, EPS_0
from .mesh import CELL, FACE


def _power_sanity_checks(pin, powers, ppml, outgoing, balance, residual,
                         power_tolerance, residual_tolerance):
    """Check power signs and independent physical-domain/PML power transfer."""
    scale = max(abs(pin), 1e-30) if np.isfinite(pin) else 1e-30
    finite = (np.isfinite(pin) and np.isfinite(ppml) and np.isfinite(outgoing)
              and np.isfinite(balance) and np.isfinite(residual)
              and all(np.isfinite(value) for value in powers.values()))
    flux_difference = abs(outgoing-ppml)/scale if finite else float('inf')
    sign_tolerance = scale*1e-4
    return {
        'finite_power_values': bool(finite),
        'positive_input_power': bool(finite and pin > 0),
        'power_balance': bool(finite and balance < power_tolerance),
        'free_equation_residual': bool(finite and residual < residual_tolerance),
        'nonnegative_material_absorption': bool(finite and all(value >= -sign_tolerance for value in powers.values())),
        'nonnegative_pml_absorption': bool(finite and ppml >= -sign_tolerance),
        'nonnegative_outward_interface_flux': bool(finite and outgoing >= -sign_tolerance),
        'pml_interface_flux_agreement': bool(finite and flux_difference < power_tolerance),
    }


def _linear_solver_options(PETSc, ranks, requested=None, memory_mb=None, scratch_dir=None):
    """Explicit choice wins over the environment; never silently replace it."""
    backend = requested if requested is not None else os.environ.get('HELICON_RF_LU_SOLVER', 'auto')
    if backend not in ('auto', 'mumps', 'mumps_ooc', 'superlu_dist', 'gmres_asm'):
        raise ValueError('RF solver must be auto, mumps, mumps_ooc, superlu_dist or gmres_asm')
    ooc = backend == 'mumps_ooc'
    if not ooc and (memory_mb is not None or scratch_dir is not None):
        raise ValueError('MUMPS workspace/scratch arguments require --linear-solver mumps_ooc')
    if backend == 'gmres_asm':
        def integer(name, default, minimum):
            value = int(os.environ.get(name, default))
            if value < minimum: raise ValueError(f'{name} must be >= {minimum}')
            return value
        return {'ksp_type':'gmres', 'ksp_gmres_restart':integer('HELICON_RF_GMRES_RESTART',30,1),
                'ksp_rtol':1e-9, 'ksp_atol':0.0,
                'ksp_max_it':integer('HELICON_RF_KSP_MAX_IT',1000,1),
                # Return the last iterate for diagnostics; solve() still rejects failure.
                'ksp_error_if_not_converged':False,
                'pc_type':'asm', 'pc_asm_overlap':integer('HELICON_RF_ASM_OVERLAP',1,0),
                'sub_ksp_type':'preonly', 'sub_pc_type':'ilu',
                'sub_pc_factor_levels':integer('HELICON_RF_ILU_LEVELS',0,0)}
    if ooc: backend = 'mumps'
    if backend == 'auto':
        backend = next((name for name in ('mumps', 'superlu_dist')
                        if PETSc.Sys.hasExternalPackage(name)), None)
        if backend is None and ranks > 1:
            raise RuntimeError('MPI RF solve needs MUMPS or SuperLU_DIST')
    elif not PETSc.Sys.hasExternalPackage(backend):
        raise RuntimeError(f'Requested RF LU solver {backend} is not available in this PETSc build')
    options = {'ksp_type':'preonly', 'pc_type':'lu', 'ksp_error_if_not_converged':True}
    if backend is not None:
        options['pc_factor_mat_solver_type'] = backend
    if backend == 'superlu_dist':
        # Avoid the METIS default: the user's complex PETSc lacks METIS/ParMETis.
        options.update(mat_superlu_dist_colperm='MMD_AT_PLUS_A',
                       mat_superlu_dist_printstat=True)
    if backend == 'mumps':
        ordering = os.environ.get('HELICON_RF_MUMPS_ORDERING')
        if ordering:
            if int(ordering) not in (0, 2, 3, 4, 5, 6, 7):
                raise ValueError('HELICON_RF_MUMPS_ORDERING must be a MUMPS ICNTL(7) value')
            options['mat_mumps_icntl_7'] = int(ordering)
        blr_tol = os.environ.get('HELICON_RF_MUMPS_BLR_TOL')
        if blr_tol:
            tol = float(blr_tol)
            if not 0 < tol < 1:
                raise ValueError('HELICON_RF_MUMPS_BLR_TOL must be in (0, 1)')
            options.update(mat_mumps_icntl_35=1, mat_mumps_cntl_7=tol)
    if ooc:
        raw = memory_mb if memory_mb is not None else os.environ.get('HELICON_RF_MUMPS_MEMORY_MB', '3000')
        try:
            budget = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError('MUMPS memory budget must be a positive integer in MB per rank') from exc
        if isinstance(raw, bool) or str(budget) != str(raw) or not 0 < budget <= 2147483647:
            raise ValueError('MUMPS memory budget must be a positive integer in MB per rank')
        directory = scratch_dir if scratch_dir is not None else os.environ.get('HELICON_RF_MUMPS_SCRATCH')
        if not directory:
            raise ValueError('mumps_ooc requires --mumps-scratch-dir or HELICON_RF_MUMPS_SCRATCH')
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f'MUMPS scratch directory must already exist: {directory}')
        if len(os.fsencode(directory)) > 240:
            raise ValueError('MUMPS scratch directory path must be at most 240 bytes')
        # Check actual creation permission on each rank, not merely os.access.
        with tempfile.TemporaryFile(dir=directory) as probe:
            probe.write(b'RF scratch check'); probe.flush()
        options.update(mat_mumps_icntl_22=1, mat_mumps_icntl_23=budget,
                       mat_mumps_ooc_tmpdir=str(directory), mat_mumps_icntl_4=2)
    return options



def _reaction_vector(form, solution, PETSc):
    """Unconstrained weak-form reaction; sum shared contributions to owners."""
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_vector
    result = assemble_vector(fem.form(ufl.action(form, solution)))
    result.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    return result


def _solve_with_options(problem, options, PETSc):
    """Keep options alive until nested PCs and external factors are set up.

    LinearProblem removes its options after KSP.setFromOptions, before lazy
    factorization and ASM sub-KSP setup. Preserve caller database entries.
    """
    database = PETSc.Options(problem.solver.getOptionsPrefix())
    previous = {key: database.getString(key) if database.hasName(key) else None for key in options}
    try:
        for key, value in options.items(): database[key] = value
        problem.solver.setFromOptions()
        return problem.solve()
    finally:
        for key in options:
            if previous[key] is None: del database[key]
            else: database[key] = previous[key]

def _owned_cell_integrals(vector, space, owned):
    """Read DG0 volume integrals from DOLFINx 0.10 arrays or 0.11 Vectors.

    These are cell-volume forms with a discontinuous basis: no shared-DOF
    accumulation is needed. Export owned cells only, excluding ghost copies.
    """
    values = vector.array if hasattr(vector, 'array') else vector
    values = np.asarray(values)
    indices = np.array([space.dofmap.cell_dofs(i)[0] for i in range(owned)],
                       dtype=np.int32)
    return np.real(values[indices]).copy()


def _select_petsc_dir():
    """Select the complex PETSc installation for RF before importing petsc4py."""
    configured = os.environ.get('HELICON_RF_PETSC_DIR')
    if configured:
        if not Path(configured).is_dir():
            raise RuntimeError(f'HELICON_RF_PETSC_DIR is not a directory: {configured}')
        os.environ['PETSC_DIR'] = configured
    elif not os.environ.get('PETSC_DIR'):
        # Debian/Ubuntu provide parallel real and complex PETSc installations.
        # Keep this RF-only default out of the magnetostatic runtime.
        debian_complex = Path('/usr/lib/petscdir/petsc-complex')
        if debian_complex.is_dir():
            os.environ['PETSC_DIR'] = str(debian_complex)


def runtime():
    _select_petsc_dir()
    from mpi4py import MPI
    from petsc4py import PETSc
    import dolfinx
    if not np.issubdtype(PETSc.ScalarType,np.complexfloating) or not np.issubdtype(dolfinx.default_scalar_type,np.complexfloating):
        raise RuntimeError('RF requires complex PETSc AND complex DOLFINx. Use a separate complex environment; the existing DC solver remains unchanged.')
    return MPI,PETSc,dolfinx


def solve(build,out,b_map=None,plasma_map=None,linear_solver=None,mumps_memory_mb=None,mumps_scratch_dir=None,
          analysis_only=False, job_memory_mb=None, memory_reserve_mb=2000):
    MPI,PETSc,dolfinx=runtime()
    options=_linear_solver_options(PETSc,MPI.COMM_WORLD.size,linear_solver,
                                   mumps_memory_mb,mumps_scratch_dir)
    from dolfinx import fem,mesh as dmesh,io
    from dolfinx.fem.petsc import LinearProblem, assemble_matrix
    from .memory import memory_snapshot, admission
    if job_memory_mb is not None and (not np.isfinite(job_memory_mb) or job_memory_mb <= 0):
        raise ValueError('job_memory_mb must be positive and finite')
    if not np.isfinite(memory_reserve_mb) or memory_reserve_mb < 0:
        raise ValueError('memory_reserve_mb must be nonnegative and finite')
    if analysis_only and options.get('pc_factor_mat_solver_type') != 'mumps':
        raise ValueError('--analysis-only requires MUMPS')
    import basix.ufl
    import ufl
    from .forms import bilinear,pml_tensors,interface_flux,interface_flux_traces
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
        from .spatial_maps import PlasmaMap
        profile=PlasmaMap(plasma_map) if plasma_map else None
        if profile and c['material_case'] != 'plasma':
            raise ValueError('Spatial plasma map requires material_case=plasma')
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
            if plasma_map: shutil.copy2(plasma_map,out/'input_plasma_fields.npz')
            if (build.parent/'input_plasma_summary.json').exists():
                shutil.copy2(build.parent/'input_plasma_summary.json',out/'input_plasma_summary.json')
            record={'status':'running','started_utc':utc_now(),'source':snapshot(out),
                    'mesh_sha256':digest(out/'rf.msh'),'config_sha256':digest(out/'rf.json'),
                    'magnetic_map_sha256':digest(out/'background_B.npz') if b_map else None,
                    'plasma_map_sha256':digest(out/'input_plasma_fields.npz') if plasma_map else None}
            write_json(out/'run.json',record)
        except Exception as exc: error=str(exc)
    error=comm.bcast(error,root=0)
    if error: raise RuntimeError(error)
    def checkpoint(stage):
        measurement = memory_snapshot(comm, stage)
        if comm.rank == 0:
            record.setdefault('memory', []).append(measurement)
            record['stage'] = stage
            write_json(out/'run.json', record)
        return measurement
    try:
        checkpoint('mesh_import')
        try: gmsh_module()
        except SystemExit as exc: raise RuntimeError(str(exc)) from exc
        from dolfinx.io import gmsh as gmshio
        # Each facet of this conforming tetrahedral mesh meets at most two cells.
        # DOLFINx 0.11 requires this bound explicitly when creating a partitioner.
        partitioner=dmesh.create_cell_partitioner(
            dmesh.GhostMode.shared_facet, max_facet_to_cell_links=2)
        data=gmshio.read_from_msh(out/'rf.msh',comm,rank=0,gdim=3,partitioner=partitioner)
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
                state=profile(centers[plasma_ids]) if profile else None
                density=state['electron_density_m3'] if state else p['electron_density_m3']
                collision=state['collision_frequency_s'] if state else p['collision_frequency_Hz']
                assign(plasma_ids,cold_epsilon(c['frequency_Hz'],density,collision,b))
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
        if comm.rank==0:
            record.update(stage='linear_solve', linear_solver_options=options,
                          dofs=V.dofmap.index_map.size_global*V.dofmap.index_map_bs)
            write_json(out/'run.json',record)
            print('RF linear solver: '+json.dumps(options)+f'; DOFs={record["dofs"]}', flush=True)
        problem=LinearProblem(form,rhs,bcs=bcs,petsc_options_prefix='helicon_rf_',petsc_options=None)
        analysis = None
        if options.get('pc_factor_mat_solver_type') == 'mumps' and (analysis_only or job_memory_mb is not None):
            # Assemble into LinearProblem's existing matrix; no second global matrix.
            problem.A.zeroEntries()
            assemble_matrix(problem.A, problem.a, bcs=bcs)
            problem.A.assemble()
            baseline = checkpoint('assembled_before_analysis')
            from .analysis import analyze_mumps
            analysis = analyze_mumps(problem.A, options, comm, out, PETSc)
            analysis['admission'] = admission(analysis, baseline['sum_rank_rss_bytes'],
                job_memory_mb, memory_reserve_mb, options.get('mat_mumps_icntl_23'))
            if comm.rank == 0:
                write_json(out/'solver_analysis.json', analysis)
                print('RF symbolic analysis: '+json.dumps(analysis), flush=True)
            checkpoint('analysis_complete')
            if analysis_only:
                del problem
                result = {'status':'analysis_completed', 'analysis':analysis,
                          'dofs':V.dofmap.index_map.size_global*V.dofmap.index_map_bs}
                if comm.rank == 0:
                    record.update(result, finished_utc=utc_now())
                    write_json(out/'run.json',record)
                return result
            if not analysis['admission']['accepted']:
                del problem
                result = {'status':'memory_rejected', 'numerical_checks_passed':False,
                          'analysis':analysis, 'error':'; '.join(analysis['admission']['reasons'])}
                if comm.rank == 0:
                    record.update(result,finished_utc=utc_now())
                    write_json(out/'run.json',record)
                comm.barrier()
                return result
        checkpoint('linear_solve')
        from .diagnostics import IterationMonitor, linear_diagnostics
        monitor = IterationMonitor(comm,out) if options['ksp_type'] != 'preonly' else None
        if monitor is not None: problem.solver.setMonitor(monitor)
        try:
            solved = _solve_with_options(problem, options, PETSc)
        finally:
            if monitor is not None:
                monitor.close()
                problem.solver.cancelMonitor()
        solver_diagnostics = linear_diagnostics(problem,PETSc,c['solver']['relative_residual_tolerance'])
        if monitor is not None:
            solver_diagnostics['history_write_error'] = comm.bcast(monitor.error,root=0)
        if comm.rank == 0:
            record['solver_diagnostics'] = solver_diagnostics
            write_json(out/'solver_diagnostics.json',solver_diagnostics)
            write_json(out/'run.json',record)
            print('RF solver result: '+json.dumps(solver_diagnostics),flush=True)
        if not solver_diagnostics['accepted']:
            del solved, problem
            checkpoint('linear_solver_failed')
            result = {'status':'linear_solver_failed','numerical_checks_passed':False,
                      'solver_diagnostics':solver_diagnostics}
            if comm.rank == 0:
                record.update(result,finished_utc=utc_now())
                write_json(out/'run.json',record)
            comm.barrier()
            return result
        # Own the field independently before LinearProblem releases PETSc objects.
        solution=fem.Function(V); solution.name='E_peak'
        solution.x.array[:]=solved.x.array; solution.x.scatter_forward()
        del solved, problem
        import gc
        gc.collect()
        checkpoint('solver_released')
        # Assemble A_unconstrained E as a vector; retain no duplicate sparse matrix.
        reaction=_reaction_vector(form, solution, PETSc)
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
        reaction.destroy()
        del rlocal
        checkpoint('postprocessing')
        def integral(expr):
            return comm.allreduce(fem.assemble_scalar(fem.form(expr)),op=MPI.SUM)
        powers={}
        for name,tag in CELL.items():
            if tag==CELL['PML']: continue
            powers[name]=float(np.real(integral(.5*w*EPS_0*ufl.imag(ufl.inner(eps*solution,solution))*dx(tag))))
        pml_form=(ufl.inner(invmu*ufl.curl(solution),ufl.curl(solution))-k02*ufl.inner(epml*solution,solution))*dx(CELL['PML'])
        ppml=-float(np.imag(integral(pml_form)))/(2*w*MU0)
        # Use the physical-side trace for the independent physical-domain
        # budget. Preserve both traces and the old average as diagnostics.
        indicator=fem.Function(fem.functionspace(domain,('DG',0)))
        indicator.x.array[:]=1
        for cell in ct.find(CELL['PML']):
            indicator.x.array[indicator.function_space.dofmap.cell_dofs(int(cell))]=0
        indicator.x.scatter_forward()
        physical_flux,pml_flux=interface_flux_traces(domain,solution,c['frequency_Hz'],indicator)
        outgoing=float(np.real(integral(physical_flux*ds(FACE['PML_INTERFACE']))))
        pml_side_outgoing=float(np.real(integral(pml_flux*ds(FACE['PML_INTERFACE']))))
        averaged_flux=interface_flux(domain,solution,c['frequency_Hz'],(0,0,(d['z_min_m']+d['z_max_m'])/2))
        averaged_outgoing=float(np.real(integral(averaged_flux*ds(FACE['PML_INTERFACE']))))
        total=sum(powers.values())+ppml
        balance=abs(pin-total)/max(abs(pin),abs(total),1e-30)
        power_tolerance=c['solver']['power_balance_tolerance']
        power_scale=max(abs(pin),1e-30)
        pml_flux_difference=abs(outgoing-ppml)/power_scale
        checks=_power_sanity_checks(pin,powers,ppml,outgoing,balance,relative_residual,
                                    power_tolerance,c['solver']['relative_residual_tolerance'])
        # A more accurate physical-side diagnostic must not hide an unresolved
        # jump of the discrete magnetic field across the PML interface.
        checks['interface_trace_agreement']=bool(
            np.isfinite(pml_side_outgoing) and
            abs(outgoing-pml_side_outgoing)/power_scale < power_tolerance)
        passed=all(checks.values())
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
        # DG0 test functions integrate power on each cell using the SAME quadrature
        # as the global material budget. Point samples above are visualization only.
        Q=fem.functionspace(domain,('DG',0)); test=ufl.TestFunction(Q)
        qexpr=.5*w*EPS_0*ufl.imag(ufl.inner(eps*solution,solution))
        local_power=fem.assemble_vector(fem.form(sum(qexpr*ufl.conj(test)*dx(tag) for tag in CELL.values() if tag!=CELL['PML'])))
        local_volume=fem.assemble_vector(fem.form(ufl.conj(test)*dx))
        cell_power=_owned_cell_integrals(local_power,Q,owned)
        cell_volume=_owned_cell_integrals(local_volume,Q,owned)
        np.savez_compressed(out/f'cells_rank{comm.rank:04d}.npz',xyz_m=centers[:owned],region=tags[:owned],E_peak_V_m=electric,B_peak_T=magnetic,q_abs_W_m3=heating,cell_absorbed_power_W=cell_power,cell_volume_m3=cell_volume)
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
                'interface_flux_vs_pml_relative_difference':pml_flux_difference,
                'interface_flux_definition':'physical-side trace, outward from non-PML domain',
                'pml_side_interface_flux_W':pml_side_outgoing,
                'averaged_interface_flux_W':averaged_outgoing,
                'interface_trace_jump_relative_to_input':abs(outgoing-pml_side_outgoing)/power_scale,
                'numerical_checks':checks,'numerical_checks_passed':bool(passed),'dofs':V.dofmap.index_map.size_global*V.dofmap.index_map_bs,
                'mpi_ranks':comm.size,'dolfinx':dolfinx.__version__,'petsc':str(PETSc.Sys.getVersion()),
                'scalar_type':str(PETSc.ScalarType),'linear_solver_options':options,
                'solver_diagnostics':solver_diagnostics, 'solver_analysis':analysis,
                'limitations':['Ideal gap-sheet source and PEC antenna; impedance is source-model dependent, not calibrated terminal impedance.',
                    'No coax, matching network, RF coil shielding, reflector, sheath, or plume plasma.',
                    ('Spatial cellwise plasma density/collision map supplied.' if profile else 'Fixed uniform plasma state; effective collision rate is a user assumption.'),
                    'Passing algebraic/power checks does not establish mesh, PML or physical convergence.']}
        checkpoint('export_complete')
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



def _result_exit_code(result):
    if result['status'] == 'analysis_completed':
        return 0 if result['analysis']['admission']['accepted'] else 3
    if result['status'] == 'memory_rejected': return 3
    if result['status'] == 'linear_solver_failed': return 4
    return 0 if result['numerical_checks_passed'] else 2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build');p.add_argument('--out');p.add_argument('--b-map');p.add_argument('--plasma-map');p.add_argument('--check-runtime',action='store_true');p.add_argument('--expected-ranks',type=int)
    p.add_argument('--linear-solver',choices=('auto','mumps','mumps_ooc','superlu_dist','gmres_asm'),
                   help='Linear solver; default HELICON_RF_LU_SOLVER or auto')
    p.add_argument('--mumps-memory-mb',type=int,help='OOC workspace MB per rank (default env or 3000); not total RSS cap')
    p.add_argument('--mumps-scratch-dir',help='Existing writable OOC scratch directory (or HELICON_RF_MUMPS_SCRATCH)')
    p.add_argument('--analysis-only',action='store_true',help='MUMPS symbolic analysis only; no factorization')
    p.add_argument('--job-memory-mb',type=float,help='Estimated whole-job admission budget, decimal MB; not an OS cap')
    p.add_argument('--memory-reserve-mb',type=float,default=2000,help='Headroom above assembled RSS plus MUMPS estimate')
    a=p.parse_args()
    if a.expected_ranks is not None:
        MPI,_,_=runtime()
        if a.expected_ranks < 1 or MPI.COMM_WORLD.size != a.expected_ranks:
            raise RuntimeError(f'Expected {a.expected_ranks} MPI ranks, got {MPI.COMM_WORLD.size}; check launcher/library pairing')
    if a.check_runtime:
        MPI,PETSc,dolfinx=runtime()
        selected=_linear_solver_options(PETSc,MPI.COMM_WORLD.size,a.linear_solver,a.mumps_memory_mb,a.mumps_scratch_dir)
        if MPI.COMM_WORLD.rank==0: print('RF linear solver: '+json.dumps(selected))
        if MPI.COMM_WORLD.rank==0: print(f'DOLFINx {dolfinx.__version__}, PETSc {PETSc.ScalarType}, PETSC_DIR={os.environ.get("PETSC_DIR", "(package default)")}, MPI ranks={MPI.COMM_WORLD.size}')
        return 0
    if not a.build or not a.out: p.error('--build and --out are required for a solve')
    result=solve(a.build,a.out,a.b_map,a.plasma_map,linear_solver=a.linear_solver,mumps_memory_mb=a.mumps_memory_mb,mumps_scratch_dir=a.mumps_scratch_dir,analysis_only=a.analysis_only,job_memory_mb=a.job_memory_mb,memory_reserve_mb=a.memory_reserve_mb)
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank==0: print(json.dumps(result,indent=2))
    return _result_exit_code(result)

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
