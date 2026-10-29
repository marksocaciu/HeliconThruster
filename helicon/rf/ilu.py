"""Controlled ILU comparisons on an already assembled RF system."""
import json
import math
from types import SimpleNamespace
from .diagnostics import IterationMonitor, linear_diagnostics
from .fingerprints import assembly_fingerprint
from ..records import write_json


def probe_pc(solver, A, P, b, comm):
    """Apply the actual ASM PC to b; report amplification and auxiliary defect.

    P z - b tests one application, not convergence of a P solve. ASM/ILU
    need not make this defect small. Large values are diagnostic, not proof
    of a specific pivot or assembly failure.
    """
    from petsc4py import PETSc
    from .matrix_checks import local_pc_checks, stable_norm
    solver.setUp()
    pc = solver.getPC()
    z = b.duplicate(); work = b.duplicate()
    def finite(value):
        value = float(value)
        return value if math.isfinite(value) else None
    try:
        pc.apply(b, z)
        bn = float(b.norm()); zn = float(z.norm())
        P.mult(z, work); work.axpy(-1, b)
        defect = float(work.norm()) / max(bn, 1e-300)
        A.mult(z, work)
        az = float(work.norm()) / max(bn, 1e-300)
        subtypes = [{'ksp': sub.getType(), 'pc': sub.getPC().getType()}
                    for sub in pc.getASMSubKSP()]
        rank_output = {'rank': comm.rank,
                       'owned_rhs_norm': finite(stable_norm(b.getArray(readonly=True))),
                       'owned_pc_output_norm': finite(stable_norm(z.getArray(readonly=True))),
                       'local_blocks': local_pc_checks(pc, PETSc)}
        rank_outputs = comm.allgather(rank_output)
        return {'rank_diagnostics': rank_outputs, 'rhs_norm': finite(bn), 'pc_output_norm': finite(zn),
                'output_over_rhs': finite(zn/max(bn, 1e-300)),
                'Pz_minus_b_over_rhs': finite(defect), 'Az_over_rhs': finite(az),
                'finite': all(math.isfinite(v) for v in (bn, zn, defect, az)),
                'actual_pc': pc.getType(), 'subsolvers_by_rank': comm.allgather(subtypes)}
    finally:
        z.destroy(); work.destroy()


def compare_ilu(problem, options, PETSc, comm, out, tolerance, levels, first):
    from .solve import _solve_with_options
    A, P = problem.solver.getOperators()
    before = assembly_fingerprint(A, problem.b, comm, PETSc)
    p_before = assembly_fingerprint(P, problem.b, comm, PETSc)
    from .matrix_checks import matrix_checks
    p_checks = matrix_checks(P, PETSc)
    if comm.rank == 0: write_json(out/'P_matrix_checks.json', p_checks)
    first_probe = probe_pc(problem.solver, A, P, problem.b, comm)
    viewer = PETSc.Viewer().createASCII(str(out/'ilu-baseline-setup.txt'), comm=comm)
    try: problem.solver.view(viewer)
    finally: viewer.destroy()
    report = {'same_assembled_system': True, 'P_matrix_checks': p_checks, 'assembly_before': before,
              'P_before': p_before, 'attempts': [dict(first,
                  ilu_level=options['sub_pc_factor_levels'], pc_probe=first_probe)]}
    if comm.rank == 0:
        write_json(out/'ilu_diagnostics.json', report)
        print('RF ILU baseline probe: '+json.dumps(first_probe), flush=True)
    problem.solver.reset()  # Release old factors before constructing another PC.
    x = problem.x.duplicate()
    try:
        for index, level in enumerate(levels, start=2):
            directory = out / ('ilu-%02d' % index)
            error = None
            if comm.rank == 0:
                try: directory.mkdir(exist_ok=True)
                except OSError as exc: error = str(exc)
            error = comm.bcast(error, root=0)
            if error: raise RuntimeError(error)
            solver = PETSc.KSP().create(comm)
            monitor = None
            try:
                solver.setOptionsPrefix('helicon_rf_ilu_')
                solver.setOperators(A, P)
                x.set(0); solver.setInitialGuessNonzero(False)
                monitor = IterationMonitor(comm, directory)
                solver.setMonitor(monitor)
                trial_options = dict(options, sub_pc_factor_levels=level)
                if level == 1:
                    trial_options["sub_pc_factor_mat_ordering_type"] = "rcm"
                probe = {}
                def run():
                    probe.update(probe_pc(solver, A, P, problem.b, comm))
                    viewer = PETSc.Viewer().createASCII(str(directory/'solver_setup.txt'), comm=comm)
                    try: solver.view(viewer)
                    finally: viewer.destroy()
                    if comm.rank == 0:
                        write_json(directory/'pc_probe.json', probe)
                        print('RF ILU probe level '+str(level)+': '+json.dumps(probe), flush=True)
                    solver.solve(problem.b, x)
                fixed = SimpleNamespace(A=A, b=problem.b, x=x, solver=solver, solve=run)
                _solve_with_options(fixed, trial_options, PETSc)
                row = dict(linear_diagnostics(fixed, PETSc, tolerance), ilu_level=level, pc_probe=probe)
                monitor.close()
                row['history_write_error'] = comm.bcast(monitor.error, root=0)
                report['attempts'].append(row)
                if comm.rank == 0:
                    write_json(directory/'solver_diagnostics.json', row)
                    write_json(out/'ilu_diagnostics.json', report)
                    print('RF ILU result: '+json.dumps(row), flush=True)
            finally:
                if monitor is not None: monitor.close()
                solver.destroy()
    finally:
        x.destroy()
    report['assembly_after'] = assembly_fingerprint(A, problem.b, comm, PETSc)
    report['P_after'] = assembly_fingerprint(P, problem.b, comm, PETSc)
    report['A_b_P_unchanged'] = before == report['assembly_after'] and p_before == report['P_after']
    if comm.rank == 0: write_json(out/'ilu_diagnostics.json', report)
    if not report['A_b_P_unchanged']: raise RuntimeError('ILU comparison changed A, b or P')
    return report
