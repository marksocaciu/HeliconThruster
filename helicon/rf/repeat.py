"""Repeat KSP solves without reimporting the mesh or reassembling A, P or b."""
from types import SimpleNamespace
from .diagnostics import IterationMonitor, linear_diagnostics
from ..records import write_json


def repeat_fixed_system(problem, options, PETSc, comm, out, tolerance, count, first, *, scales=None, rebuild=None, initial_scale=None):
    from .solve import _solve_with_options
    A, P = problem.solver.getOperators()
    # Keep the matrices owned by LinearProblem; release its potentially large PC.
    problem.solver.reset()
    rows = [dict(first, attempt=1)]
    if scales is not None: rows[0]['mass_scale'] = initial_scale
    filename = 'mass_sweep_diagnostics.json' if scales is not None else 'repeat_diagnostics.json'
    report = {'same_assembled_system': True, 'additional_attempts': count,
              'matrix_frobenius_norm': float(A.norm(PETSc.NormType.FROBENIUS)),
              'rhs_norm': float(problem.b.norm()), 'attempts': rows}
    if scales is not None:
        from .fingerprints import assembly_fingerprint
        report['assembly_before'] = assembly_fingerprint(A, problem.b, comm, PETSc)
    x = problem.x.duplicate()
    try:
        for attempt in range(2, count + 2):
            directory = out / (('mass-%02d' if scales is not None else 'repeat-%02d') % attempt)
            error = None
            if comm.rank == 0:
                try: directory.mkdir(exist_ok=True)
                except OSError as exc: error = str(exc)
            error = comm.bcast(error, root=0)
            if error: raise RuntimeError(error)
            solver = PETSc.KSP().create(comm)
            monitor = None
            try:
                solver.setOptionsPrefix('helicon_rf_repeat_')
                if scales is not None:
                    rebuild(P, scales[attempt-2])
                solver.setOperators(A, P)
                x.set(0)
                solver.setInitialGuessNonzero(False)
                monitor = IterationMonitor(comm, directory)
                solver.setMonitor(monitor)
                fixed = SimpleNamespace(A=A, b=problem.b, x=x, solver=solver,
                                        solve=lambda: solver.solve(problem.b, x))
                _solve_with_options(fixed, options, PETSc)
                row = linear_diagnostics(fixed, PETSc, tolerance)
                row['attempt'] = attempt
                if scales is not None: row['mass_scale'] = scales[attempt-2]
                monitor.close()
                row['history_write_error'] = comm.bcast(monitor.error, root=0)
                rows.append(row)
                if comm.rank == 0:
                    write_json(directory/'solver_diagnostics.json', row)
                    write_json(out/filename, report)
                    print('RF fixed-system repeat: '+str(row), flush=True)
            finally:
                if monitor is not None: monitor.close()
                solver.destroy()
    finally:
        x.destroy()
    if scales is not None:
        report['assembly_after'] = assembly_fingerprint(A, problem.b, comm, PETSc)
        report['A_and_b_unchanged'] = report['assembly_before'] == report['assembly_after']
        if comm.rank == 0: write_json(out/filename, report)
        if not report['A_and_b_unchanged']:
            raise RuntimeError('Controlled sweep changed A or b')
    return report
