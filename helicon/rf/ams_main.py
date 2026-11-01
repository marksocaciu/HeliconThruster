"""Main RF solve using the persistent real AMS worker."""
import os
from types import SimpleNamespace
from .ams_bridge import AMSBridge, BridgeError
from .ams_protocol import atomic_json
from .ams_transfer import collective_io, export_system


def solve_ams(problem, options, PETSc, comm, out, V, degree, constrained, **overrides):
    from .solve import _solve_with_options
    configuration = dict(
        rtol=float(os.environ.get('HELICON_RF_AMS_RTOL', '0.01')),
        max_it=int(os.environ.get('HELICON_RF_AMS_MAX_IT', '100')),
        timeout=float(os.environ.get('HELICON_RF_AMS_TIMEOUT', '300')),
        python=os.environ.get('HELICON_RF_AMS_PYTHON', '/usr/bin/python3'),
        mpiexec=os.environ.get('HELICON_RF_AMS_MPIEXEC', 'mpiexec'),
        petsc_dir=os.environ.get('HELICON_RF_AMS_PETSC_DIR', '/usr/lib/petscdir/petsc-real'),
        cap_mb=int(os.environ.get('JOB_MEMORY_MB', '24000')))
    configuration.update(overrides)
    bridge = AMSBridge(problem.P_mat, comm, out/'ams-export', out/'ams-worker', **configuration)
    report = {'configuration': configuration, 'status': 'starting'}

    class Context:
        def setUp(self, pc):
            # LinearProblem has now assembled A, P, and the lifted RHS.
            if bridge.ready is None:
                try:
                    export_system(problem, V, degree, constrained, out, comm)
                    bridge.start()
                    report['worker_ready'] = bridge.ready
                except Exception as exc:
                    bridge.failure = str(exc)
                    raise BridgeError(str(exc)) from exc

        def apply(self, pc, rhs, result):
            bridge.apply(pc, rhs, result)

    def run():
        problem.solver.getPC().setPythonContext(Context())
        problem.x.set(0)
        problem.solver.setInitialGuessNonzero(False)
        return problem.solve()

    try:
        solved = _solve_with_options(SimpleNamespace(solver=problem.solver, solve=run), options, PETSc)
        report['status'] = 'solve_returned'
    except (BridgeError, PETSc.Error):
        if bridge.failure is None:
            raise
        report.update(status='bridge_failed', error=bridge.failure)
        solved = problem.u
    finally:
        bridge.close()
        report['worker_shutdown'] = bridge.shutdown
        report['inner_applications'] = bridge.rows
        collective_io(comm, lambda: atomic_json(out/'ams_main_diagnostics.json', report) if comm.rank == 0 else None)
    return solved, report
