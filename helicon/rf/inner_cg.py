"""Bounded global CG/Jacobi application of P inside outer FGMRES."""
import math
import time
from types import SimpleNamespace
from .diagnostics import IterationMonitor, linear_diagnostics
from .fingerprints import assembly_fingerprint
from ..records import write_json


class InnerBreakdown(RuntimeError):
    pass


class InnerCG:
    def __init__(self, P, PETSc, comm, rtol, max_it):
        self.P, self.PETSc, self.comm = P, PETSc, comm
        self.rows = []
        self.failure = None
        self.inner = PETSc.KSP().create(comm)
        self.inner.setOptionsPrefix('helicon_rf_inner_cg_')
        self.inner.setOperators(P)
        self.inner.setType('cg')
        self.inner.getPC().setType('jacobi')
        self.inner.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        self.inner.setTolerances(rtol=rtol, atol=0., max_it=max_it)
        self.inner.setInitialGuessNonzero(False)
        self.inner.setErrorIfNotConverged(False)
        self.work = P.createVecLeft()
        self.rtol = rtol

    def apply(self, pc, rhs, result):
        start = time.monotonic()
        result.set(0)
        self.inner.solve(rhs, result)
        self.P.mult(result, self.work); self.work.axpy(-1, rhs)
        bn = float(rhs.norm()); residual = float(self.work.norm())
        relative = residual/max(bn, 1e-300)
        reason = int(self.inner.getConvergedReason())
        name = next((n for n in dir(self.PETSc.KSP.ConvergedReason)
                     if not n.startswith('_') and getattr(self.PETSc.KSP.ConvergedReason,n)==reason),str(reason))
        finite = math.isfinite(relative)
        row = {'application': len(self.rows)+1, 'iterations': int(self.inner.getIterationNumber()),
               'reason': reason, 'reason_name': name, 'rhs_norm': bn if math.isfinite(bn) else None,
               'true_relative_residual': relative if finite else None,
               'true_tolerance_met': finite and relative <= self.rtol,
               'elapsed_s': time.monotonic()-start}
        self.rows.append(row)
        if self.comm.rank == 0: print('RF inner CG: '+str(row), flush=True)
        # Iteration-limited applications are allowed in flexible GMRES, but
        # explicitly recorded as approximate. Stop on other failures.
        if not finite or (reason <= 0 and reason != int(getattr(self.PETSc.KSP.ConvergedReason, 'DIVERGED_MAX_IT', -3))):
            self.failure = 'Inner CG failed: '+name
            raise InnerBreakdown(self.failure)

    def destroy(self, pc):
        if self.inner is not None:
            self.inner.destroy(); self.work.destroy(); self.inner = None

    def view(self, pc, viewer):
        self.inner.view(viewer)


def compare_inner_cg(problem, options, PETSc, comm, out, tolerance, first, rtol, max_it):
    from .solve import _solve_with_options
    A, P = problem.solver.getOperators()
    before = assembly_fingerprint(A, problem.b, comm, PETSc)
    p_before = assembly_fingerprint(P, problem.b, comm, PETSc)
    problem.solver.reset()
    directory = out/'inner-cg'
    error = None
    if comm.rank == 0:
        try: directory.mkdir(exist_ok=True)
        except OSError as exc: error = str(exc)
    error = comm.bcast(error,root=0)
    if error: raise RuntimeError(error)
    solver = PETSc.KSP().create(comm)
    x = problem.x.duplicate(); x.set(0)
    context = InnerCG(P,PETSc,comm,rtol,max_it)
    monitor = None
    report = {'baseline': first, 'inner_rtol': rtol, 'inner_max_it': max_it,
              'inner_applications': context.rows, 'assembly_before': before, 'P_before': p_before}
    try:
        solver.setOptionsPrefix('helicon_rf_cg_outer_')
        solver.setOperators(A,P)
        solver.setInitialGuessNonzero(False)
        monitor = IterationMonitor(comm,directory); solver.setMonitor(monitor)
        trial_options = {k:v for k,v in options.items() if not k.startswith(('pc_', 'sub_'))}
        trial_options['pc_type'] = 'python'
        def run():
            solver.getPC().setPythonContext(context)
            solver.setUp()
            viewer=PETSc.Viewer().createASCII(str(directory/'solver_setup.txt'),comm=comm)
            try: solver.view(viewer)
            finally: viewer.destroy()
            solver.solve(problem.b,x)
        fixed=SimpleNamespace(A=A,b=problem.b,x=x,solver=solver,solve=run)
        try:
            _solve_with_options(fixed,trial_options,PETSc)
        except (InnerBreakdown, PETSc.Error):
            if context.failure is None: raise
            report['inner_breakdown'] = context.failure
        diagnostics=linear_diagnostics(fixed,PETSc,tolerance)
        if report.get('inner_breakdown'): diagnostics['accepted']=False
        report['outer_result']=diagnostics
        monitor.close()
        report['history_write_error']=comm.bcast(monitor.error,root=0)
    finally:
        # Preserve evidence even if PETSc reports an unexpected error.
        if comm.rank == 0: write_json(out/'inner_cg_diagnostics.json',report)
        if monitor is not None: monitor.close()
        solver.destroy(); context.destroy(None); x.destroy()
    report['assembly_after']=assembly_fingerprint(A,problem.b,comm,PETSc)
    report['P_after']=assembly_fingerprint(P,problem.b,comm,PETSc)
    report['A_b_P_unchanged']=before==report['assembly_after'] and p_before==report['P_after']
    if comm.rank == 0:
        write_json(out/'inner_cg_diagnostics.json',report)
        print('RF inner-CG outer result: '+str(report['outer_result']),flush=True)
    if not report['A_b_P_unchanged']: raise RuntimeError('Inner CG comparison changed A, b or P')
    return report
