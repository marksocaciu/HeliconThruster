"""Iteration diagnostics retained for successful and unsuccessful RF solves."""
import json
import math
from pathlib import Path
import time


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


class IterationMonitor:
    """KSP's reported norm is not necessarily the true algebraic residual."""
    def __init__(self, comm, directory, every=25):
        self.comm, self.every = comm, every
        self.start = time.monotonic()
        self.first = None
        self.file = None
        self.error = None
        if comm.rank == 0:
            try:
                self.file = (Path(directory)/'ksp_history.jsonl').open('w')
            except OSError as exc:
                self.error = str(exc)
        error = comm.bcast(self.error,root=0)
        if error: raise RuntimeError('Cannot open KSP history: '+error)

    def __call__(self, ksp, iteration, norm):
        if self.comm.rank != 0: return
        norm = _finite(norm)
        if self.first is None and norm is not None: self.first = norm
        ratio = norm/self.first if norm is not None and self.first else None
        entry = {'iteration':int(iteration), 'ksp_reported_residual_norm':norm,
                 'relative_to_initial_reported_norm':_finite(ratio) if ratio is not None else None,
                 'elapsed_s':time.monotonic()-self.start}
        # Do not throw a rank-local I/O exception from a collective KSP callback.
        if self.error is None:
            try:
                self.file.write(json.dumps(entry,allow_nan=False)+'\n')
                if iteration % self.every == 0: self.file.flush()
            except OSError as exc:
                self.error = str(exc)
        if iteration % self.every == 0:
            print('RF KSP: '+json.dumps(entry,allow_nan=False),flush=True)

    def close(self):
        if self.file is not None:
            try: self.file.close()
            except OSError as exc: self.error = str(exc)


def linear_diagnostics(problem, PETSc, tolerance):
    """Read the final iterate even after iteration-limit/nonconvergence returns."""
    residual = problem.b.duplicate()
    try:
        problem.A.mult(problem.x,residual)
        residual.axpy(-1,problem.b)
        absolute = float(residual.norm())
        rhs_norm = float(problem.b.norm())
    finally:
        residual.destroy()
    relative = absolute/max(rhs_norm,1e-300)
    reason = int(problem.solver.getConvergedReason())
    name = next((name for name in dir(PETSc.KSP.ConvergedReason)
                 if not name.startswith('_') and getattr(PETSc.KSP.ConvergedReason,name)==reason),str(reason))
    return {'converged_reason':reason, 'converged_reason_name':name,
            'iterations':int(problem.solver.getIterationNumber()),
            'ksp_reported_residual_norm':_finite(problem.solver.getResidualNorm()),
            'true_absolute_residual':_finite(absolute), 'rhs_norm':_finite(rhs_norm),
            'true_relative_residual':_finite(relative),
            'accepted':bool(reason > 0 and math.isfinite(relative) and relative <= tolerance),
            'required_true_relative_residual':tolerance}
