"""Local symmetric diagonal scaling: z = D ILU(D B D)^(-1) D r."""
import numpy as np


class EquilibratedILU:
    def __init__(self, matrix, level, ordering, PETSc):
        self.diagonal = self.matrix = self.rhs = self.inner = None
        try:
            self.diagonal = matrix.getDiagonal()
            d = self.diagonal.getArray()
            if not np.all(np.isfinite(d)) or np.any(d.real <= 0) or np.any(np.abs(d.imag) > 1e-12*d.real):
                raise ValueError('Equilibration requires finite positive real diagonals')
            d[:] = 1 / np.sqrt(d.real)
            self.matrix = matrix.copy()
            self.matrix.diagonalScale(self.diagonal, self.diagonal)
            self.rhs = matrix.createVecRight()
            self.inner = PETSc.PC().create(matrix.comm)
            self.inner.setType('ilu')
            self.inner.setOperators(self.matrix)
            self.inner.setFactorLevels(level)
            self.inner.setFactorOrdering(ordering)
            self.inner.setUp()
        except Exception:
            self.destroy(None)
            raise

    def apply(self, pc, x, y):
        self.rhs.pointwiseMult(self.diagonal, x)
        self.inner.apply(self.rhs, y)
        y.pointwiseMult(self.diagonal, y)

    def destroy(self, pc):
        for name in ('inner', 'rhs', 'matrix', 'diagonal'):
            resource = getattr(self, name)
            if resource is not None:
                resource.destroy()
                setattr(self, name, None)

    def view(self, pc, viewer):
        self.inner.view(viewer)


def install_scaled_local_ilu(solver, level, ordering, PETSc, comm):
    # Extract ASM blocks without first building the unstable unscaled ILU.
    solver.setUp()
    contexts = []
    error = None
    try:
        for sub in solver.getPC().getASMSubKSP():
            _, matrix = sub.getOperators()
            context = EquilibratedILU(matrix, level, ordering, PETSc)
            contexts.append((sub, context))
    except Exception as exc:
        error = str(exc)
    errors = comm.allgather(error)
    if any(errors):
        for _, context in contexts: context.destroy(None)
        raise RuntimeError('Scaled local ILU setup failed: '+str(errors))
    block_scaling = []
    for sub, context in contexts:
        diagonal = context.matrix.getDiagonal()
        try:
            a = diagonal.getArray(readonly=True).real
            block_scaling.append({'scaled_diagonal_min': float(a.min()),
                                  'scaled_diagonal_max': float(a.max())})
        finally: diagonal.destroy()
        pc = sub.getPC()
        pc.setType('python')
        pc.setPythonContext(context)
        pc.setUp()
    return {'scaled_blocks_by_rank': comm.allgather(block_scaling), 'mapping': 'D * ILU(D B D)^-1 * D', 'level': level, 'ordering': ordering}
