"""Element assembly shared by the modal and 3D cylinder benchmarks."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from ..rf.materials import EPS_0, MU0


def local_operator(basis, curl, epsilon, omega):
    return (curl.conj() @ curl.T
            - omega**2 * MU0 * EPS_0 * (basis.conj() @ epsilon @ basis.T))


def assemble(connectivity, blocks, size):
    rows = np.broadcast_to(connectivity[:, :, None], blocks.shape)
    cols = np.broadcast_to(connectivity[:, None, :], blocks.shape)
    return coo_matrix((blocks.ravel(), (rows.ravel(), cols.ravel())),
                      shape=(size, size)).tocsr()


def dirichlet_solve(matrix, fixed, lift, omega):
    free = ~fixed
    rhs = -(matrix @ lift)[free]
    x = lift.copy()
    x[free] = spsolve(matrix[free][:, free].tocsc(), rhs)
    if not np.isfinite(x).all():
        raise ArithmeticError("Nonfinite solution of the RF system")
    reaction = matrix @ x
    residual = np.linalg.norm(reaction[free]) / max(np.linalg.norm(rhs), 1e-300)
    input_power = -np.vdot(lift, reaction).imag / (2 * omega * MU0)
    return x, float(input_power), float(residual)
