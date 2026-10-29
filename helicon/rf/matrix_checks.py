"""Opt-in matrix diagnostics for the coarse RF ILU comparison.

Symmetry and positive diagonals are necessary checks, not an SPD proof.
"""
import math
import numpy as np
from mpi4py import MPI


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def stable_norm(values):
    a = np.abs(values)
    scale = float(np.max(a)) if a.size else 0.
    if not math.isfinite(scale): return scale
    return scale * float(np.sqrt(np.sum((a / scale)**2))) if scale else 0.


def matrix_checks(matrix, PETSc):
    comm = matrix.comm.tompi4py()
    norm = float(matrix.norm(PETSc.NormType.FROBENIUS))
    # One temporary transpose/difference matrix, released before any next block.
    difference = matrix.copy()
    difference.hermitianTranspose()
    try:
        difference.axpy(-1, matrix, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        skew = float(difference.norm(PETSc.NormType.FROBENIUS))
    finally:
        difference.destroy()
    diagonal = matrix.getDiagonal()
    try:
        d = diagonal.getArray(readonly=True)
        good = np.isfinite(d)
        positive = np.abs(d[good]); positive = positive[positive > 0]
        minimum = comm.allreduce(float(positive.min()) if positive.size else math.inf, op=MPI.MIN)
        maximum = comm.allreduce(float(np.max(np.abs(d[good]))) if good.any() else 0., op=MPI.MAX)
        zeros = comm.allreduce(int(np.count_nonzero(d == 0)), op=MPI.SUM)
        bad = comm.allreduce(int(np.count_nonzero(~good)), op=MPI.SUM)
        nonpositive = comm.allreduce(int(np.count_nonzero(d.real <= 0)), op=MPI.SUM)
    finally:
        diagonal.destroy()
    _, _, values = matrix.getValuesCSR()
    local_imag = stable_norm(values.imag)
    imag = math.hypot(*comm.allgather(local_imag))
    nonfinite = comm.allreduce(int(np.count_nonzero(~np.isfinite(values))), op=MPI.SUM)
    return {'shape': list(matrix.getSize()), 'frobenius_norm': finite(norm),
            'relative_hermitian_defect': finite(skew/max(norm, 1e-300)),
            'relative_imaginary_norm': finite(imag/max(norm, 1e-300)),
            'diagonal_abs_min_nonzero': finite(minimum), 'diagonal_abs_max': finite(maximum),
            'diagonal_dynamic_range': finite(maximum/minimum) if math.isfinite(minimum) else None,
            'zero_diagonals': zeros, 'nonpositive_real_diagonals': nonpositive,
            'nonfinite_diagonals': bad, 'nonfinite_entries': nonfinite}


def local_pc_checks(pc, PETSc):
    """Independent normalized-ones probe per local block, not ASM's restricted b."""
    blocks = []
    for index, sub in enumerate(pc.getASMSubKSP()):
        _, matrix = sub.getOperators()
        stats = matrix_checks(matrix, PETSc)
        q = matrix.createVecRight(); z = q.duplicate(); work = q.duplicate()
        try:
            q.set(1); q.scale(1/q.norm())
            sub.getPC().apply(q, z)
            matrix.mult(z, work); work.axpy(-1, q)
            zn, defect = float(z.norm()), float(work.norm())
            blocks.append({'block': index, 'matrix': stats,
                           'unit_ones_output_norm': finite(zn),
                           'unit_ones_auxiliary_defect': finite(defect),
                           'probe_finite': math.isfinite(zn) and math.isfinite(defect)})
        finally:
            q.destroy(); z.destroy(); work.destroy()
    return blocks
