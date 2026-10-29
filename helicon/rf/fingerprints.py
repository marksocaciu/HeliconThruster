"""Exact representation fingerprints; unequal hashes do not prove unequal physics."""
import hashlib
import numpy as np


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(memoryview(a).cast('B'))
    return digest.hexdigest()


def assembly_fingerprint(A, b, comm, PETSc):
    # CSR is rank-local: no global matrix gather. PETSc may copy local CSR data.
    ia, ja, values = A.getValuesCSR()
    local = {'rank': comm.rank, 'rows': list(A.getOwnershipRange()),
             'structure': array_digest(ia, ja),
             'values': array_digest(values), 'rhs': array_digest(b.getArray(readonly=True))}
    return {'ranks': comm.allgather(local),
            'matrix_frobenius_norm': float(A.norm(PETSc.NormType.FROBENIUS)),
            'rhs_norm': float(b.norm()),
            'interpretation': 'Exact distributed representation hashes, not invariant under basis changes or reordering.'}
