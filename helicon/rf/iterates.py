"""Collective, atomic RF iterate snapshots (Krylov bases are rebuilt on restart)."""
import json
import os
from pathlib import Path
import uuid
import numpy as np
from .fingerprints import assembly_fingerprint, array_digest


class IterateCheckpoint:
    def __init__(self, comm, directory, PETSc, every=100, resume=None, total_limit=1000):
        if every < 1:
            raise ValueError('checkpoint interval must be positive')
        self.comm, self.directory, self.PETSc = comm, Path(directory), PETSc
        self.every, self.resume, self.total_limit = every, resume, total_limit
        self.offset, self.best, self.last = 0, float('inf'), None

    def collective(self, action):
        value, error = None, None
        try:
            value = action()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        errors = self.comm.allgather(error)
        if any(errors):
            raise RuntimeError('RF checkpoint I/O failed: ' + str(errors))
        return value

    def prepare(self, ksp, b, x):
        A, _ = ksp.getOperators()
        self.fingerprint = assembly_fingerprint(A, b, self.comm, self.PETSc)
        if self.resume:
            path = Path(self.resume)
            metadata = self.collective(lambda: json.loads(path.read_text()))
            if metadata['fingerprint'] != self.fingerprint:
                raise ValueError('Checkpoint system or MPI layout differs from assembled system')
            self.offset = int(metadata['iteration'])
            if self.total_limit <= self.offset:
                raise ValueError('Total iteration limit must exceed saved iteration')
            values = self.collective(lambda: np.load(path.parent / metadata['files'][self.comm.rank], allow_pickle=False))
            valid = (values.shape == x.getArray().shape and np.all(np.isfinite(values))
                     and array_digest(values) == metadata['digests'][self.comm.rank])
            if not all(self.comm.allgather(bool(valid))):
                raise ValueError('Invalid checkpoint vector')
            x.getArray()[:] = values
            ksp.setInitialGuessNonzero(True)
            ksp.setTolerances(max_it=self.total_limit-self.offset)
        self.collective(lambda: self.directory.mkdir(parents=True, exist_ok=True))
        best_path = self.directory / 'best.json'
        previous = self.collective(lambda: json.loads(best_path.read_text()) if best_path.exists() else None)
        if previous is not None:
            if previous['fingerprint'] != self.fingerprint:
                raise ValueError('Output checkpoint directory belongs to a different system; use a new output directory')
            self.best = previous['reported_residual']
        self.ready = True

    def __call__(self, ksp, iteration, norm):
        if iteration % self.every == 0:
            vector = ksp.buildSolution()
            try:
                self.save(vector, iteration, norm)
            finally:
                vector.destroy()

    def save(self, vector, iteration, norm):
        values = vector.getArray(readonly=True)
        if not all(self.comm.allgather(bool(np.all(np.isfinite(values)) and np.isfinite(norm)))):
            return
        total = self.offset + int(iteration)
        if self.last == total:
            return
        generation = self.comm.bcast(uuid.uuid4().hex if self.comm.rank == 0 else None, root=0)
        filename = f'{generation}-rank-{self.comm.rank}.npy'
        def write_vector():
            with (self.directory / filename).open('wb') as handle:
                np.save(handle, values, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
        self.collective(write_vector)
        files = self.comm.allgather(filename)
        digests = self.comm.allgather(array_digest(values))
        metadata = dict(version=1, iteration=total, reported_residual=float(norm),
                        fingerprint=self.fingerprint, files=files, digests=digests,
                        note='Numerical iterate only; not an accepted physical solution. Krylov basis is rebuilt.')
        improved = norm < self.best
        def publish():
            if self.comm.rank != 0:
                return
            for name in (['latest.json', 'best.json'] if improved else ['latest.json']):
                target = self.directory / name
                temporary = target.with_suffix('.tmp')
                with temporary.open('w') as handle:
                    json.dump(metadata, handle, allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(target)
            # Keep only generations referenced by the two committed manifests.
            retained = set()
            for name in ('latest.json', 'best.json'):
                path = self.directory / name
                if path.exists():
                    retained.update(json.loads(path.read_text())['files'])
            for path in self.directory.glob('*-rank-*.npy'):
                if path.name not in retained:
                    path.unlink()
        self.collective(publish)
        if improved:
            self.best = norm
        self.last = total
