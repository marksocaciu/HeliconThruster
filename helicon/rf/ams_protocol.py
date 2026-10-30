"""Bounded, atomic file messages between independent PETSc MPI jobs."""
import json
import os
import time
from pathlib import Path
import numpy as np


def atomic_json(path, value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False)+'\n')
    os.replace(temporary,path)


def atomic_npz(path, **values):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream: np.savez(stream,**values)
    os.replace(temporary,path)


def wait_json(path, timeout, *, failed=None, process=None, sequence=None):
    end=time.monotonic()+timeout
    while True:
        if failed is not None and Path(failed).exists():
            raise RuntimeError('AMS worker failed: '+Path(failed).read_text())
        if Path(path).exists():
            value=json.loads(Path(path).read_text())
            if sequence is None or value.get('sequence')==sequence:return value
        if process is not None and process.poll() is not None:
            raise RuntimeError('AMS worker exited with code '+str(process.returncode))
        if time.monotonic()>=end:raise TimeoutError('Timed out waiting for '+str(path))
        time.sleep(0.05)


def cgroup_identity():
    path=Path('/proc/self/cgroup')
    if not path.exists():return None
    return next((line for line in path.read_text().splitlines() if line.startswith('0::')),None)


def clean_worker_environment(environment, petsc_dir):
    # MPI launchers must not attach a new job to the parent's PMI/PMIx session.
    blocked=('OMPI_', 'PMI_', 'PMIX_', 'PRTE_', 'PMI2_', 'I_MPI_', 'HYDRA_', 'MPICH_')
    result={key:value for key,value in environment.items() if not key.startswith(blocked)}
    for key in ('PETSC_ARCH','PETSC_OPTIONS','PYTHONPATH','PYTHONHOME'):result.pop(key,None)
    result.update(PETSC_DIR=petsc_dir,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1')
    return result
