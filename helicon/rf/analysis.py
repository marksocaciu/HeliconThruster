"""Collective, symbolic-only MUMPS analysis through a public C API bridge."""
import ctypes
import os
from pathlib import Path
import shlex
import subprocess
import numpy as np


def analyze_mumps(matrix, options, comm, directory, PETSc):
    directory = Path(directory)
    library = directory/'mumps_analysis.so'
    error = None
    if comm.rank == 0:
        try:
            root = Path(os.environ['PETSC_DIR']).resolve()
            env = dict(os.environ)
            # Use only the selected PETSc's metadata, not a real PETSc elsewhere.
            env['PKG_CONFIG_PATH'] = str(root/'lib/pkgconfig')
            pc = root/'lib/pkgconfig/PETSc.pc'
            if not pc.exists():
                pc = root/'lib/pkgconfig/petsc.pc'
            if not pc.exists():
                raise RuntimeError(f'PETSc development metadata missing in {root}/lib/pkgconfig')
            flags = subprocess.check_output(['pkg-config','--cflags','--libs',str(pc)],env=env,text=True)
            command = shlex.split(os.environ.get('MPICC','mpicc')) + [
                '-shared','-fPIC','-O2',str(Path(__file__).with_name('mumps_analysis.c')),
                '-o',str(library)] + shlex.split(flags)
            result = subprocess.run(command,env=env,text=True,capture_output=True)
            (directory/'analysis-build.log').write_text(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError('Native analysis helper build failed; see analysis-build.log')
        except Exception as exc:
            error = str(exc)
    error = comm.bcast(error,root=0)
    if error:
        raise RuntimeError(error+'; install the matching complex PETSc development package, pkg-config and MPI compiler')
    helper = None
    try:
        helper = ctypes.CDLL(str(library))
        abi = (ctypes.c_int*5)()
        helper.helicon_petsc_abi(abi)
        expected = (*PETSc.Sys.getVersion(),np.dtype(PETSc.IntType).itemsize,np.dtype(PETSc.ScalarType).itemsize)
        if tuple(abi) != expected:
            raise RuntimeError(f'Native PETSc ABI {tuple(abi)} differs from petsc4py {expected}')
    except Exception as exc:
        error = str(exc)
    errors = comm.allgather(error)
    if any(errors):
        raise RuntimeError(str(errors))
    prefix = 'helicon_rf_analysis_'
    database = PETSc.Options(prefix)
    keys = [k for k in options if k.startswith('mat_mumps_')]
    previous = {key: database.getString(key) if database.hasName(key) else None for key in keys}
    for key in keys:
        database[key] = options[key]
    call = helper.helicon_mumps_analysis
    call.argtypes = [ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_longlong)]
    call.restype = ctypes.c_int
    values = (ctypes.c_longlong*13)()
    try:
        code = call(matrix.handle,prefix.encode(),values)
        codes = comm.allgather(code)
        if any(codes):
            raise RuntimeError(f'MUMPS symbolic analysis failed with PETSc codes {codes}')
        data = comm.bcast(list(values) if comm.rank == 0 else None,root=0)
        if data[0] < 0:
            raise RuntimeError(f'MUMPS analysis INFOG(1:2)={data[:2]}')
        ooc = options.get('mat_mumps_icntl_22') == 1
        blr = options.get('mat_mumps_icntl_35',0) in (1,2)
        if blr and not ooc:
            raise ValueError('BLR analysis admission is currently supported only with mumps_ooc')
        total_index,max_index,local_index = (8,7,10) if blr else ((6,5,9) if ooc else (4,3,11))
        raw_estimates = comm.allgather(int(values[local_index]))
        if any(value < 0 for value in raw_estimates) or data[total_index] < 0:
            raise RuntimeError('Invalid MUMPS memory estimate')
        # MUMPS integer MB can truncate small per-rank estimates to zero.
        estimates = [value+1 for value in raw_estimates]
        return {'effective_ordering':data[2], 'requested_ordering':options.get('mat_mumps_icntl_7',7),
                'estimated_total_mb':max(data[total_index],sum(estimates)),
                'estimated_max_rank_mb':max(data[max_index],max(estimates)),
                'rank_reported_mb':raw_estimates,
                'rank_estimated_mb':estimates, 'out_of_core':ooc, 'blr':blr,
                'infog':dict(zip((1,2,7,16,17,26,27,38,39),data[:9]))}
    finally:
        for key in keys:
            if previous[key] is None: del database[key]
            else: database[key] = previous[key]
