# Explicit SuperLU_DIST selection

The solve CLI accepts --linear-solver superlu_dist, mumps, or auto. An explicit CLI choice wins over HELICON_RF_LU_SOLVER. Unavailable explicit backends fail before mesh loading, rather than silently falling back. Auto preserves the existing MUMPS-first behavior.

Set HELICON_RF_LU_SOLVER=superlu_dist to use the same backend in RF solves launched by global/spatial coupling drivers. This only changes the linear solver, not acceptance checks or the physical boundary model.

SuperLU_DIST uses MMD_AT_PLUS_A column ordering to avoid requiring METIS/ParMETIS, and prints factorization statistics. Selected options and DOF count are written to run.json before the linear solve and retained in the successful RF summary. A hard process kill can still leave status=running, but the last stage and settings are recorded.

The user's Linux complex PETSc reports SuperLU_DIST available. Local unit tests validate option selection, not Linux memory use or numerical performance. No guarantee of reduced memory or successful convergence is implied.

Reuse the existing compact prepared mesh, with a fresh output directory:

```bash
cd ~/Documents/HeliconThruster
unset PYTHONPATH PETSC_ARCH
export PYTHONNOUSERSITE=1
export PETSC_DIR=/usr/lib/petscdir/petsc-complex
export HELICON_RF_PETSC_DIR="$PETSC_DIR"
export HELICON_RF_LU_SOLVER=superlu_dist
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE="$PWD/runs/rf-compact-20261023-194156"
OUT="$BASE/superlu-$(date +%Y%m%d-%H%M%S)"
mpiexec -n 4 /usr/bin/python3 -m helicon.rf.solve --check-runtime --expected-ranks 4 --linear-solver superlu_dist
set -o pipefail
mpiexec -n 4 /usr/bin/python3 -u -m helicon.rf.solve \
 --build "$BASE/prepared/mesh" --out "$OUT" --expected-ranks 4 \
 --linear-solver superlu_dist 2>&1 | tee "${OUT}.log"
```

Run the solve only if preflight succeeds. Keep the log even if the process terminates. A completed solve can still fail physical checks; inspect rf_summary.json before coupling.
