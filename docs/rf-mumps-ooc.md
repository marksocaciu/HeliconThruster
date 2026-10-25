# Optional MUMPS out-of-core solve

Use `--linear-solver mumps_ooc`. Existing `auto`, `mumps` and `superlu_dist` choices are unchanged. The new mode sets MUMPS ICNTL(22)=1 and ICNTL(23) to a positive MB-per-process workspace budget (default 3000). It passes the scratch path through PETSc's mat_mumps_ooc_tmpdir option and enables MUMPS diagnostic printing with ICNTL(4)=2. No pivoting, compression or physical acceptance tolerances are changed.

Explicit CLI values override environment values. For coupling subprocesses/in-process solves, set HELICON_RF_LU_SOLVER=mumps_ooc, HELICON_RF_MUMPS_MEMORY_MB and HELICON_RF_MUMPS_SCRATCH. Workspace/scratch environment values are ignored for other backends. Explicit workspace/scratch CLI options require mumps_ooc, to avoid silent misuse.

The scratch directory must already exist and be writable on every rank; paths longer than 240 bytes are rejected. Use a fresh dedicated directory for each independent run, on the NVMe filesystem, not a RAM-backed /tmp. Do not share a scratch directory between concurrent runs. A normal MUMPS teardown manages its files; a process kill can leave files behind. Inspect and remove only the dedicated directory after all processes from that run have stopped. This implementation never recursively deletes scratch files.

3000 MB per rank means 12000 MB of configured MUMPS workspace for four ranks, NOT a 12 GB total-process limit. Matrices, vectors, other MUMPS structures and application allocations need additional RAM. Out-of-core still may fail from insufficient RAM/workspace/disk. Failure -19 can indicate an inadequate workspace budget; increasing it blindly may cause another OOM kill. No solve on the user's complex Linux stack has yet validated this setting.

## Run as mark, not root

After copying the update into ~/Documents/HeliconThruster:

```bash
cd ~/Documents/HeliconThruster
unset PYTHONPATH PETSC_ARCH
export PYTHONNOUSERSITE=1
export PETSC_DIR=/usr/lib/petscdir/petsc-complex
export HELICON_RF_PETSC_DIR="$PETSC_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HELICON_RF_LU_SOLVER=mumps_ooc
export HELICON_RF_MUMPS_MEMORY_MB=3000
BASE="$PWD/runs/rf-compact-20261023-194156"
OUT="$BASE/mumps-ooc-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$HOME/mumps-scratch"
export HELICON_RF_MUMPS_SCRATCH="$(mktemp -d "$HOME/mumps-scratch/rf-XXXXXXXX")"
df -h "$HELICON_RF_MUMPS_SCRATCH"
mpiexec -n 4 /usr/bin/python3 -m helicon.rf.solve \
 --check-runtime --expected-ranks 4 --linear-solver mumps_ooc
```

Only if preflight succeeds, reuse the unchanged prepared mesh:

```bash
set -o pipefail
mpiexec -n 4 /usr/bin/python3 -u -m helicon.rf.solve \
 --build "$BASE/prepared/mesh" --out "$OUT" --expected-ranks 4 \
 --linear-solver mumps_ooc \
 --mumps-memory-mb 3000 --mumps-scratch-dir "$HELICON_RF_MUMPS_SCRATCH" \
 2>&1 | tee "${OUT}.log"
```

Retain the log and output directory whether successful or not. Selected budget/path are stored in run.json before factorization and in the completed summary. Do not start feedback until the baseline also passes accuracy/convergence checks.

References: https://mumps-solver.org/doc/userguide_5.8.2.pdf and PETSc MATSOLVERMUMPS documentation.
