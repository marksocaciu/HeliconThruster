#!/usr/bin/env bash
set -euo pipefail
cd ~/Documents/HeliconThruster

# ---- tunables ----------------------------------------------------------
RANKS=${RANKS:-1}                    # 1 (safe) or 2
BLAS_THREADS=${BLAS_THREADS:-6}      # ranks * threads <= 6 physical cores
MEM_MB=${MEM_MB:-20000}              # MUMPS OOC workspace PER RANK (use 10000 for 2 ranks)
ORDERING=${ORDERING:-3}              # 3=Scotch, 2=AMF, 6=QAMD, 4=PORD (old default)
BLR_TOL=${BLR_TOL:-}                 # e.g. 1e-8; leave empty for exact LU
# ------------------------------------------------------------------------

unset PYTHONPATH PETSC_ARCH PETSC_OPTIONS
export PYTHONNOUSERSITE=1
export PETSC_DIR=/usr/lib/petscdir/petsc-complex
export HELICON_RF_PETSC_DIR="$PETSC_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=$BLAS_THREADS MKL_NUM_THREADS=$BLAS_THREADS

# solver settings read by solve.py
export HELICON_RF_LU_SOLVER=mumps_ooc
export HELICON_RF_MUMPS_MEMORY_MB=$MEM_MB
export HELICON_RF_MUMPS_ORDERING=$ORDERING
[ -n "$BLR_TOL" ] && export HELICON_RF_MUMPS_BLR_TOL=$BLR_TOL

# ---- 1. mesh (interface refinement stays in config/rf_nagoya_smooth_pec.json) ----
PREVIOUS="$PWD/runs/rf-interface-1p25mm-20261026-140957/solution"
RUN_ROOT="$PWD/runs/rf-smooth-$(date +%Y%m%d-%H%M%S)"

/usr/bin/python3 -m helicon.rf.run \
  --config config/rf_nagoya_smooth_pec.json --case plasma --mesh-only \
  --plasma-summary "$PREVIOUS/input_plasma_summary.json" \
  --b-map "$PREVIOUS/background_B.npz" --out "$RUN_ROOT/prepared"

# ---- 2. out-of-core scratch ----
mkdir -p "$HOME/mumps-scratch"
export HELICON_RF_MUMPS_SCRATCH="$(mktemp -d "$HOME/mumps-scratch/rf-XXXXXXXX")"
trap 'rm -rf "$HELICON_RF_MUMPS_SCRATCH"' EXIT   # factors are ~100 GB, clean up always
free -h; df -h "$HELICON_RF_MUMPS_SCRATCH"

# ---- 3. solve (--out must not exist yet) ----
mpiexec -n "$RANKS" --bind-to none /usr/bin/python3 -u -m helicon.rf.solve \
  --build "$RUN_ROOT/prepared/mesh" --out "$RUN_ROOT/solution" \
  --expected-ranks "$RANKS" \
  --linear-solver mumps_ooc --mumps-memory-mb "$MEM_MB" \
  --mumps-scratch-dir "$HELICON_RF_MUMPS_SCRATCH" \
  2>&1 | tee "$RUN_ROOT/solve.log"
