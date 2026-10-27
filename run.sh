#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# All budgets are decimal MB. The job limit excludes other system processes.
RANKS=${RANKS:-2}
BLAS_THREADS=${BLAS_THREADS:-1}
MEM_MB=${MEM_MB:-10000}          # MUMPS workspace PER RANK; not total RSS
JOB_MEMORY_MB=${JOB_MEMORY_MB:-24000}
MEMORY_RESERVE_MB=${MEMORY_RESERVE_MB:-2000}
MEMORY_SCOPE=${MEMORY_SCOPE:-1}  # requires Linux systemd user services + cgroup v2
SOLVER=${SOLVER:-mumps_ooc}      # mumps_ooc, mumps, superlu_dist, gmres_asm
ORDERING=${ORDERING:-7}          # automatic; effective ordering is in analysis JSON
BLR_TOL=${BLR_TOL:-}
ANALYSIS_ONLY=${ANALYSIS_ONLY:-0}
PYTHON=${PYTHON:-/usr/bin/python3}
CONFIG=${CONFIG:-config/rf_nagoya_smooth_pec.json}
PREVIOUS=${PREVIOUS:-"$PWD/runs/rf-interface-1p25mm-20261026-140957/solution"}
RUN_ROOT=${RUN_ROOT:-"$PWD/runs/rf-memory-$(date +%Y%m%d-%H%M%S)-$$"}

for value in "$RANKS" "$BLAS_THREADS" "$MEM_MB" "$JOB_MEMORY_MB"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Ranks, threads and budgets must be positive integers' >&2; exit 2; }
done
[[ "$MEMORY_RESERVE_MB" =~ ^[0-9]+$ ]] || exit 2
[[ "$MEMORY_SCOPE" =~ ^[01]$ && "$ANALYSIS_ONLY" =~ ^[01]$ ]] || exit 2
case "$SOLVER" in mumps_ooc|mumps|superlu_dist|gmres_asm) ;; *) echo "Unknown SOLVER: $SOLVER" >&2; exit 2;; esac

# The entire launcher, mesh preparation, MPI children and charged cache share a cap.
# Explicit opt-out is useful on systems without systemd; it is never silent.
if [[ "$MEMORY_SCOPE" == 1 && "${HELICON_MEMORY_SCOPE_ACTIVE:-0}" != 1 ]]; then
  command -v systemd-run >/dev/null || { echo 'systemd-run is required for the memory cap; MEMORY_SCOPE=0 explicitly disables it' >&2; exit 2; }
  export RANKS BLAS_THREADS MEM_MB JOB_MEMORY_MB MEMORY_RESERVE_MB MEMORY_SCOPE
  export SOLVER ORDERING BLR_TOL ANALYSIS_ONLY PYTHON CONFIG PREVIOUS RUN_ROOT
  exec systemd-run --user --scope --quiet \
    -p "MemoryMax=$((JOB_MEMORY_MB * 1000000))" -p MemorySwapMax=0 \
    env HELICON_MEMORY_SCOPE_ACTIVE=1 bash "$PWD/run.sh"
fi
if [[ "$MEMORY_SCOPE" == 0 ]]; then echo 'WARNING: OS memory cap disabled; estimates do not guarantee peak RAM' >&2; fi

unset PYTHONPATH PETSC_ARCH PETSC_OPTIONS HELICON_RF_MUMPS_BLR_TOL
export PYTHONNOUSERSITE=1
export PETSC_DIR=${HELICON_RF_PETSC_DIR:-/usr/lib/petscdir/petsc-complex}
export HELICON_RF_PETSC_DIR="$PETSC_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS="$BLAS_THREADS" MKL_NUM_THREADS="$BLAS_THREADS"
export HELICON_RF_LU_SOLVER="$SOLVER" HELICON_RF_MUMPS_MEMORY_MB="$MEM_MB"
export HELICON_RF_MUMPS_ORDERING="$ORDERING"
if [[ -n "$BLR_TOL" ]]; then export HELICON_RF_MUMPS_BLR_TOL="$BLR_TOL"; fi

mkdir -p "$RUN_ROOT"
finish() {
  code=$?
  "$PYTHON" - "$RUN_ROOT" "$code" <<'PYEND' || true
import json, sys
from pathlib import Path
from helicon.rf.memory import cgroup_memory
Path(sys.argv[1], 'launcher_status.json').write_text(json.dumps(
    {'exit_code':int(sys.argv[2]), 'cgroup_memory':cgroup_memory()}, indent=2)+'\n')
PYEND
  if [[ -n "${HELICON_RF_MUMPS_SCRATCH:-}" ]]; then rm -rf -- "$HELICON_RF_MUMPS_SCRATCH"; fi
  exit "$code"
}
unset HELICON_RF_MUMPS_SCRATCH
trap finish EXIT
if [[ "$MEMORY_SCOPE" == 1 ]]; then
  "$PYTHON" - "$JOB_MEMORY_MB" <<'PYEND'
import sys
from helicon.rf.memory import cgroup_memory
counters=cgroup_memory()
maximum=counters.get('max')
if maximum is None or maximum > int(sys.argv[1])*1000000 or counters.get('swap.max') != 0:
    raise SystemExit('Expected cgroup RAM/swap limits are not active; refusing an unbounded run')
PYEND
fi
if [[ -z "${BUILD:-}" ]]; then
  "$PYTHON" -m helicon.rf.run --config "$CONFIG" --case plasma --mesh-only \
    --plasma-summary "$PREVIOUS/input_plasma_summary.json" \
    --b-map "$PREVIOUS/background_B.npz" --out "$RUN_ROOT/prepared" \
    2>&1 | tee "$RUN_ROOT/prepare.log"
  BUILD="$RUN_ROOT/prepared/mesh"
fi
args=(--build "$BUILD" --out "$RUN_ROOT/solution" --expected-ranks "$RANKS"
      --linear-solver "$SOLVER" --job-memory-mb "$JOB_MEMORY_MB"
      --memory-reserve-mb "$MEMORY_RESERVE_MB")
if [[ "$ANALYSIS_ONLY" == 1 ]]; then args+=(--analysis-only); fi
if [[ "$SOLVER" == mumps_ooc ]]; then
  scratch_parent=${SCRATCH_ROOT:-"$HOME/mumps-scratch"}
  mkdir -p "$scratch_parent"
  export HELICON_RF_MUMPS_SCRATCH="$(mktemp -d "$scratch_parent/rf-XXXXXXXX")"
  args+=(--mumps-memory-mb "$MEM_MB" --mumps-scratch-dir "$HELICON_RF_MUMPS_SCRATCH")
  df -h "$HELICON_RF_MUMPS_SCRATCH"
fi
free -h
mpiexec -n "$RANKS" --bind-to none "$PYTHON" -u -m helicon.rf.solve "${args[@]}" \
  2>&1 | tee "$RUN_ROOT/solve.log"
