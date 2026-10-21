#!/usr/bin/env bash
# One controller; MPI RF children; optionally threaded transport BLAS.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: bash tools/run_spatial_mpi.sh RUN_ROOT [RF_RANKS=4] [TRANSPORT_THREADS=4] [CASE=medium]" >&2
    exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN_ROOT="$1"
RF_RANKS="${2:-4}"
TRANSPORT_THREADS="${3:-4}"
CASE="${4:-medium}"
if [[ ! "$RF_RANKS" =~ ^[1-9][0-9]*$ || ! "$TRANSPORT_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "RF ranks and transport threads must be positive integers." >&2
    exit 2
fi
if [[ "$CASE" != medium && "$CASE" != fine && "$CASE" != outer ]]; then
    echo "CASE must be medium, fine, or outer." >&2
    exit 2
fi

# Debian/Ubuntu complex packages. Override these paths for another installation.
HELICON_PYTHON="${HELICON_PYTHON:-/usr/bin/python3}"
HELICON_MPIEXEC="${HELICON_MPIEXEC:-mpiexec}"
unset PYTHONPATH PETSC_ARCH
export PYTHONNOUSERSITE=1
export HELICON_RF_PETSC_DIR="${HELICON_RF_PETSC_DIR:-/usr/lib/petscdir/petsc-complex}"
export PETSC_DIR="$HELICON_RF_PETSC_DIR"
export OMP_NUM_THREADS="$TRANSPORT_THREADS"
export OPENBLAS_NUM_THREADS="$TRANSPORT_THREADS"
export MKL_NUM_THREADS="$TRANSPORT_THREADS"
export BLIS_NUM_THREADS="$TRANSPORT_THREADS"

RF_CONFIG="$RUN_ROOT/config/rf-$CASE.json"
PLASMA_CONFIG="$RUN_ROOT/config/plasma-$CASE.json"
test -f "$RF_CONFIG"
test -f "$PLASMA_CONFIG"
TAG="$(date +%Y%m%d-%H%M%S)-r$RF_RANKS-t$TRANSPORT_THREADS"

"$HELICON_PYTHON" -m helicon.rf.solve --check-runtime

prepare_mesh() {
    local material="$1"
    local directory="$2"
    if [[ -f "$directory/mesh/rf.msh" ]]; then
        # Reuse only if the prepared configuration matches the requested case.
        "$HELICON_PYTHON" - "$RF_CONFIG" "$directory/mesh/rf.json" "$material" <<'PY'
import json
import sys
from pathlib import Path
expected = json.loads(Path(sys.argv[1]).read_text())
actual = json.loads(Path(sys.argv[2]).read_text())
expected['material_case'] = sys.argv[3]
if expected != actual:
    raise SystemExit('Existing mesh configuration differs. Use a new run root or prepared directory.')
PY
    else
        "$HELICON_PYTHON" -m helicon.rf.run \
            --config "$RF_CONFIG" --case "$material" \
            --mesh-only --mesh-scale 1 --out "$directory" \
            2>&1 | tee "$directory.log"
    fi
}

# Vacuum passivity/power checks must pass before attempting plasma feedback.
VACUUM_BUILD="$RUN_ROOT/vacuum-$CASE-mpi-prepared"
prepare_mesh vacuum "$VACUUM_BUILD"
VACUUM_OUT="$RUN_ROOT/vacuum-$CASE-$TAG"
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
    "$HELICON_MPIEXEC" -n "$RF_RANKS" "$HELICON_PYTHON" -m helicon.rf.solve \
    --build "$VACUUM_BUILD/mesh" --out "$VACUUM_OUT" \
    --expected-ranks "$RF_RANKS" \
    2>&1 | tee "$VACUUM_OUT.log"

PLASMA_BUILD="$RUN_ROOT/rf-$CASE-mpi-prepared"
prepare_mesh plasma "$PLASMA_BUILD"
COUPLED_OUT="$RUN_ROOT/coupled-$CASE-$TAG"

# This is ONE Python controller, not mpiexec -n N copies of the controller.
"$HELICON_PYTHON" -m helicon.rf.spatial_coupled \
    --build "$PLASMA_BUILD/mesh" \
    --plasma-config "$PLASMA_CONFIG" \
    --feed-power-W "${HELICON_FEED_POWER_W:-133.3333333333}" \
    --rf-ranks "$RF_RANKS" --mpi-launcher "$HELICON_MPIEXEC" \
    --max-iterations 40 --tolerance 0.005 --relaxation 0.4 \
    --out "$COUPLED_OUT" \
    2>&1 | tee "$COUPLED_OUT.log"

printf '\nCompleted coupled run: %s\n' "$COUPLED_OUT"
printf 'Plasma report: %s/final_plasma/report.html\n' "$COUPLED_OUT"
