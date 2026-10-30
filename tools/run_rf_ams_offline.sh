#!/usr/bin/env bash
# Usage: bash tools/run_rf_ams_offline.sh /absolute/path/to/solution/ams-export
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXPORT=$(cd "${1:?Pass the ams-export directory}" && pwd)
RANKS=${RANKS:-2}
RESULT=$(mktemp -d "${EXPORT}/test-XXXXXX")
echo "AMS result: $RESULT/result.json"
exec systemd-run --user --scope --quiet \
  -p MemoryMax=24000000000 -p MemorySwapMax=0 \
  env -u PETSC_ARCH -u PETSC_OPTIONS -u PYTHONPATH \
    PETSC_DIR=/usr/lib/petscdir/petsc-real \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    mpiexec -n "$RANKS" /usr/bin/python3 -m helicon.rf.ams_offline \
      --export "$EXPORT" --out "$RESULT/result.json" --max-it 100 --require-cap-mb 24000
