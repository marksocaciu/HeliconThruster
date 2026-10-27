#!/usr/bin/env bash
# Usage: bash tools/sweep_rf_mass.sh [reference-directory] [mass-scales...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REF=${1:-"$PWD/runs/rf-reference-20261027-121634"}
if (( $# )); then shift; fi
SCALES=("${@:-}")
if (( $# == 0 )); then SCALES=(0.1 1 10); fi
PYTHON=${PYTHON:-/usr/bin/python3}
REF=$(cd "$REF" && pwd)
[[ -f "$REF/prepared/mesh/rf.msh" && -f "$REF/direct/solution/rf_summary.json" ]] || {
  echo 'Reference mesh or direct solution is missing.' >&2; exit 2;
}
"$PYTHON" - "${SCALES[@]}" <<'PY'
import math, sys
for text in sys.argv[1:]:
    try: value=float(text)
    except ValueError: raise SystemExit('Invalid mass scale: '+text)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit('Mass scales must be positive and finite: '+text)
PY
SWEEP=$(mktemp -d "$REF/mass-sweep-$(date +%Y%m%d-%H%M%S)-XXXXXX")
echo "Results: $SWEEP"
trap 'echo "Interrupted; no further trials will be launched. Results: $SWEEP" >&2; exit 130' INT
trap 'exit 143' TERM
index=0
for scale in "${SCALES[@]}"; do
  index=$((index + 1))
  TRIAL="$SWEEP/trial-$index"
  mkdir "$TRIAL"
  echo "=== Trial $index: mass scale $scale ==="
  set +e
  env BUILD="$REF/prepared/mesh" RUN_ROOT="$TRIAL" PYTHON="$PYTHON" \
    SOLVER=fgmres_positive RANKS=2 BLAS_THREADS=3 \
    HELICON_RF_PC_MASS_SCALE="$scale" HELICON_RF_ILU_LEVELS=0 \
    HELICON_RF_ASM_OVERLAP=1 HELICON_RF_GMRES_RESTART=100 \
    HELICON_RF_KSP_MAX_IT=1000 JOB_MEMORY_MB=24000 \
    MEMORY_SCOPE=1 ANALYSIS_ONLY=0 BLR_TOL= \
    bash run.sh 2>&1 | tee "$TRIAL/launcher.log"
  codes=("${PIPESTATUS[@]}")
  set -e
  run_code=${codes[0]}
  compare_code=0
  "$PYTHON" -m helicon.rf.compare --reference "$REF/direct/solution" \
    --candidate "$TRIAL/solution" > "$TRIAL/comparison.json" 2> "$TRIAL/comparison.stderr" || compare_code=$?
  "$PYTHON" - "$SWEEP" "$TRIAL" "$scale" "$run_code" "$compare_code" <<'PY'
import csv, json, sys
from pathlib import Path
sweep,trial=map(Path,sys.argv[1:3])
def read(path):
    try: return json.loads(path.read_text())
    except (OSError,ValueError): return {}
d=read(trial/'solution/solver_diagnostics.json')
c=read(trial/'comparison.json')
row={'mass_scale':sys.argv[3], 'run_exit':int(sys.argv[4]),
     'comparison_exit':int(sys.argv[5]), 'iterations':d.get('iterations'),
     'true_relative_residual':d.get('true_relative_residual'),
     'reason':d.get('converged_reason_name'),
     'peak_job_mb':c.get('candidate_peak_job_mb'),
     'comparison_passed':c.get('passed',False), 'directory':str(trial)}
(trial/'trial.json').write_text(json.dumps(row,indent=2)+'\n')
path=sweep/'summary.csv'
exists=path.exists()
with path.open('a',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(row))
    if not exists: writer.writeheader()
    writer.writerow(row)
print(json.dumps(row,indent=2))
PY
  # Numerical nonconvergence (4) and failed physics checks (2) are results.
  # Infrastructure errors or a memory kill stop the sweep; do not repeat them.
  if (( codes[1] != 0 || compare_code > 1 )); then
    echo "Logging/comparison error; stopping. See $TRIAL" >&2; exit 1
  fi
  case "$run_code" in
    0|2|4) ;;
    *) echo "Run exited $run_code; stopping. See $TRIAL" >&2; exit "$run_code" ;;
  esac
done
echo "Sweep complete. Summary: $SWEEP/summary.csv"
echo 'Compare true residuals at the same iteration budget; only comparison_passed=true is a passed reference comparison.'
