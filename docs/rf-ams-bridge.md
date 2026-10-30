# Experimental persistent AMS comparison

The complex RF job exports its current real auxiliary operator P and starts a
separate real-PETSc MPI worker. The worker loads P, G and Pi once and retains its
AMS hierarchy. Each outer FGMRES application sends rank-local real/imaginary
components, solves both with FGMRES/AMS, and returns a complex vector. The parent
independently checks the P residual before accepting it. A, b and P fingerprints
must remain unchanged. The baseline solution is preserved.

This is a diagnostic comparison, not a production solver or physical validation.
The original baseline still controls run.json and exit status (usually exit 4
for this short experiment), even if the AMS candidate converges. Inspect
solution/ams_bridge_diagnostics.json for the candidate result.

Both MPI jobs run on the same machine and share the launcher's verified cgroup:
24 GB total, zero swap. The worker checks cgroup identity and limits. It uses
one thread per rank. No second 24 GB allocation is created. Fresh exports and
IPC directories are required. Atomic files, request sequence numbers, bounded
waits, and explicit process-group cleanup protect against stale responses and
failed workers. Filesystem communication adds overhead; measure before scaling.

## First coarse Linux run

Run from ~/Documents/HeliconThruster after transferring the repository changes:

```bash
REF="$PWD/runs/rf-reference-20261027-121634"
BUILD="$REF/prepared/mesh" \
RUN_ROOT="$REF/ams-bridge-$(date +%Y%m%d-%H%M%S)" \
SOLVER=fgmres_positive RANKS=2 BLAS_THREADS=1 \
HELICON_RF_PC_MASS_SCALE=10 HELICON_RF_ILU_LEVELS=0 \
HELICON_RF_ASM_OVERLAP=1 HELICON_RF_GMRES_RESTART=100 \
HELICON_RF_KSP_MAX_IT=25 HELICON_RF_REPEAT_SOLVES=0 \
HELICON_RF_MASS_SCALES="" HELICON_RF_COMPARE_ILU="" \
HELICON_RF_COMPARE_SCALED_ILU=0 HELICON_RF_COMPARE_INNER_CG=0 \
HELICON_RF_EXPORT_AMS=0 HELICON_RF_COMPARE_AMS_BRIDGE=1 \
HELICON_RF_AMS_RTOL=0.01 HELICON_RF_AMS_MAX_IT=100 \
JOB_MEMORY_MB=24000 MEMORY_SCOPE=1 ANALYSIS_ONLY=0 BLR_TOL= \
bash run.sh
```

Default worker: /usr/bin/python3, mpiexec, PETSC_DIR=/usr/lib/petscdir/petsc-real.
Override with HELICON_RF_AMS_PYTHON, HELICON_RF_AMS_MPIEXEC,
HELICON_RF_AMS_PETSC_DIR. HELICON_RF_AMS_TIMEOUT defaults to 300 seconds per
response (also worker idle timeout). The worker launch removes inherited MPI
runtime variables and complex PETSc settings; Linux launcher compatibility still
requires this first run.

## Inspect before extending

- solution/ams_bridge_diagnostics.json: candidate true A residual, unchanged
  fingerprints, worker startup/shutdown, inner applications, memory snapshots.
- solution/ams-worker/applications.json: independently checked P residuals and
  real/imaginary iteration counts. Every component must meet the 1% target;
  failure stops the comparison and retains its request/answer files.
- solution/ams-worker/worker.log and failed.json: worker errors if any.
- launcher_status.json: shared job peak and exit status.

Compare the baseline and AMS residual after the same 25 iterations. Only extend
if the candidate makes useful progress, P applications pass, worker exits cleanly,
and shared peak memory leaves headroom. Do not jump to the large mesh yet.

## Local regression

The opt-in test builds a small boundary-constrained degree-2 system, uses a
persistent two-rank worker, checks multiple applications and immutable originals,
checks both complex RHS components, and tests failed-launch cleanup:

```bash
HELICON_FEM_DIAGNOSTICS=1 mpiexec -n 2 python3 -m unittest discover \
  -s tests -p test_rf_ams_bridge.py
```

Use a real-PETSc+hypre environment with mpiexec beside the Python executable.
The test disables the Linux cgroup requirement explicitly; production comparison
does not. Local real-parent tests do not certify the Linux complex-parent run.

## Controlled inner-tolerance comparison

Add `HELICON_RF_AMS_RTOLS="0.01 0.001"` to the bridge command and set
`HELICON_RF_KSP_MAX_IT=100`. This list overrides HELICON_RF_AMS_RTOL.
Both trials start from zero in the same parent process, using the same A, b, P,
partition and export. Each trial starts a fresh worker/hierarchy sequentially;
there is never more than one worker job alive. All other options stay fixed.
The single-tolerance behavior is preserved when the list is empty.

Read `solution/ams_tolerance_comparison.json` for the compact results and
`solution/ams-tolerance-1/ams_bridge_diagnostics.json` and
`solution/ams-tolerance-2/ams_bridge_diagnostics.json` for full diagnostics.
Each trial's `ams-worker` directory contains its iteration history and worker log.
Compare true outer residual at the same iteration count, inner application time,
and successful residual checks. A materially smaller residual with 0.001 supports
inner accuracy as a contributor; similar progress suggests testing a different P.
Neither result alone proves the cause. Cgroup peak is cumulative across trials.
The original baseline still controls exit status; completed trials need not have
converged. A worker failure stops the sweep and retains diagnostics.
