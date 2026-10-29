# Fixed-system ILU diagnostic

After updating the Linux checkout, run:

```bash
cd ~/Documents/HeliconThruster
REF="$PWD/runs/rf-reference-20261027-121634"
BUILD="$REF/prepared/mesh" \
RUN_ROOT="$REF/ilu-check-$(date +%Y%m%d-%H%M%S)" \
SOLVER=fgmres_positive RANKS=2 BLAS_THREADS=3 \
HELICON_RF_PC_MASS_SCALE=10 HELICON_RF_ILU_LEVELS=0 \
HELICON_RF_COMPARE_ILU="1 0" \
HELICON_RF_ASM_OVERLAP=1 HELICON_RF_GMRES_RESTART=100 \
HELICON_RF_KSP_MAX_IT=200 HELICON_RF_REPEAT_SOLVES=0 \
HELICON_RF_MASS_SCALES="" \
JOB_MEMORY_MB=24000 MEMORY_SCOPE=1 ANALYSIS_ONLY=0 BLR_TOL= \
bash run.sh
```

Three attempts: original ILU(0), fresh ILU(1), fresh ILU(0). Each starts from
zero; no A, b or P reassembly. The original field is preserved and controls
normal run acceptance/exit code. Supplementary trials are diagnostics only.
Exit 4 is expected if the original solve fails the 200-iteration budget.

Read solution/ilu_diagnostics.json. A_b_P_unchanged must be true. Compare
true_relative_residual and pc_probe across attempts. For z = PC(b), probes
record ||z||, ||z||/||b||, ||Pz-b||/||b|| and ||Az||/||b||, plus finite flags
and actual ASM subsolver types. One application is not an exact solve of P;
a large auxiliary defect is not independently a failure criterion. The RHS
probe samples one direction and cannot establish complete preconditioner stability.

The baseline PC is probed after its solve. Fresh PCs are probed before solving,
while their options are active. ilu-baseline-setup.txt and ilu-02/solver_setup.txt
(and ilu-03) show PETSc's actual configuration, including factor levels. Each
additional attempt saves its history and pc_probe.json before iteration.
Reports are saved incrementally; incomplete reports lack A_b_P_unchanged.

Only levels 0 and 1 are permitted, with up to three additional attempts.
Do not combine this mode with repeat-solves or mass-scale comparison modes.
The 24 GB cgroup limit remains active. This diagnostic does not establish
physical validity or fix the convergence failure.

Optional regression test with a compatible PETSc/MPI installation:

```bash
HELICON_FEM_DIAGNOSTICS=1 mpiexec -n 2 python3 -m unittest discover -s tests -p test_rf_ilu.py
```

## Matrix and rank diagnostics

The comparison now saves `solution/P_matrix_checks.json` and embeds those checks
in `ilu_diagnostics.json`. Global P and each local ASM matrix are checked for
relative Hermitian defect, relative imaginary-entry norm, zero/nonpositive real
diagonals, nonfinite entries, and diagonal magnitude range. These are not a proof
of positive definiteness. Symmetry checks use temporary sparse matrix storage;
keep this diagnostic on the coarse reference under the existing memory cap.

Each attempt's `pc_probe.rank_diagnostics` contains owned portions of the real
ASM output PC(b), and a separate unit-norm all-ones probe of each local ILU.
The latter identifies local amplification but is not the actual restricted RHS.
A small response to this single direction does not prove stability.

The user's conditional RCM edit for ILU(1) is retained; the ILU(0) control keeps
its original ordering. For a short diagnostic use the command above with
HELICON_RF_KSP_MAX_IT=25. The probes, matrix checks and unchanged-system check
still run; do not use that short iteration budget to assess final convergence.

## Diagonally equilibrated local ILU

Set HELICON_RF_COMPARE_SCALED_ILU=1 with HELICON_RF_COMPARE_ILU="1" to run
baseline ILU(0)/natural, ILU(1)/RCM, then scaled ILU(1)/RCM in one process.
Use 25 iterations initially. Each local block B is copied and scaled to D B D,
where D_ii=1/sqrt(B_ii). Apply the mapped inverse D ILU(D B D)^(-1) D.
Invalid, nonpositive or significantly complex diagonals reject setup explicitly.
Original A, b, P and the subdomain matrices are preserved. Extra local matrix
storage is required; retain the 24 GB scope limit on the coarse mesh.

The report uses diagonally_scaled to identify variants, records the scaled
local diagonal ranges (approximately one), and retains existing PC output
probes. The Python local PC wraps a PETSc ILU, whose setup is included in the
viewer output. Local matrix checks still refer to original unscaled blocks.
Scaling is not a new physical operator and is not guaranteed to cure ILU
instability. The normal exit code still describes the original solve.
