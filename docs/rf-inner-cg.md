# Bounded global inner CG diagnostic

This mode compares the normal positive-P ASM/ILU(0) baseline with outer FGMRES
using a Python PC that solves P z = r using global CG/Jacobi. Both use the same
assembled A, b, P and zero initial guesses. This is not a production solver.

```bash
cd ~/Documents/HeliconThruster
REF="$PWD/runs/rf-reference-20261027-121634"
BUILD="$REF/prepared/mesh" \
RUN_ROOT="$REF/inner-cg-check-$(date +%Y%m%d-%H%M%S)" \
SOLVER=fgmres_positive RANKS=2 BLAS_THREADS=3 \
HELICON_RF_PC_MASS_SCALE=10 HELICON_RF_ILU_LEVELS=0 \
HELICON_RF_ASM_OVERLAP=1 HELICON_RF_GMRES_RESTART=100 \
HELICON_RF_KSP_MAX_IT=25 HELICON_RF_REPEAT_SOLVES=0 \
HELICON_RF_MASS_SCALES="" HELICON_RF_COMPARE_ILU="" \
HELICON_RF_COMPARE_SCALED_ILU=0 HELICON_RF_COMPARE_INNER_CG=1 \
HELICON_RF_INNER_CG_RTOL=0.01 HELICON_RF_INNER_CG_MAX_IT=100 \
JOB_MEMORY_MB=24000 MEMORY_SCOPE=1 ANALYSIS_ONLY=0 BLR_TOL= \
bash run.sh
```

Two outer trials, each capped at 25 iterations. The CG trial allows up to 100
inner iterations per PC application, targeting a 1% true relative residual.
Every application starts from zero. The inner iteration limit is recorded and
its approximate result is allowed in flexible GMRES. Other inner divergence
reasons or a nonfinite true relative residual interrupt the diagnostic trial;
known breakdowns are recorded without being accepted as convergence. Unexpected
PETSc errors propagate after saving available evidence. Absence of a CG
breakdown does not prove P is positive definite.

Read solution/inner_cg_diagnostics.json: baseline, outer_result, inner_applications,
inner_breakdown if present, and A_b_P_unchanged. Each inner entry records the reason,
iterations, true relative residual, tolerance-met flag, and solve/residual-check
elapsed time. The console prints the same information per application. The
solution/inner-cg folder contains the outer history and PETSc setup.

The original baseline governs the usual run status/exit code and field outputs.
The comparison does not promote a candidate to a physically accepted solution.
Exit 4 is expected when the baseline fails this short budget. A failed/incomplete
comparison may lack A_b_P_unchanged; never interpret a missing flag as true.

Interpretation: if inner residuals remain large, this run cannot assess the
quality of an accurate P inverse; a stronger P solver is needed. If inner
residuals are controlled, compare outer reduction against ILU(0). A 25-iteration
trial is only an initial diagnostic, not a full convergence verdict.

Optional local regression:

```bash
HELICON_FEM_DIAGNOSTICS=1 mpiexec -n 2 python3 -m unittest discover -s tests -p test_rf_inner_cg.py
```
