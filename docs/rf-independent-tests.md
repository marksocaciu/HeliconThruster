# Independent RF preconditioner tests

These commands run on your computer; no assistant/API calls are involved.
Use the updated repository on Linux with the same complex PETSc/DOLFINx setup.
No new mesh or repeat LU reference solve is required.

## What changed

`SOLVER=fgmres_positive` is an explicit experimental backend. It solves the
original `A E = b`, but applies ASM/ILU to a different matrix `P`:

```
p(u,v) = integral(curl(u) . conjugate(curl(v)) + beta*u . conjugate(v))
beta = mass_scale * k0^2 * max(1, Frobenius_norm(epsilon_r)/sqrt(3))
```

`mass_scale` defaults to 1. The positive material scale is a heuristic; it does
not guarantee rapid convergence. Permeability in P is the real identity,
including PML cells. Physical tensor permittivity, losses, PML and boundary
conditions in A remain unchanged. P uses the same space and constrained DOFs.
The extra sparse P matrix costs memory, but there is no global LU factorization.

In code, `forms.positive_preconditioner` constructs P's form and the solver uses:

```python
problem = LinearProblem(form, rhs, P=preconditioner_form, bcs=bcs,
                        petsc_options_prefix='helicon_rf_', petsc_options=None)
```

FGMRES uses right preconditioning, unpreconditioned residual norms and modified
Gram-Schmidt. The final explicit residual and existing physics checks still
gate acceptance. This is **not AMS**, and local ILU on P can still be ineffective.
The previous MUMPS and GMRES/ASM choices remain available unchanged.

## First check the comparison tool

From `~/Documents/HeliconThruster`, with the updated code:

```bash
REF="$PWD/runs/rf-reference-20261027-121634"
/usr/bin/python3 -m helicon.rf.compare \
  --reference "$REF/direct/solution" \
  --candidate "$REF/direct/solution"
```

This self-comparison should report `"passed": true`. Exit 0 means pass, 1 means
rejected, 2 means missing/malformed inputs or invalid arguments. It reads the
existing output files and does not run a simulation.

## Run one bounded candidate

```bash
REF="$PWD/runs/rf-reference-20261027-121634"
TRIAL="$REF/positive-s1-$(date +%Y%m%d-%H%M%S)"
BUILD="$REF/prepared/mesh" RUN_ROOT="$TRIAL" \
SOLVER=fgmres_positive RANKS=2 BLAS_THREADS=1 \
HELICON_RF_PC_MASS_SCALE=1 HELICON_RF_ILU_LEVELS=0 \
HELICON_RF_GMRES_RESTART=30 HELICON_RF_KSP_MAX_IT=1000 \
JOB_MEMORY_MB=24000 ANALYSIS_ONLY=0 MEMORY_SCOPE=1 bash run.sh
```

The run keeps the 24 GB cgroup cap. It may fail: this is an experiment, not a
verified RF convergence fix. Use a new TRIAL directory on every run.

## Check the result, even if the simulation returned nonzero

```bash
/usr/bin/python3 -m helicon.rf.compare \
  --reference "$REF/direct/solution" \
  --candidate "$TRIAL/solution" > "$TRIAL/comparison.json"
cat "$TRIAL/comparison.json"
```

The comparison requires:

- Identical mesh, configuration, magnetic map and optional plasma inputs.
- Both summaries completed with numerical checks passed.
- True relative residual at most 1e-8.
- Complex impedance and power agreement within 0.1% relative tolerance,
  plus a 1e-10 W absolute allowance for nearly-zero powers.
- Recorded cgroup peak within 24,000 MB; recorded swap usage zero.

The tool checks scalar outputs, not full high-order field equivalence. These
conditions are solver regression criteria, not mesh convergence or physical
validation. It does not automatically loosen any tolerances. Missing peak-memory
evidence is a rejection, not assumed success.

For a failed solve, inspect:

```bash
cat "$TRIAL/solution/solver_diagnostics.json"
tail -5 "$TRIAL/solution/ksp_history.jsonl"
cat "$TRIAL/launcher_status.json"
```

A decreasing monitored norm alone is insufficient; the final true residual and
comparison decide acceptance. A stopped/OOM-killed run may have only partial
files. Preserve its directory.

## A small experiment budget

Keep the mesh, ranks, ILU level and restart fixed. If scale 1 fails without a
memory kill, try scale 0.1 and then scale 10, each in a new directory, changing
only `HELICON_RF_PC_MASS_SCALE`. Stop after these three tests. If all fail,
this auxiliary operator/local solver is inadequate; increasing iteration limits
or loosening validation is not evidence of a fix. The next development step is
an auxiliary-space or two-level method, rather than more full-size trials.

If a candidate passes, repeat it once to check reproducibility, then test one
intermediate mesh resolution before returning to the original 2.87M-DOF mesh.
A pass on this coarse reference does not guarantee convergence at full size.

## Optional code regression checks

Portable checks (no FEM solves):

```bash
/usr/bin/python3 -m unittest discover -s tests -p test_rf_positive.py -v
```

Numerical checks on the installed PETSc build:

```bash
PETSC_DIR=/usr/lib/petscdir/petsc-complex HELICON_FEM_DIAGNOSTICS=1 \
  /usr/bin/python3 -m unittest discover -s tests -p test_rf_positive.py -v
mpiexec -n 2 env PETSC_DIR=/usr/lib/petscdir/petsc-complex HELICON_FEM_DIAGNOSTICS=1 \
  /usr/bin/python3 -m unittest discover -s tests -p test_rf_positive.py -v
```

The numerical tests exercise degree-1/2 known-field systems with a separate P,
positive quadratic energy, complex-form compilation and independent residual
checks. They do not establish performance on the supplied RF case.
