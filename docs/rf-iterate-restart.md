# RF iterate checkpoints

Iterative RF solvers save `OUT/iterate-checkpoints/latest.json` and `best.json`
plus rank-local vector files every 100 iterations and when the solve returns,
including iteration-limit or divergence returns. `--checkpoint-every N` changes
the interval. Nonfinite vectors never replace a checkpoint. Best means lowest
KSP-reported residual among saved iterates, not physical acceptance.

For example, keep the original build, maps, MPI rank count, and solver settings:

```sh
export HELICON_RF_KSP_MAX_IT=13000
mpiexec -n 6 python -m helicon.rf.solve \
  --build runs/rf-prepared/mesh --out runs/rf-resumed \
  --linear-solver fgmres_ams \
  --resume-iterate runs/rf-original/iterate-checkpoints/latest.json
```

Use the same B/plasma map arguments as the original run. To recover from
worsening residuals, select `best.json` instead. A saved iteration of 10000
leaves 3000 additional iterations with this total limit. History and diagnostics
report iterations in the current invocation; checkpoint metadata reports the
cumulative count. Use a new output directory to preserve previous diagnostics.

The exact assembled matrix, lifted RHS, and distributed row layout must match.
Changed mesh, coefficients, RHS, or MPI partition are rejected. Each vector is
checksummed. Copy the entire checkpoint directory when moving between machines.
Save publication is atomic and only follows successful writes on all ranks;
latest and best retain their own vector generations. One solve must own each
output directory. Hard kills can lose work since the last published save.

This is a warm restart from the solution vector. Krylov bases and preconditioners
are rebuilt, and relative stopping tolerance is measured from the new initial
residual. A restarted trajectory can therefore differ from an uninterrupted
solve. Existing power and residual acceptance checks remain required. No prior
run can be recovered if it did not save a vector. Direct factorizations are not
checkpointed. Requires validation in the target complex DOLFINx/PETSc MPI runtime.
