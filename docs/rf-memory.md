# RF solves with a bounded memory budget

`run.sh` now defaults to two MPI ranks, one BLAS thread each, a **24,000 MB
whole-job cgroup limit**, no swap, and MUMPS out-of-core with 10,000 MB workspace
per rank. All MB here are decimal (1,000,000 bytes). The job limit includes mesh
preparation, MPI children and charged file cache, but not other applications or
the rest of the operating system. Leave enough system headroom for a 30 GB total
machine target; unrelated processes can still increase their own usage.

These controls prevent an unlimited solve; they cannot make an oversized
factorization fit or guarantee iterative convergence. A cap may terminate a job.
The original 2,870,484-DOF case estimated 26,017 MB of MUMPS OOC workspace alone,
so rejection by the default admission check is expected.

## Linux prerequisites

Use the existing complex DOLFINx/PETSc installation. Symbolic analysis additionally
needs `mpicc`, `pkg-config`, and the development headers/library metadata for that
same PETSc installation. `PETSC_DIR/lib/pkgconfig/PETSc.pc` (or `petsc.pc`) must
exist. `HELICON_RF_PETSC_DIR` selects the installation and `MPICC` selects its MPI
compiler. The helper checks PETSc version and integer/scalar sizes before use.

The default hard limit requires systemd user scopes with memory-controller
support on cgroup v2. If unavailable, the launcher fails. `MEMORY_SCOPE=0` is an
explicit opt-out: it prints a warning and provides **no hard RAM limit**.

## Reuse a prepared mesh

Run from the repository. Replace the example path if the prepared case moved:

```bash
BUILD="$PWD/runs/rf-smooth-20261026-180559/prepared/mesh" \
  ANALYSIS_ONLY=1 ORDERING=2 ./run.sh
```

This assembles the matrix and performs **symbolic analysis only**, writes
`solution/solver_analysis.json`, then exits. Exit 0 means the estimates passed
the admission check; exit 3 means analysis completed but the estimates were
rejected. No field solution is produced in either case. Other nonzero exits
indicate an error. The check compares each rank's estimate to `MEM_MB` and adds
assembled rank RSS, the total MUMPS estimate and `MEMORY_RESERVE_MB` (default 2000)
against `JOB_MEMORY_MB`. A 1 MB allowance is added to each rank estimate to
cover integer-MB truncation (including zero estimates on tiny tests). RSS sums can double-count shared pages; the gate is
conservative and estimates are not guarantees.

Compare orderings 0 (AMD), 2 (AMF), 4 (PORD), or 7 (automatic), and rank counts
1, 2, 4. Record **effective_ordering**, not merely the requested setting. Scotch
(3) and METIS (5) require support in the linked MUMPS library. Rejected candidates
should not be retried by blindly increasing `MEM_MB`.

To attempt a direct solve after finding a viable candidate:

```bash
BUILD="$PWD/runs/rf-smooth-20261026-180559/prepared/mesh" \
  ORDERING=2 RANKS=2 MEM_MB=10000 ./run.sh
```

Every MUMPS solve launched by this script performs a fresh analysis/admission
check. The analysis helper is released before the ordinary solve; PETSc repeats
symbolic analysis during its own setup. This avoids relying on unsupported
petsc4py symbolic-factor APIs, at the cost of another analysis pass. Native
helper build output is retained in `solution/analysis-build.log`.

`BLR_TOL` is optional and experimental. Empty means disabled, even if a stale
`HELICON_RF_MUMPS_BLR_TOL` exists in the environment. BLR admission is supported
for `mumps_ooc`; compressed factors can remain resident, so compression does not
necessarily reduce total RAM. Compare residuals, impedance and powers to an
exact smaller reference.

## Iterative alternative

```bash
BUILD="$PWD/runs/rf-smooth-20261026-180559/prepared/mesh" \
  SOLVER=gmres_asm RANKS=4 ./run.sh
```

GMRES uses restart 30, relative tolerance 1e-9, up to 1000 iterations, additive
Schwarz with overlap 1 and local ILU(0). Optional environment controls:

- `HELICON_RF_GMRES_RESTART`
- `HELICON_RF_KSP_MAX_IT`
- `HELICON_RF_ASM_OVERLAP`
- `HELICON_RF_ILU_LEVELS`

There is no symbolic memory predictor for ASM/ILU or SuperLU_DIST; the cgroup
limit still applies. Increasing restart, overlap or ILU fill increases memory.
No automatic fallback silently changes the solver. Nonconvergence is an error.
The explicit constrained-system residual must pass the configured RF residual
tolerance, and the existing reaction, power and interface checks remain active.

ASM/ILU is an experimental baseline for this indefinite anisotropic Maxwell
operator. A shifted auxiliary-space FGMRES/AMS preconditioner is **not implemented**;
it needs compatible complex/higher-order support and separate numerical validation.

## Outputs and validation

- `solve.log`: MPI output and analysis estimates.
- `solution/solver_analysis.json`: requested/effective ordering, rank and total
  estimates and admission decision (MUMPS gated runs).
- `solution/run.json`: stage RSS sums, cgroup counters, status and solver residual.
- `solution/rf_summary.json`: completed field diagnostics, including iterations
  and true algebraic residual; absent for analysis-only or failed solves.
- `launcher_status.json`: exit code and final cgroup counters, including peak
  memory where supported. It may be absent if the entire scope is killed.

The reaction is assembled as `action(a, E)` with reverse ghost accumulation,
instead of assembling a second unconstrained matrix. The solution is copied to
an independently owned Function and LinearProblem is released before reactions
and exports, releasing its matrix and factors. Native helper source is included
in source snapshots.

Run portable checks:

```bash
python3 -m unittest discover -s tests -p 'test_rf*.py' -v
bash -n run.sh
```

Run numerical regressions in the selected PETSc environment:

```bash
HELICON_FEM_DIAGNOSTICS=1 python3 -m unittest discover -s tests -p test_rf_memory.py -v
mpiexec -n 2 env HELICON_FEM_DIAGNOSTICS=1 python3 -m unittest discover -s tests -p test_rf_memory.py -v
```

Tests cover degree-1/2 reaction-vector equivalence, a known-solution iterative
solve, option lifetime, native symbolic-only analysis, budget rejection, and
launcher mesh reuse/error propagation. Local real-PETSc testing is not a full
complex-plasma solve or proof of the target case's memory/convergence behavior.
