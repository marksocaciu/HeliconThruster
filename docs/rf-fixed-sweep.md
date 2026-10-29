# Controlled RF mass-scale comparison

Run on the coarse reference with the complex Linux runtime. After updating the repository:

```bash
cd ~/Documents/HeliconThruster
HELICON_RF_REPEAT_SOLVES=0 HELICON_RF_MASS_SCALES="5 15 10" \
  bash tools/sweep_rf_mass.sh "$PWD/runs/rf-reference-20261027-121634" 10
```

The final argument selects the original scale 10. The environment list adds scales
5, 15, and 10 in that same MPI process: four solves total. A and b are assembled
once; only P is reassembled, using a mutable mass coefficient. Each solve starts
from zero with a fresh KSP/ASM/ILU setup. Existing rank/thread settings and memory
cap apply. Do not supply multiple outer sweep scales for this diagnostic.

In the printed sweep directory, inspect `trial-1/solution/mass_sweep_diagnostics.json`.
`attempts` includes mass_scale, iterations, true_relative_residual and accepted.
`A_and_b_unchanged` must be true. Compare the first and final scale-10 attempts;
small roundoff differences are possible because P was reassembled. Reproducibility
does not establish convergence. The original solution is preserved; supplementary
trials produce diagnostics only, with no accepted fields or physics comparisons.
The launcher exit code and summary.csv continue to describe the original solve.
Thus exit 4 can accompany a successfully completed diagnostic sweep.

Each supplementary attempt has `mass-02`, `mass-03`, etc. with its own history.
The report records per-rank CSR structure/value and owned-RHS SHA-256 fingerprints,
and global matrix/RHS norms before and after the sweep. `partition_fingerprint.json`
records ordered owned input-cell IDs, local DOF ranges and cell-to-DOF maps.
Input hashes are already recorded in run.json. Compare these across separate launches:

- Input hashes differ: first resolve the input mismatch.
- Partition/maps differ: the distributed algebraic representations differ.
- Partition/maps match but matrix/RHS hashes differ: investigate assembly, coefficients,
  boundary handling and floating-point reproducibility.
- All hashes match but convergence differs: inspect effective solver/runtime settings.

Hashes are sensitive to ordering and basis representation; they do not prove physical
inequivalence. Fingerprinting uses local CSR arrays (potential temporary copies), not
a gathered global matrix. Keep initial investigations on the coarse reference.

Optional small PETSc regression, on one or two MPI ranks:

```bash
HELICON_FEM_DIAGNOSTICS=1 mpiexec -n 2 python3 -m unittest discover -s tests -p test_rf_fixed_sweep.py
```
