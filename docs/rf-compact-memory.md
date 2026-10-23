# Reduced-size interface diagnostic

The Linux failure at MUMPS INFOG(1)=-13 happened during allocation for factorization. The machine reports 30 GiB RAM and 4 GiB swap. This config reduces mesh size; it does not guarantee that complex LU will fit, and is not a convergence result.

Use config/rf_nagoya_interface_compact_pec.json. It retains degree 2 and interface target 1.5 mm, but narrows the interface fine band to 1.5 mm and transition to 4.5 mm. Wall target is 2 mm, feed 1.5 mm, antenna 2 mm and bulk 70 mm. Antenna distance refinement now has optional antenna_band_m and antenna_transition_m (1 mm/5 mm here). Old configs retain the original 3 mm/14 mm distances. The existing minimum of antenna and gap targets still applies on antenna/feed surfaces.

This changes multiple mesh controls to reduce cost: it is a new diagnostic baseline, not a one-variable convergence comparison. Verify absorption and heating distribution before coupling. The PEC boundary model and all physical tolerances remain unchanged.

Run from the Linux project root after copying the updated files:

```bash
unset PYTHONPATH PETSC_ARCH
export PYTHONNOUSERSITE=1
export PETSC_DIR=/usr/lib/petscdir/petsc-complex
export HELICON_RF_PETSC_DIR="$PETSC_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
INPUT_ROOT="$PWD/runs/clean-20261022-121857"
RUN_ROOT="$PWD/runs/rf-compact-$(date +%Y%m%d-%H%M%S)"
/usr/bin/python3 -m helicon.rf.run \
 --config config/rf_nagoya_interface_compact_pec.json --case plasma --mesh-only \
 --plasma-summary "$INPUT_ROOT/global-plasma/plasma_summary.json" \
 --b-map "$INPUT_ROOT/magnetic/field_full.npz" --out "$RUN_ROOT/prepared"
```

After successful preparation, attempt the smaller solve with a saved log. Peak memory is still unverified; do not regard this as guaranteed to fit in 30 GiB. Keep the previous failed run intact.

```bash
set -o pipefail
mpiexec -n 4 /usr/bin/python3 -m helicon.rf.solve \
 --build "$RUN_ROOT/prepared/mesh" --out "$RUN_ROOT/solution" --expected-ranks 4 \
 2>&1 | tee "$RUN_ROOT/solve.log"
if test -f "$RUN_ROOT/solution/rf_summary.json"; then
 /usr/bin/python3 -m json.tool "$RUN_ROOT/solution/rf_summary.json"
else
 /usr/bin/python3 -m json.tool "$RUN_ROOT/solution/run.json"
fi
```

Increasing MPI ranks on the same machine does not increase physical RAM. Do not change passivity tolerances or treat this allocation failure as proof of bad physical signs.
