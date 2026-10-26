# Gradual RF mesh sizing

New opt-in config: config/rf_nagoya_smooth_pec.json. It retains the 1.25 mm interface target, existing fine-band widths and all other target sizes from the reviewed 1.25 mm PEC case. The only new mesh parameter is max_size_gradient=0.8. This limits each distance field's target-size slope dh/dd before applying mesh scale. It is not a bound on actual edge lengths or adjacent tetrahedron size ratios.

For antenna and interface distance fields, the effective transition end is the larger of the configured transition distance and band + (bulk_target - fine_target)/max_size_gradient. This extends the antenna transition to 86.625 mm and interface transition to 87.4375 mm instead of 5 and 4.5 mm respectively. Constant fine bands remain unchanged. Gmsh still combines all region/distance fields by their minimum; its standard tetrahedral optimization remains enabled. No optional external mesh optimizer is required.

Without max_size_gradient, the original size fields remain unchanged. The physical geometry, boundary model, RF solver, and acceptance tolerances are not modified. This update does not overwrite solve.py or the user's exploratory tolerance choice.

All generated meshes now record minSICN minimum, first percentile, median, and fraction below 0.1 per region in mesh_report.json. Empty, nonfinite or nonpositive quality values stop generation. The 0.1 threshold is diagnostic, not a proof of electromagnetic accuracy. Actual effective transition distances are recorded too.

## Linux preparation

Copy the updated files first. Run as mark:

```bash
cd ~/Documents/HeliconThruster
unset PYTHONPATH PETSC_ARCH
export PYTHONNOUSERSITE=1
export PETSC_DIR=/usr/lib/petscdir/petsc-complex
export HELICON_RF_PETSC_DIR="$PETSC_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PREVIOUS="$PWD/runs/rf-interface-1p25mm-20261026-140957/solution"
RUN_ROOT="$PWD/runs/rf-smooth-$(date +%Y%m%d-%H%M%S)"
/usr/bin/python3 -m helicon.rf.run \
 --config config/rf_nagoya_smooth_pec.json --case plasma --mesh-only \
 --plasma-summary "$PREVIOUS/input_plasma_summary.json" \
 --b-map "$PREVIOUS/background_B.npz" --out "$RUN_ROOT/prepared"
/usr/bin/python3 -m json.tool "$RUN_ROOT/prepared/mesh/mesh_report.json"
```

Review cell counts and quality first. The locally generated smooth mesh has 451456 tetrahedra, more than the reviewed Linux 1.25 mm mesh. Memory and RF accuracy have NOT been validated on this new mesh. Smoother elements do not guarantee lower factorization memory or convergence.

## Optional RF solve after reviewing the mesh

```bash
export HELICON_RF_LU_SOLVER=mumps_ooc
export HELICON_RF_MUMPS_MEMORY_MB=8000
mkdir -p "$HOME/mumps-scratch"
export HELICON_RF_MUMPS_SCRATCH="$(mktemp -d "$HOME/mumps-scratch/rf-XXXXXXXX")"
free -h
df -h "$HELICON_RF_MUMPS_SCRATCH"
set -o pipefail
mpiexec -n 4 /usr/bin/python3 -u -m helicon.rf.solve \
 --build "$RUN_ROOT/prepared/mesh" --out "$RUN_ROOT/solution" --expected-ranks 4 \
 --linear-solver mumps_ooc --mumps-memory-mb 8000 \
 --mumps-scratch-dir "$HELICON_RF_MUMPS_SCRATCH" \
 2>&1 | tee "$RUN_ROOT/solve.log"
```

Run only one solve at a time. A fresh mesh and output directory are required. Compare absorption and its spatial distribution to the saved 1.25 mm result; do not interpret improved mesh quality alone as converged physics.
