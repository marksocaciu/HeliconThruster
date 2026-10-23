# Interface refinement and coupling checkpoints

New opt-in configs preserve existing defaults:
- `config/rf_nagoya_interface_pec.json`: reflecting-box diagnostic, PML strength 1e-9.
- `config/rf_nagoya_interface.json`: original open-domain PML, NOT a PML fix.

Both use second-order fields, 6 mm plasma bulk, 1.5 mm wall/antenna targets and 1 mm feed target. A 1.5 mm target extends 3 mm from every plasma boundary surface on both sides, transitioning to bulk size at 9 mm. Gmsh takes the minimum of this field and existing region/antenna fields. Targets do not guarantee a specific number of layers. Geometry remains linear tetrahedra. Mesh scale multiplies element sizes, not band distances. Optional interface_m, interface_band_m and interface_transition_m must be supplied together.

Local Gmsh 4.14.1 generated 115,607 nodes and 656,748 tetrahedra at scale 1. Second-order direct solves may need substantially more memory. No Linux RF solution on this new mesh is claimed. A scale of 1.5 is a coarser resource-check option, not equivalent resolution.

## Linux baseline

Copy updated files to Linux, then run from the project root in Bash:

```bash
unset PYTHONPATH PETSC_ARCH
export PYTHONNOUSERSITE=1
export PETSC_DIR=/usr/lib/petscdir/petsc-complex
export HELICON_RF_PETSC_DIR="$PETSC_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
INPUT_ROOT="$PWD/runs/clean-20261022-121857"
RUN_ROOT="$PWD/runs/rf-interface-$(date +%Y%m%d-%H%M%S)"
mpiexec -n 4 /usr/bin/python3 -m helicon.rf.solve --check-runtime --expected-ranks 4
/usr/bin/python3 -m helicon.rf.run \
  --config config/rf_nagoya_interface_pec.json --case plasma --mesh-only \
  --plasma-summary "$INPUT_ROOT/global-plasma/plasma_summary.json" \
  --b-map "$INPUT_ROOT/magnetic/field_full.npz" \
  --out "$RUN_ROOT/prepared"
mpiexec -n 4 /usr/bin/python3 -m helicon.rf.solve \
  --build "$RUN_ROOT/prepared/mesh" --out "$RUN_ROOT/solution" --expected-ranks 4
/usr/bin/python3 -m json.tool "$RUN_ROOT/solution/rf_summary.json"
```

Check each command succeeds before proceeding. Do not start feedback on the strength of one passing mesh: compare controlled refinement and heating distributions. Copy the config and reduce interface_m for a second interface mesh, retaining other inputs; inspect actual resolution. Wall/feed refinement also needs independent checks. The old rf_convergence_sweep.py overrides wall/feed/bulk sizes with its defaults: use direct preparation above for these configs.

## Failure handling

Both global and spatial coupling drivers save last_accepted.json only after a complete RF/plasma iteration passes checks. It identifies the RF input and the unrelaxed plasma response separately. Global feedback saves its response summary and next relaxed state. Caught failures and keyboard interruptions save failure.json and coupling_summary.json with failed stage, exception, history and last accepted entry (null if none). Candidate output and MPI logs remain available.

Commands exit nonzero with concise errors. Iteration limits retain not_converged; keyboard interruptions record interrupted. Hard termination/OOM cannot be caught, but earlier atomic checkpoints remain. These are diagnostic checkpoints, NOT an automatic restart API. Accepted does not mean mesh-converged or fixed-point-converged. Preflight errors before output creation may only print an error.

No acceptance tolerances were relaxed; interface_trace_agreement is now explicitly required by coupling. Iteration builds verify source mesh/config hashes and reject geometry/domain/mesh changes without remeshing. The global driver remains serial; spatial feedback supports --rf-ranks but requires its separate uniform-B setup and cannot use the nonuniform B-map example above.
