# Steady RF and global-plasma coupling

`python -m helicon.rf.coupled` iterates the frequency-domain RF solver with the
existing steady, volume-averaged argon model. The RF result supplies integrated
plasma absorption; that power becomes the input to the global model. Updated
uniform `ne`, `Te`, and `nn` are then used to rebuild the cold-plasma permittivity
for the next RF solve.

This is a scalar-power fixed point. It does not pass the RF heating map into a
spatial plasma transport model, alter the collision frequency, or compute a
matching network. A positive, numerically checked RF solve is mandatory at every
iteration; the driver stops on failed power/residual checks rather than feeding
invalid power into the plasma model.

## Prepare a plasma RF mesh

Run mesh preparation once, using the already-saved global plasma example and
magnetic field if available:

```bash
python3 -m helicon.rf.run --out runs/rf-plasma --case plasma --mesh-only \
  --plasma-summary runs/plasma-global-example/plasma_summary.json \
  --b-map runs/convergence/domain_02/field_full.npz --mesh-scale 2
```

Omit `--b-map` to use the configured uniform axial field. The preparation must
finish with `material_case` set to `plasma`.

## Run the coupled fixed point on complex Linux DOLFINx

```bash
python3 -m helicon.rf.coupled \
  --build runs/rf-plasma/mesh \
  --out runs/rf-coupled-1062W \
  --b-map runs/rf-plasma/background_B.npz \
  --feed-power-W 1062.5 \
  --max-iterations 12 --tolerance 0.01 --relaxation 0.5
```

`--feed-power-W` means accepted RF power at the ideal gap port. It is an
explicit required input. The paper's 1250 W RF-system input and assumed 85%
transmission efficiency give 1062.5 W as one possible delivered-power starting
value, not a measured gap-port load or target impedance. Use another value when
your definition of feed power differs.

On Debian/Ubuntu, RF commands automatically select
`/usr/lib/petscdir/petsc-complex` when it is installed. To select another
complex PETSc installation, set `HELICON_RF_PETSC_DIR` for that RF command.
This selection is local to the RF runtime and does not change the
magnetostatic solver's real-valued environment. The Python environment must
still provide DOLFINx built against the selected complex PETSc installation.

At each plasma state, the driver first solves at the configured voltage, then
uses linear `P ~ V²` scaling to choose the voltage that delivers the requested
feed power. It runs the scaled case and passes the integrated `PLASMA`
absorption to the global model. State updates use log-space relaxation. It
requires both plasma-state change and absorbed-power change to be below the
requested relative tolerance on consecutive iterations.

The output includes `coupling_summary.json`, a per-iteration global state and
RF run, and `report.html` for the converged final RF field. A failed RF check
retains its iteration output and stops the coupling run. Earlier RF runs
exposed negative PML and interface-flux signs; the added checks reject such
results. Resolve the underlying power-sign issue and pass the vacuum,
dielectric, and plasma checks before using the coupled result.

Each RF result is checked for finite powers, positive accepted input,
nonnegative material/PML absorption and outward interface flux, total power
balance, free-equation residual, and agreement between PML absorption and the
interface flux. The coupling boundary also verifies the RF check set is
present, plasma absorption does not exceed accepted RF input, and each global
plasma state has finite positive densities, an in-range electron temperature,
and closed particle and power balances. `run.json` records the current stage,
iteration, completed-iteration count, and failed stage. Failed run artifacts
and `progress.json` retain the latest successful iteration history and error
stage for diagnosis. The driver records progress but does not resume a failed
run automatically. After correcting the cause, use a new output directory for
a clean attempt.

Even after convergence, interpret this as a closed loop only between the
selected lumped feed-power abstraction, the present cold-plasma RF model, and
the uniform 0D argon model. The existing assumptions on wall/exit losses,
argon kinetics, plasma uniformity, and collision frequency remain in force.

## Spatial alternative

The separate `helicon.rf.spatial_coupled` command replaces scalar feedback with
a 2D axisymmetric argon fluid solve and conservative RF heating-map transfer.
See [spatial-plasma.md](spatial-plasma.md) for equations, setup and the explicit
verification limits. The scalar command documented above retains its existing
behavior.
