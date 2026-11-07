# Initializing a finer grid from an accepted coarse state

The optional continuation argument `--initial-state` loads a saved accepted
**alpha=0 uniform-heating** step and interpolates it to the target transport
grid. It supplies the initial guess for the target's first solve only. Later
continuation steps still initialize from their preceding accepted target-grid
response. Evaluation limits, chemistry, equations, heating and acceptance
thresholds remain unchanged.

## Linux retry

The earlier 4x16 continuation provides an accepted uniform solution in
`steps/step-000`. Its centroid transfer method is immaterial at alpha=0, where
the heating is uniform. Reuse that source and the mapping already produced by
the failed 4x32 run:

```bash
cd ~/Documents/HeliconThruster
python3 -m helicon.rf.continuation \
  --rf-solution runs/rf-reference-20261027-121634/ams-main-20261102-164445/solution \
  --plasma-config config/plasma_spatial_4x32_phelps.json \
  --absorbed-power-W 100 \
  --steps 40 \
  --heating-transfer sampled \
  --transfer-samples 512 \
  --transfer-mapping runs/coarse-continuation-4x32-100W-phelps-sampled/transfer/mesh_mapping.npz \
  --initial-state runs/coarse-continuation-4x16-100W-phelps/steps/step-000 \
  --allow-exploratory-interface \
  --out runs/coarse-continuation-4x32-100W-phelps-gridseed
```

No RF solve is performed. The source and failed target run remain intact; the
output directory must be new. The target retains `max_nfev=400` and its existing
conservation tolerances. Omit `--initial-state` to retain the generic initial
conditions used by previous runs. The optional mapping cache may also be
omitted to rebuild the verified geometry mapping.

## Source acceptance

The loader finds the source continuation record and verifies that the requested
step is marked accepted at alpha=0. It requires matching saved artifact hashes,
the same geometry and transport configuration apart from `nr`/`nz`, and actual
uniform heating at the same absorbed power. It re-evaluates the source's local
and integrated residuals using the budget diagnostic. A rejected state, changed
chemistry/data, stale file hash, incompatible physics or nonuniform heating is
rejected before the target solve. A source run that later failed at a larger
alpha can still supply its separately accepted alpha=0 state.

The original RF field need not determine a uniform initial state; all spatial
transport inputs and power must match. Solver evaluation limits and acceptance
tolerances are also compared, so the intended change is isolated to initial
conditions on the new grid.

## Interpolation

Cell-center ne, nn and Te are interpolated linearly in r,z after taking their
logarithms. Potential uses ordinary linear interpolation and is shifted so its
first target cell is zero. Since fine cell centers can lie between a physical
boundary and the first/last coarse center, values there are extended constantly
from the nearest coarse center. No unconstrained linear extrapolation outside
the coarse center range or physical domain is used. The physical domains must
coincide. Positive fields remain positive and interpolated fields must fit the
existing target solver bounds.

Collision frequency is recomputed from the interpolated temperature/neutral
density, target chemistry and unchanged collision floor. Source collision rates
are not interpolated independently. This is an initial guess, not a conservative
remap of an accepted target solution. The new solve must close all conservation
balances independently.

## Saved diagnostics

`initialization/interpolated_initial_state.npz` stores the actual initial fields,
new grid, uniform heating and recomputed collision frequency. Original source
fields, summary, resolved configuration, geometry and continuation record are
copied alongside it. `initialization.json` records hashes, grid sizes,
interpolation/edge treatment and target initial residuals. These residuals are
informational: an initial guess is not required to be balanced.

`continuation_summary.json` records `grid_initialization`. Its first history row
has `initialization=coarse_grid_seed`; later rows use `previous_alpha`. The
external seed is never recorded as an accepted target-grid state. Each history
row also records `solver_nfev`, so work to achieve acceptance can be compared
with the earlier unseeded run. Sources are checked again before finishing.

Failure at alpha=0 after seeding means initialization alone was insufficient;
retain its fields and residuals before changing solver strategy. Completion
establishes numerical acceptance on this grid, not uniqueness, stability, grid
convergence or physical validation.
