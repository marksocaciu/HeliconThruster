# Local budget comparison from saved plasma states

The diagnostic re-evaluates the existing equations at saved states. It performs
no nonlinear transport solve and no RF solve. It automatically compares the last
accepted state before the first rejected continuation candidate.

After transferring the changes to Linux:

```bash
cd ~/Documents/HeliconThruster
python3 -m helicon.spatial_budget \
  --continuation runs/coarse-continuation-4x4-100W-step002 \
  --out runs/budget-4x4-alpha0220-0222
```

For the recorded 500-step experiment this selects alpha=0.220 and alpha=0.222.
The first is an accepted steady numerical state. The second is a rejected
iterate: its budgets diagnose the equation imbalance and do not represent a
physical plasma solution. Run once with Python, without MPI. NumPy and SciPy
are required; `--no-plots` avoids matplotlib. Use a new output directory.

The command reads the continuation history, resolved plasma configuration,
geometry and the two `plasma_fields.npz`/`spatial_summary.json` pairs. Their
hashes must match the saved continuation manifest. It checks grid dimensions,
field shapes, collision frequency, local residuals and integrated budgets
against the saved solver output. Unexpected differences stop the diagnostic
rather than attributing changed equations to the old solution. Inputs remain
unchanged, and the original/current source hashes are recorded separately.

## Energy signs and units

Every energy term is in watts per finite-volume cell, not W/m3:

    energy residual = ionization loss + excitation loss + elastic exchange
                    + conduction out + enthalpy out + boundary out
                    - RF heating - electrostatic work.

Internal conduction and enthalpy are signed net outward transfers; negative
values mean net energy entering a cell. Electrostatic work is a signed source
in the existing electron equation. Positive energy residual means modeled
loss/net outflow exceeds the source. A negative residual means excess source.
This sign convention exposes the current equation; it does not independently
validate its physical closure. The solver's existing `inelastic_power_W`
includes elastic energy exchange; diagnostics split the three contributions.

Internal face fluxes are stored once in left-to-right orientation. Their cell
contributions are equal and opposite and must cancel globally. The diagnostic
verifies that cancellation and reconstruction of all four local residuals.
Particle residuals are particles/s. The current-continuity residual is ion minus
electron particle-flux divergence, not amperes. Relative residuals use the
same fixed per-cell feed/power scales as the solver, before its redundant
current row is replaced by the potential gauge equation.

## Outputs

- `report.html`: hottest-cell locations, a same-cell energy comparison and maps.
- `comparison_summary.json`: global terms, reconstruction/cancellation errors,
  three hottest cells and the worst cell for each conservation equation.
- `accepted/cells.csv`, `rejected/cells.csv`: every cell's r/z coordinates,
  temperature, density, potential, solver-bound margins, energy and particle
  terms, and signed dimensional/normalized residuals. Indices are zero-based;
  radial columns run outward from the axis.
- `cell_changes.csv`: rejected minus accepted values at each corresponding cell.
- Each case's `cell_budget.npz` and `face_budget.npz`: signed cell arrays and
  internal/boundary face transfers. Cell arrays are flattened in z-major,
  r-minor order; face axis 0 is radial and axis 1 is axial. Boundary kind 0 is
  side wall, 1 is back, 2 is exit. `budget_summary.json` carries the checks.
- `run.json`, `source/`: input provenance, diagnostic source and failure status.

A temperature within 0.01 eV of either chemistry limit is flagged for reporting;
this flag does not alter the solver bounds or numerical acceptance tolerance.
Density-bound margins are in log space relative to the solver's 1e18/1e19 m^-3
scales; the potential bound is +/-500 V in the solver's gauge.

The console prints limiting coordinates, worst residuals by equation and the
energy terms at the rejected state's hottest cell, evaluated for both states.
Use those results to distinguish local heating/loss trends from current or
particle-balance problems before changing chemistry data or transport inputs.
No chemistry fit, collision model, transport coefficient or boundary condition
is changed by this diagnostic.
