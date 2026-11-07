# Gradual prescribed heating redistribution

If a finer grid fails at the uniform alpha=0 step, use the optional
[accepted coarse-state initialization](spatial-grid-initialization.md) to test
whether a more informed initial guess resolves the nonlinear failure.

If fine transport grids have empty centroid bins, use the optional
[verified mesh mapping and sampled-volume transfer](rf-volume-transfer.md).
It reuses the saved RF solution and checks transfer sampling before solving.

For the same experiment with a documented rate backend extending above 7 eV,
see [selectable argon chemistry](argon-chemistry.md).

After a candidate is rejected, use the [saved-state budget diagnostic](spatial-budget.md)
to compare it with the preceding accepted state without repeating the solves.

Use continuation to test whether changing heating gradually helps the nonlinear
transport solver reach the RF-shaped endpoint. At each step, solve a steady
plasma response to

    q_alpha = (1-alpha) q_uniform + alpha q_RF.

Both endpoint maps integrate to the same prescribed power. Every intermediate
map therefore has that power too. Alpha is a numerical parameter, not time,
RF voltage, or a plasma feedback fraction. At alpha=1 the heating is exactly the
normalized original RF-shaped map, with no residual uniform contribution.

After transferring the project changes to Linux:

```bash
cd ~/Documents/HeliconThruster
python3 -m helicon.rf.continuation \
  --rf-solution runs/rf-reference-20261027-121634/ams-main-20261102-164445/solution \
  --plasma-config config/plasma_spatial_4x4.json \
  --absorbed-power-W 100 \
  --steps 10 \
  --allow-exploratory-interface \
  --out runs/coarse-continuation-4x4-100W
```

The 4x4 config is the existing configuration used for the failed direct
prescribed-heating run. `--steps 10` means 11 steady solves at alpha=0, 0.1,
..., 1. Each accepted response initializes the next step. The same RF exports,
transport grid, total power, chemistry interval and residual thresholds apply
throughout. The original failed run is preserved. This uses NumPy/SciPy and
optionally matplotlib; no new Maxwell solve or MPI launch is needed.
Use a new output directory and `--no-plots` if matplotlib is unavailable.

Every candidate is saved in `steps/step-NNN/`, including its heating, density,
temperature, potential, conservation diagnostics and optional plots. A compact
`continuation_summary.json` and `report.html` list alpha, acceptance, temperature
extrema, local residual maxima, radial power integrals and particle-flow metrics.
`last_accepted_alpha` and `last_accepted_directory` identify the last solution
that passed all transport checks. The command stops on the first rejected
candidate, retains it separately, and returns exit code 2. A failed candidate
never initializes another step. Unexpected solver exceptions retain the latest
progress and prior accepted state. Restart/resume is not implemented.

Read the console step lines or open `report.html`. To print a compact summary:

```bash
python3 - <<'PY'
import json
from pathlib import Path
s = json.loads(Path('runs/coarse-continuation-4x4-100W/continuation_summary.json').read_text())
print('Status:', s['status'], 'Last accepted alpha:', s['last_accepted_alpha'])
for step in s['history']:
    print(step['alpha'], step['accepted'], step['Te_max_eV'],
          step['max_local_relative_residual'])
PY
```

Completion at alpha=1 establishes a numerically accepted endpoint reached by
this path; it does not prove uniqueness, stability, grid convergence or physical
validity. A stop near 7 eV does not prove that a physical solution requires higher
temperature: initialization, other state bounds, nonlinear conditioning and
transport closure remain possible contributors. Examine the rejected residuals
and fields before choosing a smaller alpha increment or reviewing model inputs.

This is distinct from LES. LES represents effects of unresolved turbulent scales
on resolved flow using a subgrid treatment. The present model solves steady,
axisymmetric plasma transport without resolved turbulent dynamics or a subgrid
turbulence model. Coarse spatial averaging can hide sharp heating and plasma
structure, but continuation addresses solver initialization, not missing scales.
The rate-fit ceiling of 7 eV is a chemistry-data limit, independent of cell size.
