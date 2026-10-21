# HeliconThruster magnetostatics and global plasma model

## Start the global argon plasma model (Mac or Linux)

```sh
python3 -m helicon.plasma --out runs/plasma-baseline --no-plots
```

This solves steady particle, neutral and power balances for mean electron
density, electron temperature, neutral density, exit ion flow and power losses.
The solver needs only standard Python: **no MPI, FEniCS or meshing step**.
Open `runs/plasma-baseline/report.html`. For a budget figure, install
`requirements-plasma.txt` and omit `--no-plots`, using a new output folder.

The example is **500 W absorbed power and 1.5 mg/s argon**, with illustrative
boundary loss ratios. It does not predict magnetic confinement, RF absorption,
spatial profiles or thrust. See [the model specification](docs/plasma-model.md)
for equations, rate sources, assumptions, overrides and verification. Rate fits
are restricted to 1–7 eV; unsupported cases fail rather than extrapolate.

For why the present magnetic solve uses real scalar elements, and why the
future RF solve should use complex Nedelec elements, see
[electromagnetic formulations](docs/electromagnetic-formulations.md).

## Magnetostatic workflow

Parameterized axisymmetric Gmsh geometry, a serial FEniCSx magnetostatic solver,
repeatable run records, field reports, numerical sensitivity studies, and coil
parameter sweeps. The FEM workflow runs on Ubuntu. Meshing, study preparation,
and plotting saved results run on the Mac without MPI.

## Start here on Ubuntu

From this project folder, using the Python environment that ran your previous
FEniCSx solve:

```sh
python3 -m helicon.run --out runs/baseline --target-gauss 140
```

Open `runs/baseline/report.html` in a browser. It contains field magnitude maps,
magnetic field lines, axial and radial profiles, and a summary in gauss.
`--out` must be a **new directory**; previous results are never overwritten by
this workflow. For another run use another name, such as `runs/baseline-02`.

Your current configuration retains **100 turns and 18.98 A in both coils**.
The original Ubuntu solver run you supplied reported a chamber mean of 140.022 G
and an on-axis exit field of 133.16 G. That is evidence for the original solver;
the new sampling and reporting additions still require their first Ubuntu run.
Matching the mean by adjusting current is calibration, not independent validation.

The target option changes scoring/reporting; it **does not automatically tune
currents**. Currents and coil positions come from `config/geometry.json` or a
configuration supplied with `--config`.

## 1. Reproducible run records

Each run directory contains:

| File | Purpose |
|---|---|
| `run.json` | Status, full configuration, effective mesh sizes, coil ampere-turns/current densities, solver settings, code identity, stage commands, timestamps, hashes, results |
| `source/` | Snapshot of Python source and dependency recipes used for this run |
| `input.json`, `geometry.json` | Requested and meshed input snapshots |
| `mesh_report.json` | Physical tags, region areas, node/triangle counts |
| `mesh.log`, `solve.log`, `plot.log` | Complete stage output and errors |
| `solver_environment.json` | Actual Python, DOLFINx, PETSc, MPI and package versions |
| `fem_summary.json` | Metrics in SI and convenient gauss/percentage units |
| `fem_axis.csv` | Chamber centerline FEM and independent finite-pack reference |
| `fem_exit.csv` | Radius, Br, Bz and magnitude at the chamber exit |
| `field_device.npz`, `field_full.npz` | Sampled FEM potential and magnetic field on plotting grids |
| `report.html`, `report.md`, `*.png` | Human-readable results and plots |
| `field.xdmf`/`.h5`, `potential.xdmf`/`.h5` | ParaView visualization |

Every coil record includes N, I, N I, its center, mean radius, winding dimensions,
and uniform current density `N I / (radial_thickness * axial_width)`.
Software versions and source snapshots aid reproducibility; dependency recipes
are not exact cross-platform binary lockfiles.

A failed stage marks the run **failed**, retaining its diagnostic log and inputs.
A failed or incomplete run is never included in a successful design ranking.

The old commands still work, but their `build/` directory is a replaceable scratch
workspace. Use `helicon.run` for preserved experiments:

```sh
python3 -m helicon.mesh
python3 -m helicon.solve --build build --target-gauss 140
python3 -m helicon.plot --build build
```

Replotting requires mesh/configuration hashes to match the solve. No MPI is
needed to plot the saved CSV/NPZ results, so a full run folder can be copied back
to the Mac for inspection. The HTML uses local image files: keep the folder intact.

## 2. Mesh and outer-domain sensitivity

```sh
python3 -m helicon.study convergence \
  --out runs/convergence \
  --mesh-scales 1 0.5 0.25 \
  --domain-factors 1 1.5 2 \
  --target-gauss 140 --tolerance-percent 1
```

This runs six serial cases:

1. Three mesh sizes on the original domain. A factor of 0.5 halves **all** target
   element lengths, including transition lengths.
2. Three outer-domain sizes using the finest requested mesh scale (0.25 here).
   The outer radius and distances of the axial bounds from the chamber midpoint
   grow with the domain factor. Chamber, wall, coil and downstream geometry stay
   fixed, as do all mesh size settings during the domain study. The actual
   triangulation can change when the domain is remeshed.

It writes `comparison.csv`, `comparison.png`, `study_report.md`, and
`study_summary.json`. The report tracks chamber mean Bz, Bz standard deviation,
Br RMS, on-axis exit Bz, exit area-average Bz, uniformity, and reference error.

The convergence decision requires **all cases completed with the same source
snapshot and runtime environment**, both final successive pairs within tolerance, and the final enlarged
domain's centerline reference error within tolerance. Each scalar change is
relative to its value in the finer/larger case, with a denominator floor of
`1e-6 * target field` to handle nearly zero quantities. Absolute changes as a
percentage of the target are also reported. The centerline uses a relative
Euclidean norm on the same 121 sample locations.

Passing means the **tested sensitivity criteria** are met. It is not a proof of
experimental validity or accuracy everywhere. Review the complete trends: if
changes are irregular or near the threshold, add a finer mesh/larger domain.
If the mesh sequence fails, resolve that before accepting the domain sequence.
The exit-area Bz is calculated from flux, `2*A_theta(R,z_exit)/R`, rather than
integrating plot pixels. Exit mean magnitude uses 64-point radial quadrature.

## 3. Field plots and readable results

Every full run automatically creates:

- `profiles.png`: chamber centerline versus the independent Biot–Savart
  finite-winding-pack reference, plus the radial exit profile.
- `field_device.png`: field magnitude and field lines around the source and coils.
- `field_full.png`: field magnitude and field lines over the computational domain.

The field lines are contours of the axisymmetric flux function `psi = r*A_theta`.
They show paths; they do not show arrow direction. Colors show magnitude. The
full-domain plot helps reveal the effect of the artificial outer boundary.

The regular plotting grids are visualization samples only. Metrics and design
scores use FEM volume integrals and direct FEM point evaluations. Gradients are
obtained from the quadratic potential and represented exactly as DG1 within each
linear triangular cell. At interfaces the point sampler selects one incident
cell, so derivative traces can differ until the mesh is converged.

The chamber mean, axis exit value, and exit disk average are distinct quantities.
They are all printed in gauss, so checking the 130 G exit benchmark no longer
requires a separate shell calculation. `1 T = 10,000 G`.

## 4. Coil parameter studies

Two example specifications are provided. Each describes a Cartesian product
of parameter choices (nine runs per example).

```sh
# Independently vary each coil's ampere-turns.
python3 -m helicon.study sweep \
  --spec config/sweep_currents.json --out runs/current-sweep \
  --scale 0.5 --target-gauss 140 --tolerance-percent 1

# Independently move the two coil centers.
python3 -m helicon.study sweep \
  --spec config/sweep_positions.json --out runs/position-sweep \
  --scale 0.5 --target-gauss 140 --tolerance-percent 1
```

Use a mesh scale established by your sensitivity study for design decisions;
0.5 is just a convenient starting example. The same `parameters` list can mix
positions and ampere-turns. Supported fields:

- `z_center`: metres, absolute axial coordinate.
- `ampere_turns`: signed A-turn; mapped to current using the configured turn count.

Example:

```json
{
  "parameters": [
    {"coil": "coil_1", "field": "ampere_turns", "values": [1700, 1898, 2100]},
    {"coil": "coil_2", "field": "z_center", "values": [-0.02, 0.0, 0.02]}
  ]
}
```

Invalid geometries, such as overlapping coils, are recorded and excluded.
The default maximum is 64 cases; larger studies require an explicit `--max-cases`.
Each case currently generates its own mesh and FEM solution. Current-only sweeps
could later reuse unit-current basis fields, but this implementation uses full
solves for consistency and transparent run records.

The ranking minimizes the dimensionless score

```
sqrt( <(Bz - B_target)^2 + Br^2>_chamber ) / B_target
```

where the brackets use the axisymmetric volume weight `2*pi*r`. It penalizes
mean-field error, nonuniformity and radial field with equal weights. Ranking is
by this score; a separate flag says whether each case meets the requested
mean-field tolerance. A best-scoring case can still miss the target, so check
that flag. A low score says nothing about coil power, mass, heating, RF coupling,
plasma production or thrust. Those constraints and models need additional data.

## Prepare on Mac, execute on Ubuntu, continue interrupted studies

Prepare studies without loading MPI or generating their meshes:

```sh
python3 -m helicon.study convergence --out runs/prepared --prepare-only
python3 -m helicon.study sweep --spec config/sweep_currents.json \
  --out runs/prepared-sweep --prepare-only
```

Copy the project and the prepared study directory to Ubuntu, then:

```sh
python3 -m helicon.study execute runs/prepared
```

The plan stores complete configuration snapshots. It does not depend on later
edits to `config/geometry.json`. Completed cases with matching inputs and hashes
are preserved and skipped. Failed cases can be retried explicitly:

```sh
python3 -m helicon.study execute runs/prepared --retry-failed
```

Previous failed folders are archived under `attempts/`. A folder marked running
or partial is preserved; inspect whether its process is still active before
starting a new study. Recalculate a comparison without rerunning any solver:

```sh
python3 -m helicon.study summarize runs/prepared
```

Incomplete studies return a nonzero exit status. A fully completed study returns
zero even when its sensitivity criterion fails; read `sensitivity_criteria_met`
in the JSON/report. Do not equate process success with numerical convergence.

For a local Mac geometry-only run:

```sh
/opt/homebrew/bin/python3 -m helicon.run --out runs/mac-preview --mesh-only
```

FEM cases currently run sequentially on one MPI rank. Invoke Python directly.
Do not launch the study runner with `mpirun`: that would launch duplicate drivers.
The solver itself rejects multi-rank execution. Distributed solves and parallel
sweep scheduling are not implemented.

## Geometry, formulation, and provenance

All lengths in `config/geometry.json` are metres. Gmsh coordinates are
`(x,y,z) = (r, physical axial z, 0)`.

| Quantity | Value / source |
|---|---|
| Chamber radius / length | 30 / 120 mm; paper benchmark |
| Chamber axial interval | −120 to 0 mm |
| Coil mean radius / centers | 120 mm / z = −120, 0 mm; paper benchmark |
| Turns / currents | 100 / 18.98 A each; chosen to match the target, not measured winding data |
| Side-wall thickness | 3 mm; prototype assumption, no end plates |
| Pack radial × axial dimensions | 10 × 15 mm; prototype assumption |
| Outer domain | r = 0…600 mm, z = −720…600 mm; convergence parameter |
| Downstream observation region | r = 0…60 mm, z = 0…180 mm; diagnostic volume |

The published example sets the **volume-average axial source field to 140 G**
and reports an exit magnitude slightly above 130 G. The geometry dimensions
above are stated in its preliminary sizing/detailed design sections. Winding
cross-sections and turns remain our assumptions, so this is an idealized
magnetostatic benchmark inspired by that configuration, not an exact hardware
or full multiphysics reproduction.

Source: [Iannarelli et al., A novel simulation framework for Helicon plasma
thrusters design (2026)](https://link.springer.com/article/10.1007/s44205-026-00197-7).

OpenCASCADE fragments material regions into one conforming mesh. Physical IDs:
chamber 1, wall 2, downstream 3, air 4, coils 101 onward; axis 11 and far boundary
12. Coil IDs follow configuration order. All permeabilities equal mu0, with
`J_theta=N*I/pack_area` in each coil. No magnetic BC is imposed on chamber walls
or its inlet/exit.

The P2 scalar unknown is A_theta, with `Br=-A_theta,z` and
`Bz=A_theta,r + A_theta/r`. The weak form is

```
∫ (r/mu0) [A_z*v_z + (A_r+A/r)*(v_r+v/r)] dr dz = ∫ J_theta*v*r dr dz.
```

A_theta=0 on the axis and artificial far boundary. The on-axis field uses the
regular limit Bz=2*A_theta,r. The XDMF field is a cell-center visualization; the
potential XDMF interpolates P2 to P1 and is not a restart file.

## Installation and checks

Use the existing Ubuntu environment that ran the benchmark, or create one:

```sh
conda env create -f environment.yml
conda activate helicon
```

The MPI-free modules require `requirements.txt`; Homebrew Gmsh is detected on
the Mac when its Python module is outside site-packages.

```sh
python3 -m unittest discover -s tests -v
```

Tests cover geometry topology, independent loop/pack formulas, field
reconstruction and regular axis limits, exit weighting, objective definitions,
sweep plans, independent domain changes, failure preservation/retries, study
completion criteria, and rendering with explicitly synthetic field data.
These tests do not substitute for running the new FEM exports and convergence
study on Ubuntu. No RF or plasma model has been introduced.

## Spatial argon transport and RF feedback

A new axisymmetric quasineutral drift-diffusion benchmark resolves argon
particle transport, electron energy, current-continuity potential and sheath
losses. Start with `python3 -m helicon.spatial --out runs/spatial-argon-100W`.
It needs `requirements-spatial.txt`. Spatial feedback to the existing 3D RF
solver is available through `python3 -m helicon.rf.spatial_coupled`, requiring a
complex DOLFINx/PETSc runtime. See [the equations, commands and validation
limits](docs/spatial-plasma.md). The supplied grid is a numerical starter;
full spatial RF convergence has not been verified on the local real-valued
DOLFINx installation.
