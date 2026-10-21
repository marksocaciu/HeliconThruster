# Spatial argon benchmark and RF feedback

This adds a separate steady 2D axisymmetric, quasineutral fluid benchmark. The
existing 0D argon model and its coupling command remain available. This is a
first transport implementation, not a kinetic helicon/nozzle model or an
experimentally validated prediction.

## Run the prescribed-heating benchmark

From the repository root, in a Python environment with `requirements-spatial.txt`:

```sh
python3 -m helicon.spatial --out runs/spatial-argon-100W
```

The supplied case is **100 W absorbed by the plasma**, 1.5 mg/s argon, 400 K
neutrals, uniform 0.014 T axial field, and a deliberately small 4 radial × 8
axial cell grid. The 0D configuration remains 500 W. The spatial starter is
100 W because uniform heating at 500 W and this field can drive some cells
against the existing 7 eV chemistry limit. The solver never extrapolates those
rate fits; a constrained least-squares termination is NOT sufficient to pass.
Check `status` and all local/integrated conservation residuals. Failed physical
residual checks produce `not_converged` and CLI exit code 2.

Output: `spatial_summary.json`, `plasma_fields.npz`, `spatial_fields.png`, and
`report.html`, plus the geometry, source snapshot and artifact hashes in
`run.json`. Potential is relative to the first cell. Arrays have shape
`(nz, nr)`; positions and cell faces are in metres.

To prescribe nonuniform RF-cycle-averaged heating, supply `--heating path.npz`.
The archive must contain `q_abs_W_m3`, `r_faces_m`, and `z_faces_m` matching the
configured grid. Its integral sets actual absorbed power; the config's
`absorbed_power_W` is used only when no heating map is supplied. No rescaling of
a supplied map is performed. `--no-plots` avoids the plotting dependency.

## Equations and discretization

Unknowns per cell are `n = ni = ne`, neutral density `nn`, electron temperature
`Te` in eV, and electrostatic potential `phi` in volts. In the bulk:

```
div(Gamma_i) = S_iz
-div(Dn grad(nn)) = -S_iz
 div(Gamma_i - Gamma_e) = 0
 div(2.5 e Te Gamma_e + q_e) = Q_RF - P_inelastic + e Gamma_e . grad(phi)

Gamma_i = -mu_i n grad(phi) - mu_i Ti grad(n)
Gamma_e =  mu_e n grad(phi) - mu_e grad(n Te)
q_e     = -2.5 e n mu_e Te grad(Te)
S_iz    = n nn k_iz(Te)
```

`mu_e` is diagonal in the r/z plane: axial `e/(me nu)`, radial
`mu_parallel/(1 + (omega_ce/nu)^2)`. `nu = nn k_elastic(Te) + collision_floor`.
The floor is a prescribed effective momentum-relaxation assumption, not a
self-consistent turbulence model. Ion mobility and neutral diffusivity are
prescribed constants; a constant cold-ion temperature supplies the Einstein
ion diffusion coefficient. These assumptions need review against ion mean
free paths, collisionality, magnetization and chamber pressure before using
the results for design. Neutral diffusion may be inappropriate when Knudsen
numbers are large; a kinetic neutral model is then the appropriate upgrade.

The grid uses exact cylindrical annular volumes and face areas (`2 pi r`);
there is no singular cell at r=0. Ion/electron drift-diffusion fluxes use
Scharfetter–Gummel exponential fitting. Electron enthalpy is upwinded and heat
conduction is centred. Electron electrostatic work is split between the two
cells adjoining each face. Internal face terms cancel in integrated balances.
Neutral transport is conservative diffusion.

The axis has zero normal flux. Side and back surfaces lose equal ion/electron
particle flux `n cs`, with `cs = sqrt(e Te / mAr)`. A configurable fraction of
wall ions is returned to the adjacent neutral cell. The back wall also
receives uniform neutral injection; neutral gas escapes at the downstream end
with thermal effusion. The downstream plasma boundary uses equal sonic
ion/electron particle losses and electron enthalpy `2.5 e Te` per electron.

Floating, non-emitting dielectric wall patches use electron energy loss
`e Te [2 + 0.5 ln(mAr/(2 pi me))]` per lost electron. The sheath is an algebraic
boundary closure, not a resolved space-charge region. Its Bohm loss is
applied at the boundary cell value, so near-wall grid refinement matters.
The potential is determined by current continuity with zero boundary current;
a single redundant equation is replaced by `phi[0,0] = 0`. The omitted current
row is still checked independently after solving. There is no imposed common
metal wall voltage.

The power budget is specifically an **electron energy budget**:

```
Q_absorbed = P_inelastic + electron_boundary_power - electron_electrostatic_work
```

It includes excitation, ionization and elastic gas heating using the existing
1–7 eV Maxwellian argon fits. Electron electrostatic work is usually negative
in an ambipolar discharge: energy transfers toward ions. Ion acceleration,
ion kinetic energy transport, sonic exit kinetic power and total thrust
conversion efficiency are not solved here; do not describe the electron
budget as the total thruster energy budget.

A bounded nonlinear least-squares solve uses positive log densities, bounded
Te, and a potential gauge. Coloured local finite differences provide a dense
Jacobian for a robust small-problem trust-region solve. The implementation is
limited to 512 cells to avoid an accidental production-scale dense solve.
Every cell's ion, neutral, current and electron-energy residual is scaled by
its share of feed/power. Integrated mass and energy closure must pass too.
This is a steady branch calculation; it predicts neither ignition nor dynamic
stability. An unsuccessful run should lead to investigation of rate bounds,
transport, initial state and heating, rather than loosening conservation checks.

## Spatial RF coupling

Prepare an independent RF mesh without a magnetic-map override:

```sh
python3 -m helicon.rf.run --case plasma --mesh-only \
  --out runs/rf-spatial-prepared
```

In a **complex DOLFINx/PETSc** environment that also has SciPy:

```sh
python3 -m helicon.rf.solve --check-runtime
python3 -m helicon.rf.spatial_coupled \
  --build runs/rf-spatial-prepared/mesh \
  --out runs/rf-spatial-coupled \
  --feed-power-W 133.3333333333 \
  --max-iterations 20 --tolerance 0.01 --relaxation 0.4
```

133.33 W is an illustrative accepted-feed input, **not** a prediction of the
power needed to sustain the 100 W seed. Actual plasma absorption is solved by
the RF model and may lead outside the admissible chemistry interval. The
configuration's prescribed power supplies the initial plasma seed only.
Both models must use the same chamber origin/dimensions and uniform axial B.
An RF build carrying a saved B map is rejected: nonuniform tensor transport
has not been implemented in this first benchmark.

The driver retains the existing 3D antenna Maxwell solver. It does not replace
that solver with an m=0 field. Each RF material cell gets density and collision
frequency from the axisymmetric FV cell containing its centroid. Material
properties are piecewise constant; lookups cover the axis and boundary half
cells, and reject points outside the chamber. To run a single mapped RF solve:

```sh
python3 -m helicon.rf.solve --build runs/rf-spatial-prepared/mesh \
  --out runs/rf-spatial-single \
  --plasma-map runs/spatial-argon-100W/plasma_fields.npz
```

At each iteration a unit-voltage solve determines accepted port power. The
voltage is then scaled using P proportional to V squared and a second RF solve
verifies the requested feed power. The existing passivity, equation-residual,
material/PML absorption and PML-interface flux checks are mandatory. A failed
check stops the driver and preserves its stage and history.

RF exports now include quadrature-integrated `cell_absorbed_power_W` and
`cell_volume_m3`. Owned cells only are exported per MPI rank. Heating transfer
sums **plasma-region integrated watts** into r/z bins by tetrahedron centroid,
then divides by each FV volume. All azimuths contribute. PML and dielectric
losses never heat the plasma. The transferred total must agree with integrated
RF plasma absorption to 1e-6 relative. Point-sampled `q_abs_W_m3` remains for
visualization and is not used for conservative coupling. Old RF exports
without integrals are rejected. Every heating bin must contain an RF plasma
cell centroid; otherwise refine the RF mesh or coarsen the transport grid.

Centroid binning conserves power but is not a geometrically exact mesh-overlap
projection. The ideal cylindrical FV volume and faceted RF volume are reported
separately; power is not renormalized to hide their difference. Refine both
meshes to assess deposition-shape errors.

Convergence requires the unrelaxed full-field changes in ne, nn, Te,
collision frequency and gauge-independent potential, together with the RF
heating-map change, to satisfy the requested tolerance. Small relaxed updates
alone cannot pass. Every target plasma solve must close its local and global
balances. `final_rf_input_plasma.npz` records the exact RF material state;
`final_plasma` records its transport response. Their nonzero discrepancy is
reported honestly in the last iteration, rather than merging inconsistent
states into one field file. Iteration exhaustion returns an error and saves
`not_converged`. This is a numerical fixed point, not a stability test.

## Verification and its limits

```sh
python3 -m unittest discover -s tests -p 'test_spatial*.py' -v
python3 -m unittest discover -s tests -v
# In the complex runtime, including actual variable-material RF integration:
HELICON_RF_INTEGRATION=1 python3 -m unittest discover -s tests -p test_rf_integration.py -v
```

The spatial tests solve the PDEs and test nonuniform heating, local and
integrated conservation, failed-convergence rejection, variable-dielectric
passivity, bounded material lookup and power-preserving heating transfer.
Driver tests combine actual transport with explicitly synthetic RF power
exports; they are not evidence of a Maxwell solve. A separate test ensures
tiny relaxation cannot mask a nonconverged response. Compiler-only checks
cover the actual complex RF and integrated-cell diagnostic forms at both
Nedelec orders.

A 100 W test with the supplied transport assumptions gave exit ion rates
1.5380e18 /s on 4×8 and 1.7380e18 /s on 6×12: a roughly 13% increase despite
very small algebraic residuals. These are starter grids, **not a mesh-converged
solution**. Electron temperature ranges were 3.50–6.28 eV and 3.41–5.87 eV.
Further refinement, boundary-closure assessment and transport validation are
required. The model omits metastable kinetics, ion inertia, nonuniform magnetic
geometry, a kinetic EEDF, plume/nozzle physics and atmospheric mixtures.

The local installed DOLFINx build uses real arithmetic. Thus transport runs,
conservative-transfer tests and complex form generation can be verified here,
but a full spatial RF fixed point has not been executed or claimed. The
complex integration test remains mandatory before interpreting RF results.
Previously saved RF results that fail the existing power/PML checks must not
be reused or bypassed to obtain a coupled answer.

## References for the formulation

- [COMSOL: introduction to drift-diffusion theory](https://doc.comsol.com/6.3/doc/com.comsol.help.plasma/plasma_ug_drift_diffusion.07.17.html)
- [COMSOL: wall boundary condition](https://doc.comsol.com/6.3/doc/com.comsol.help.plasma/plasma_ug_drift_diffusion.07.22.html)
- [Ahedo and Navarro-Cavallé: 2D helicon fluid dynamics](https://ep2.uc3m.es/assets/docs/pubs/journal_publications/ahed13c.pdf)

These motivate the hierarchy and closures; this implementation does not claim
to reproduce either a commercial plasma solver or the cited helicon model.


## Parallel RF inside a serial transport controller

The spatial coupling command now accepts `--rf-ranks N` (default 1) and
`--mpi-launcher PATH` (default `mpiexec`). Launch the controller ONCE with
Python; do not place `mpiexec` in front of `helicon.rf.spatial_coupled`.
For N > 1, it starts a new MPI job for each unit-voltage and driven RF solve,
using the controller's exact Python executable and inherited complex-runtime
environment. The controller does not initialize MPI before launching these
jobs. Rank counts are checked in preflight and in each RF summary to catch
launcher/library mismatches. Rank output is retained in `*.mpi.log` files.
The same numerical power/passivity checks apply to serial and parallel RF.
Nonzero child exits, missing summaries and mismatched rank counts stop the run.

For the Ryzen 3600 (6 physical cores / 12 logical threads) and 32 GB RAM, start
with 4 RF ranks and compare wall time and memory with 6 ranks. This is a starting
configuration, not a measured optimum. Avoid running medium/fine/outer cases
concurrently until peak memory is known. MUMPS or SuperLU_DIST must be available
in the complex PETSc installation. MPI changes neither the equations nor the
required mesh resolution, and a failed serial RF power check remains a failure.

The spatial nonlinear solver is still serial at the algorithm level. Its dense
linear algebra can use threaded BLAS/LAPACK where the installed backend supports
it. Set thread limits BEFORE Python starts, for example:

```sh
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
python3 -m helicon.rf.spatial_coupled ... --rf-ranks 4
```

The MPI launcher explicitly sets BLAS/OpenMP limits to 1 for each RF child rank,
while leaving the controller's limits unchanged. This permits threaded transport
linear algebra without accidentally using 4 RF ranks times 4 BLAS threads.
Threading does not parallelize the Python chemistry/Jacobian assembly and does
not remove the 512-cell dense-solver limit. Test 1, 2, 4 and 6 BLAS threads on the
same transport case; small dense systems can be faster with fewer threads.

A genuinely distributed 2D transport solver remains future work: partition the
FV mesh and residual/Jacobian assembly, add halo exchanges and replace dense
bounded least-squares with a distributed sparse nonlinear solver. PETSc SNES is
a candidate; positivity, the 1–7 eV chemistry interval, gauge handling and the
conservation acceptance checks must survive that change. Numerical agreement
between 1, 2, 4 and 6 ranks is required before using it for larger meshes/server
runs. No transport MPI scaling or complete coupled MPI field solve has been
verified on the local real-valued runtime.

For the previously generated Linux medium/fine/outer configuration directory,
`tools/run_spatial_mpi.sh` supplies a Debian/Ubuntu launcher:

```sh
bash tools/run_spatial_mpi.sh runs/spatial-linux-20261021-184111 4 4 medium
```

Arguments are run root, RF ranks, transport BLAS threads, and case. It checks
the complex runtime, prepares/reuses a matching mesh, runs a vacuum MPI check,
and starts one spatial coupling controller. A failed vacuum check stops the
script. Existing completed outputs are retained in timestamped directories.
Set `HELICON_FEED_POWER_W` to override the illustrative 133.33 W accepted feed.
`HELICON_PYTHON` and `HELICON_MPIEXEC` select a compatible interpreter/launcher;
the default interpreter is `/usr/bin/python3`. This helper is specific to the
Debian/Ubuntu complex-path setup documented above. Threaded transport BLAS
still depends on the installed backend. The script is syntax-checked and its
MPI controller is covered by launch-contract tests; a full Linux parallel
field run is still required to establish performance and serial/MPI agreement.
