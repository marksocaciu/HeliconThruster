# Independent frequency-domain RF prototype

The RF code lives in `helicon/rf/`. Existing magnetic and global-plasma source
files and input configurations are untouched. Runs go to new folders and never
replace an existing folder. This is an initial RF implementation, not a paper
reproduction or a validated thruster simulation.

## First run on the Mac: antenna and mesh only

Use a Python environment with NumPy, Gmsh and (for plots) matplotlib:

```bash
python3 -m pip install -r requirements-rf.txt
python3 -m helicon.rf.run --out runs/rf-vacuum --case vacuum --mesh-only --mesh-scale 2
python3 -m helicon.rf.preview --build runs/rf-vacuum/mesh
```

Open `runs/rf-vacuum/mesh/antenna_mesh.png` to inspect the actual triangulated
antenna. Open `rf.geo` in Gmsh to inspect the entire mesh and physical groups.
The coarse factor 2 is a workflow smoke test, not a converged RF mesh.

## Solve on Linux in the complex environment

Keep the working real-valued FEniCS environment for the old magnetic solver.
Activate a separate complex DOLFINx/PETSc environment. The Python packages alone
in `requirements-rf.txt` do NOT install that compiled stack. The implementation
targets the DOLFINx 0.10/0.11 APIs; check the actual environment before running:

```bash
python3 -m helicon.rf.solve --check-runtime
python3 -m helicon.rf.solve --build runs/rf-vacuum/mesh --out runs/rf-vacuum-solution
python3 -m helicon.rf.report --run runs/rf-vacuum-solution
```

The runtime check requires both PETSc and DOLFINx to use complex scalars. A real
build is rejected. Source/mesh records and failure details are saved for solves
that pass runtime/input preflight. Failed preflight prints an error and does not
create a run directory.

Once a serial solve works, use a parallel LU build (MUMPS or SuperLU_DIST):

```bash
mpiexec -n 4 python3 -m helicon.rf.solve \
  --build runs/rf-vacuum/mesh --out runs/rf-vacuum-mpi
```

Prepare meshes with one process. `helicon.rf.run` can prepare and solve serially
if `--mesh-only` is omitted. Do not launch the mesh-preparation command with MPI.
The parallel solve uses shared-facet ghosts, global reductions, and rank-local
cell exports. Serial-versus-MPI agreement still needs to be checked on Linux.

## Material cases

`--case vacuum` sets both the chamber and tube electromagnetic properties to
vacuum. The metal antenna/source and PML remain. This isolates the source and
open-domain numerics.

`--case dielectric` uses the configured lossy isotropic test medium in the
chamber, and the configured dielectric tube. It is an absorption-sign and power
accounting test, not a plasma approximation.

`--case plasma` uses a magnetized cold-electron dielectric in the chamber with
a stationary neutralizing ion background. Start by importing the working 0D
state and saved magnetic field:

```bash
python3 -m helicon.rf.run --out runs/rf-plasma --case plasma --mesh-only \
  --plasma-summary runs/plasma-global-example/plasma_summary.json \
  --b-map runs/convergence/domain_02/field_full.npz
python3 -m helicon.rf.solve --build runs/rf-plasma/mesh \
  --out runs/rf-plasma-solution --b-map runs/rf-plasma/background_B.npz
python3 -m helicon.rf.report --run runs/rf-plasma-solution
```

Use the paths that exist on your machine. The saved plasma result must be a
nonzero steady state with the same chamber radius/length. `ne`, `nn`, `Te` are
copied into the resolved RF config; density is assumed uniform. The B map is
interpolated in r,z and rotated into Cartesian components at plasma cell centers;
extrapolation is rejected. The solver automatically uses `background_B.npz` in
the preparation folder when present, so omitting `--b-map` does not silently
replace that map with a uniform field. It remains a sampled map, so map resolution and RF
cell refinement both matter. Without a map, the configured uniform B is used.

**Collision frequency remains an explicit assumed input.** The default 2e7 s^-1
is a numerical starter, not a measured or inferred value. The imported Te/nn
are recorded for provenance; they do not automatically change that rate. Supply
an independently justified effective electron momentum relaxation frequency.
Collision-frequency units are s^-1 (the JSON key ends in Hz); do not multiply
it by 2 pi. Only RF frequency is converted to omega=2 pi f.

The initial model has no downstream plasma; the exterior is vacuum. It omits
kinetic/Landau damping, warm-plasma corrections, sheath response, density-profile
feedback, metallic RF shielding by the DC coils, the paper's reflector, and
plasma response to the RF heating. The 0D loss ratios do not define an RF density
profile and are not reused as boundary densities here.

## Antenna and feed

The new 3D mesh contains two full end rings and two opposite axial rails around
the dielectric tube, a Nagoya-style topology. The example spans the 12 cm source,
with inner radius 38 mm, radial thickness 2 mm, 6 mm end-ring width, rails spanning
0.16 rad, and a 3 mm break at the middle of the +x rail. These dimensions and the
wall relative permittivity 3.8 are explicit starter assumptions, not a recovered
engineering drawing or a complete reproduction of the reference paper.

Metal volumes are excluded from the field mesh and their surfaces get a PEC
condition. Consequently there are no copper skin-depth cells or resistive copper
losses. The gap is a separately meshed vacuum volume. Its lateral internal
surfaces form an ideal voltage sheet: tangential Ez=Vpeak/gap, other tangential
components zero. Gap endcaps belong to the PEC antenna. This imposed field source
is a finite-gap approximation, not a coaxial wave port; the gap length and mesh
must be varied before treating its loading as reliable.

The prescribed voltage is a PEAK phasor, initially 1 V. The complex source current
is extracted as the generalized reaction to that voltage degree of freedom,
using the unconstrained finite-element matrix. `ideal_gap_impedance_ohm` is
therefore a loading diagnostic of this source model, NOT calibrated hardware
input impedance, cable impedance, or a prediction of S11/reflected power.

## Frequency-domain equations

Use E(x,t)=Re[Etilde(x) exp(-i omega t)]. The solution is the steady sinusoidal
response at a fixed frequency (27.12 MHz by default), with amplitude and phase;
there are no startup cycles, ignition or time-evolving plasma densities.

The physical-domain weak form is

    a(E,v) = integral curl(E) dot conj(curl(v))
                      - omega^2 mu0 eps0 (eps_r E) dot conj(v) dV.

The unknown uses first-kind Nedelec elements of degree 1 or 2; all field
coefficients and the PETSc system are complex. No vector nodal substitution is
made. Nonzero feed trace and homogeneous PEC/outer traces drive the problem.

The dielectric follows the linear electron momentum equation rather than a
hardcoded sign-sensitive Stix table. For q=-e and C(B)v=B cross v:

    M = (nu - i omega) I + (q/me) C(B)
    sigma = ne e^2/me * inverse(M)
    eps_r = I + i sigma/(eps0 omega).

This automatically rotates the tensor with B. In zero B it becomes the Drude
permittivity 1-wp^2/[omega(omega+i nu)]. The anti-Hermitian part is positive for
positive nu. Ions provide charge neutrality but no RF current in this first
high-frequency approximation. `Te=0` is not assumed by the source model; the
wave model neglects pressure/kinetic terms in the RF momentum response.

The Cartesian PML uses quadratic stretches si=1+i alpha (di/thickness)^2 and
transforms BOTH epsilon and mu. Outer E tangentially vanishes beyond the PML.
PML absorption is artificial outgoing-wave attenuation, not plasma heating.
Its performance, especially around this electrically small near-field source,
is unverified until domain/thickness/strength and mesh sensitivity are checked.

## Power accounting and output

    qabs = (omega eps0/2) Im[(eps_r E) dot conj(E)]
    Pinput = 1/2 Re[V conj(I)]

The Hermitian/anti-Hermitian distinction matters for a gyrotropic tensor; taking
only componentwise imaginary diagonal entries is not sufficient. The code
checks the free-equation residual, input power sign, material and PML
absorption signs, input versus material+PML power, and agreement between PML
absorption and outward Poynting flux on the physical-domain/PML interface.

The reaction power and volume power use the same weak operator: their agreement
is an algebraic consistency check, not independent proof of field accuracy.
Interface flux, feed-gap and PML refinement are needed in addition.

Outputs include `rf_summary.json`, immutable inputs/source/hashes, `fields.xdmf`
and HDF5 for ParaView, and per-rank `cells_rankNNNN.npz` with E/B peak phasors and
material absorption. XDMF and plots contain DG0 CELL SAMPLES, not the full
Nedelec solution or restart data. The central-slab plot must not be mistaken for
an exact plane section. `rf.report` runs without FEniCS and can be used on the Mac
with copied Linux results. No absorbed-power rescaling is applied silently.

## Verification and what remains to run

Portable tests verify the Drude limit, circular-mode response, B reversal,
rotation covariance, passive absorption, peak/RMS normalization, magnetic-map
interpolation, and mesh topology (including the INTERNAL feed and PML interface). An additional
compiler-only check generated complex C code for the actual degree-1/2 weak
forms, heating and interface-flux diagnostics. This does not execute PETSc.

```bash
python3 -m unittest discover -s tests -p 'test_rf.py'
```

The following opt-in test requires the complex Linux environment. It runs
vacuum, lossy dielectric and plasma solves and checks quadratic drive scaling:

```bash
HELICON_RF_INTEGRATION=1 python3 -m unittest discover -s tests -p 'test_rf_integration.py'
```

A skipped integration test does not mean the PDE solver passed. The first local
implementation was mesh/material tested on macOS; the complex DOLFINx PDE and
MPI execution still require these Linux checks.

After the smoke tests, halve local mesh sizes, refine the field map, and vary
PML/domain parameters. Track absorbed power, ideal-gap impedance, interface flux
and field profiles. Add an analytical wave/dispersion benchmark before using
plasma maps for design decisions. Free-space wavelength alone is insufficient
for meshing plasma waves and interface layers. Direct LU is a robust starting
choice but becomes memory intensive in 3D; this is not yet a scalable production
RF solver.

## Sources

- [DOLFINx 0.11 complex Maxwell/PML demo](https://docs.fenicsproject.org/dolfinx/v0.11.0/python/demos/demo_pml.html):
  Nedelec spaces, complex weak forms and coordinate-stretch tensors.
- [Iannarelli et al., 2026](https://doi.org/10.1007/s44205-026-00197-7):
  reference geometry/frequency and staged wave-plasma framework. Its full RF
  configuration, nonuniform plasma and reflector are not reproduced here.
- [Nagoya antenna sizing paper](https://doi.org/10.3390/aerospace11121056):
  antenna context; dimensional assumptions in this code remain explicit.
