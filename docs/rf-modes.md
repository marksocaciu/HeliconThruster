# Fourier-mode RF benchmark

`python3 -m helicon.rf_modes` solves a uniformly filled cylinder on a meridional
mesh. Each selected integer mode has three complex cylindrical field components.
The separate 3D reference uses prismatic edge elements. Both use sparse direct
solves in serial SciPy and the existing cold-plasma tensor. The inputs in
`config/rf_modes_uniform.json` are independent of the 3D antenna configuration.
The numerical example uses 27.12 MHz and the chamber dimensions and uniform
magnetized argon parameters of the RF starter configuration.

```sh
python3 -m pip install -r requirements-spatial.txt
python3 -m helicon.rf_modes --config config/rf_modes_uniform.json \
    --out runs/rf-modes --compare-3d --phi-deg 0 90
python3 -m helicon.rf_modes.slices --run runs/rf-modes \
    --out runs/rf-modes-slices --phi-deg 30 60 120
```

Run from the repository root and use a fresh output directory each time.
`run.json` records inputs, checks, source hashes and elapsed time. Exit code 4
indicates a numerical check failed. Partial runs retain diagnostic files.
The sign convention is exp(imφ − iωt), with peak amplitudes throughout.

## Excitation and boundaries

At the entrance, each complex amplitude A is specified in V/m as a real and
imaginary pair. For nonzero m the transverse drive has
Er = A (r/R)^(|m|−1) and Eφ = i sign(m) Er. The m=0 drive is azimuthal with
Eφ = A r/R. The default ±1 pair produces a 1 V/m x-directed entrance field.
Arbitrary specified mode weights represent this idealized end excitation.
A voltage across the Nagoya antenna gap would require a separate source model.

The far end is PEC. The cylindrical side imposes Ez = 0 as an essential condition
and Hz = 0 as a natural condition. This admits a uniform transverse wave and
zero normal time-averaged power on the side. The closed benchmark isolates bulk
dispersion, axis regularity and collisional heating. Antenna structure, exterior
radiation, a cylindrical PML and iterative plasma feedback are outside its scope.

## Discretization and verification

The meridional weak form integrates curl(E) · conj(curl(v)) − k₀² (εE) · conj(v)
with measure 2πr dr dz. Azimuthal derivatives act as im. On rectangular cells
Er and Ez use oriented edge components and rEφ is a continuous bilinear field.
For nonzero m the code constrains Ez and rEφ to zero at the axis and links
the first radial edge to the first off-axis angular node so that
∂r(rEφ) − imEr vanishes there. For m=0 it uses a quadratic radial angular
shape in the first cell to admit Eφ proportional to r. The saved coefficients
retain these shapes. B is curl(E)/(iω).

For m=±1, parallel propagation gives k² = (ω/c)² [1 − ωp² /
(ω(ω − mΩe + iν))], with signed Ωe = eBz/me. Reflection at the PEC end gives
Er(z) = A sin(k(L−s))/sin(kL), where s is distance from the entrance.
The code fits cos(k Δz) using the three-point recurrence on the computed axial
profile, compares the complex field and entrance power to the analytic values,
and saves the windowed axial spectrum. With nonzero collisions the real part
of the entrance admittance gives the absorbed power.

Power is evaluated from the constrained-DOF reaction, volume integration of
plasma heating, and an inlet Poynting integral with the raw one-sided curl.
The reaction and volume powers share the weak operator. The independent inlet
trace converges more slowly on coarse meshes. Inspect all three. The numerical
limits in run.json apply to this prototype benchmark.

`--compare-3d` solves the Cartesian vector equations on a disk approximated by
a polygon, extruded along z. It uses triangular Whitney and nodal functions,
assembles a separate full 3D matrix and applies the combined end field.
Comparison is available for the transversely uniform ±1 drives. It compares
complex E and B at prism centres and absorbed watts per cross-sectional area
to account for the polygon. Both solvers use the same axial spacing and
constitutive tensor. Agreement checks the modal reconstruction. Axial accuracy
is measured independently by the analytical solution. Other m values receive
the residual, passivity and power checks.

## Saved results

| File | Contents |
|---|---|
| `mode_+1.npz` (one per signed m) | Mesh edges, complex Er and Ez edge coefficients, rEφ vertex coefficients, E and B at cell centres, quadrature-integrated watts per cell. The ±1 files also contain analytic and computed axial profiles, angular wavenumbers and the Hann-windowed FFT. |
| `heating_rz.npz` | Azimuthally averaged cell heating `q_abs_W_m3`, annular volumes, integrated watts, and `r_faces_m`, `z_faces_m`. Shape is (nz,nr). This uses the existing `helicon.spatial --heating` format on matching geometry and grid. Power follows the specified drive. |
| `slice_*.npz` | Cylindrical E and B peak phasors and cycle-averaged heating at the selected angle and cell centres. Complex fields are summed before evaluating heating, retaining interference. |
| `reference_3d.npz` | Prism geometry, Cartesian E and B at prism centres and integrated watts for the optional 3D comparison. |

The azimuthal integral of cross terms vanishes for distinct integer modes in
this uniform axisymmetric medium. Summing modal cell powers yields the source
for axisymmetric transport. Local heating at a particular angle can differ.
Radial and axial resolution are set by the meridional mesh. Angular detail is
set by the retained modes. All arrays and angles can be reconstructed from the
recorded inputs and mode files.

Research precedent: Granetzny and Schmitz, *Computational and analytical
optimization of helicon antennas with a fast full wave solver exploiting
azimuthal Fourier decomposition*, [doi:10.1088/1361-6595/adffe7](https://doi.org/10.1088/1361-6595/adffe7).
The excitation and boundaries above define a separate verification problem.
