# Magnetostatics versus the future RF solver

The current coil calculation and the future antenna-wave calculation solve
different equations. A complex PETSc build is not needed for the global plasma
balances, which do not use PETSc at all.

| Stage | Unknown | Discretization | Arithmetic |
|---|---|---|---|
| Existing axisymmetric magnetostatics | Azimuthal potential A_theta(r,z) | Scalar P2 Lagrange with cylindrical weights and axis regularity | Real |
| Current global plasma model | Mean ne, nn and Te | Nonlinear algebraic balances | Real |
| Future frequency-domain vector RF model | Complex electric-field phasor E | H(curl)-conforming Nedelec elements in 3D, or a properly derived azimuthal-mode reduction | Complex |

## Why scalar elements are appropriate in the present magnetic solve

Axisymmetry and purely azimuthal coil current permit A=A_theta(r,z) e_theta.
The vector curl-curl problem reduces to the existing scalar weak form:

    integral r/mu [ A_z v_z + (A_r + A/r)(v_r + v/r) ] dr dz
        = integral r J_theta v dr dz

The field components are Br=-dA/dz and Bz=dA/dr+A/r; A=0 on the axis enforces
the regularity condition. This is not a generic nodal discretization of the
full vector Maxwell problem. The scalar symmetry reduction is why Lagrange
elements work here. Changing element family alone would not be meaningful.
In a general 3D vector-potential magnetostatic solver, an H(curl) formulation
with appropriate gauge/nullspace treatment would be a different valid approach.

## What the RF implementation will need

With the exp(-i omega t) convention, a driven electric-field formulation is

    curl(mu^-1 curl(E)) - omega^2 epsilon(omega,B,ne,nn,Te) E
        = i omega J_antenna

Here epsilon includes epsilon0 and the complex magnetized-plasma dielectric
tensor. Conductivity/collisions must not be included twice. The magnetic-field
direction rotates that tensor into the computational coordinates. Frequency,
collisions, boundary conditions, antenna excitation and absorbed-power
normalization must use the same phasor convention.

For a 3D solve, use Nedelec H(curl) elements (e.g. Basix N1curl), because the
electric field has tangential continuity across material interfaces while its
normal component may jump. A reduced r-z Fourier-mode formulation needs its
own mixed spaces, azimuthal terms and axis conditions; it cannot be obtained
by treating all three cylindrical components as an ordinary planar vector.

The practical FEniCSx implementation should use a compatible complex-scalar
DOLFINx/PETSc/petsc4py environment on Linux. It is not sufficient to cast a
coefficient to complex inside a real-scalar PETSc build. Use conjugated test
functions (`ufl.inner`) consistently. MPI parallelism and complex arithmetic
are separate choices; complex problems can run on one process too.

Check the active Linux PETSc environment with:

```bash
python3 -c 'from petsc4py import PETSc; print(PETSc.ScalarType)'
```

For RF, this should report a complex type. The existing magnetostatic code has
been exercised in the user's real environment; it is not being certified for
a complex build by this plasma addition. Keep the working environment and
use a separate complex environment when developing the RF solver.

The official [DOLFINx 0.11 electromagnetic scattering demo](https://docs.fenicsproject.org/dolfinx/v0.11.0/python/demos/demo_pml.html)
uses complex arithmetic and Nedelec elements for time-harmonic Maxwell equations.
It is a useful numerical starting point, but its material model is not a
helicon plasma model. Boundary/PML reflection, power conservation, dispersion
and interface transmission will need verification before plasma coupling.

This change implements only the 0D plasma model. It does not implement the RF
equation above, antenna loading, the plasma dielectric tensor or a PML.
