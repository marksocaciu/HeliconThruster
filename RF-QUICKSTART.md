# RF module: start here

This addition leaves the magnetic and global-plasma solvers unchanged.
Full equations, assumptions, outputs and verification are in `docs/rf-model.md`.

## Prepare and inspect the antenna on Mac or Linux

```bash
python3 -m helicon.rf.run --out runs/rf-vacuum --case vacuum --mesh-only --mesh-scale 2
python3 -m helicon.rf.preview --build runs/rf-vacuum/mesh
```

Open `runs/rf-vacuum/mesh/antenna_mesh.png` or the `rf.geo` file in Gmsh.
The coarse mesh is for workflow checks, not design conclusions.
The preparation environment needs `requirements-rf.txt` dependencies.

## Solve in the Linux complex DOLFINx environment

```bash
python3 -m helicon.rf.solve --check-runtime
python3 -m helicon.rf.solve --build runs/rf-vacuum/mesh --out runs/rf-vacuum-solution
python3 -m helicon.rf.report --run runs/rf-vacuum-solution
```

Use a NEW output directory each time. Open its `report.html` or ParaView
`fields.xdmf`. The runtime must have complex DOLFINx and complex PETSc; a pip
install of requirements-rf.txt does not provide that compiled solver stack.
On Debian/Ubuntu the RF code selects `/usr/lib/petscdir/petsc-complex`
automatically when present. Set `HELICON_RF_PETSC_DIR` on an RF command to
select a different complex PETSc installation. This does not affect the
magnetostatic solver.

Run the opt-in integration check in that environment:

```bash
HELICON_RF_INTEGRATION=1 python3 -m unittest discover -s tests -p 'test_rf_integration.py'
```

Then progress to `--case dielectric`, then `--case plasma`, with saved plasma
and B-field inputs as documented in the full guide. MPI instructions are there
as well. Refine the mesh and vary PML/domain settings before interpreting power
or impedance. The ideal feed, PEC metal and assumed collision rate limit the
physical interpretation even if all numerical checks pass.

Local verification: 35 standard tests passed; an additional FFCx test generated
complex code for both element orders. The PDE/MPI integration test was not run
on the Mac because DOLFINx/PETSc are absent. No RF field results are claimed.
## Couple RF absorption back to the plasma state

The new `helicon.rf.coupled` driver iterates plasma RF absorption and the global
argon model at a fixed accepted power at the antenna gap. See
[`docs/rf-coupling.md`](docs/rf-coupling.md) for preparation and commands.
The existing saved vacuum solve has failed its passivity/power-sign checks, so
fix and revalidate those before expecting the coupled driver to proceed; it
stops on failed RF checks by design.

## Spatial plasma feedback

For the new 2D argon benchmark and conservative heating-map coupling, see
[`docs/spatial-plasma.md`](docs/spatial-plasma.md). The command is
`python3 -m helicon.rf.spatial_coupled`; the original `helicon.rf.coupled`
continues to use the uniform 0D model. Spatial coupling updates RF density and
collision frequency cell by cell, checks integrated power transfer, and tests
unrelaxed field convergence. It requires the complex runtime, a uniform axial
B field shared by both models, and a separate RF build without a saved B map.


### MPI during spatial feedback

Launch `python3 -m helicon.rf.spatial_coupled ... --rf-ranks 4` once. Its RF
children run under MPI; the controller and transport algorithm remain serial.
Do not launch multiple controllers with `mpiexec`. The optional
`--mpi-launcher /path/to/mpiexec` must match the Python MPI installation.
The controller may use threaded BLAS for transport; each RF child rank is
limited to one BLAS/OpenMP thread automatically. See the spatial-model document
for setup, rank-count checks and Ryzen 3600 starting recommendations.

With the Linux configs already generated, the provided launcher starts a
4-rank RF / 4-thread transport-BLAS run and first checks vacuum passivity:

```sh
bash tools/run_spatial_mpi.sh runs/spatial-linux-20261021-184111 4 4 medium
```


### Interface diagnostic correction (2026-10-22)

The clean Linux vacuum case still fails the RF checks. Its negative accepted
power is not fixed by this update. `outward_interface_flux_W` now uses the
physical-side trace at the physical/PML interface. The previous averaged trace
is retained as `averaged_interface_flux_W`, and the other side is reported as
`pml_side_interface_flux_W`. A new `interface_trace_agreement` check prevents
this diagnostic change from hiding an unresolved jump between the traces.
Existing tolerances and negative-power rejection remain enabled.

See `validation/rf-clean-interface-investigation.json` for the saved-field
analysis. Run the new regression in the complex Linux environment:

```sh
HELICON_FEM_DIAGNOSTICS=1 /usr/bin/python3 -m unittest discover -s tests -p 'test_rf_interface_flux.py' -v
mpiexec -n 2 env HELICON_FEM_DIAGNOSTICS=1 /usr/bin/python3 -m unittest discover -s tests -p 'test_rf_interface_flux.py' -v
```

Reusing the unchanged prepared mesh with the updated solver is supported; use
a NEW output directory. Expect the negative-power rejection to remain until
the underlying open-boundary/PML solution is corrected and converged. Do not
start coupled plasma feedback from this failed vacuum baseline.
