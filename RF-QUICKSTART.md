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
