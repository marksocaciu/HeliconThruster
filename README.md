# HeliconThruster: first magnetostatic geometry

A parameterized **2D axisymmetric meridional mesh** for a cylindrical discharge
chamber, dielectric side wall, rectangular coil winding packs, surrounding air,
and a downstream observation region. The mesh generator and independent field
reference run on Apple Silicon without MPI. A separate FEniCSx starter solver is
provided for Ubuntu.

## Quick start on this Mac (no MPI)

The installed Homebrew Python and Gmsh were used for verification:

```sh
cd /Users/mark/Documents/HeliconThruster
/opt/homebrew/bin/python3 -m helicon.mesh
/opt/homebrew/bin/python3 -m helicon.preview
/opt/homebrew/bin/python3 -m helicon.reference
/opt/homebrew/bin/gmsh build/helicon.msh
```

For a portable, isolated geometry environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m helicon.mesh
python -m helicon.preview
python -m helicon.reference
python -m unittest discover -s tests -v
```

The Homebrew Gmsh Python API is discovered automatically when it is not on
Python's import path. This does not modify the installed environment.
`python -m helicon.mesh --gui` opens the generated mesh directly.
`geometry/geom.geo` is a convenience entry that opens `build/helicon.msh`;
the parameterized geometry source is `helicon/mesh.py`, not the `.geo` viewer.

## Geometry and assumptions

Edit `config/geometry.json`. All lengths are **metres**, current is **amperes**.
Gmsh stores `(x,y,z) = (r, axial_z, 0)`; its y coordinate is the physical axial
coordinate. Revolving this half-plane around r=0 describes the 3D device.
Do not interpret the displayed rectangles as planar current bars.

| Parameter | Initial value | Provenance |
|---|---:|---|
| Chamber inner radius | 30 mm | Existing project configuration/shared chat |
| Chamber length | 120 mm | Existing project configuration/shared chat |
| Chamber axial interval | −120 to 0 mm | Existing project configuration/shared chat |
| Coil mean radius | 120 mm | Existing project configuration/shared chat |
| Coil centers | −120 and 0 mm | Existing project configuration/shared chat |
| Turns / current per coil | 100 / 1 A | Existing placeholders; not published winding data |
| Dielectric side-wall thickness | 3 mm | Prototype assumption |
| Winding pack radial × axial size | 10 × 15 mm | Prototype assumption |
| Outer domain | r=0…600 mm, z=−720…600 mm | Prototype truncation; requires convergence study |
| Downstream observation region | r=0…60 mm, z=0…180 mm | Prototype diagnostic region |

The wall is an open-ended cylindrical **side wall**, without end plates,
backplate, feed system, or antenna. The downstream region is an observation
volume, not a solid nozzle. Every material initially uses μ=μ₀. No plasma
currents, permanent magnets, magnetic steel, RF fields, or thrust model are
included. These dimensions are a starting point, not an exact replication of
a published device; the source paper and winding details have not been
independently verified in this implementation.

The generator rejects non-finite/non-positive sizes, duplicate physical names,
overlapping regions, and regions outside the outer boundary. The JSON schema
also documents allowed fields. Geometry relationships are checked in Python.

## Mesh and physical groups

OpenCASCADE fragments the full domain so all interfaces share nodes. Every
surface belongs to exactly one material group. No magnetic boundary condition
is applied at the chamber wall, inlet, exit, or downstream observation interface.

| Physical name | Dimension | ID |
|---|---:|---:|
| PLASMA_CHAMBER | 2 | 1 |
| CHAMBER_WALL | 2 | 2 |
| DOWNSTREAM_REGION | 2 | 3 |
| SURROUNDING_AIR | 2 | 4 |
| COIL_1, COIL_2, … | 2 | 101, 102, … |
| AXIS | 1 | 11 |
| FAR_BOUNDARY | 1 | 12 |

Coil names come from the configuration and IDs follow list order. Mesh sizes
are 3 mm in the chamber, 1.5 mm in the wall, 2 mm in the coils, 5 mm downstream,
and 40 mm far away, with transition zones. The initial mesh has linear triangles;
the FEniCSx solution space is independently chosen as quadratic (P2).

```sh
python -m helicon.mesh --scale 0.5 --out build-fine
python -m helicon.preview --build build-fine
```

A smaller scale refines all target sizes. A custom configuration can be supplied
with `--config path/to/geometry.json`. Each build saves a configuration snapshot;
the solver reads that snapshot to avoid using currents or dimensions from a
subsequently edited configuration with an old mesh.

Generated files in `build/`:

- `helicon.msh`: Gmsh MSH 4.1 with physical tags, suitable for FEniCSx import.
- `helicon.brep`: CAD geometry only; BREP does not preserve physical tags.
- `helicon.geo`: viewer entry that loads the mesh.
- `geometry.json`, `mesh_report.json`: configuration, physical IDs, areas, counts.
- `mesh_preview.png`: actual mesh, colored by physical group.
- `reference_axis.csv`, `.png`, and `reference_summary.json`: independent field check.

## Immediate independent magnetic-field check

`helicon.reference` integrates the analytical circular-loop field over each
rectangular winding pack using Gauss–Legendre quadrature. The total ampere-turns
are N I, and the homogenized current density is `J_theta = N I / (width*thickness)`.
The sign of current sets the sign of the axial field.

For the checked-in 100-turn, 1 A configuration, the free-space reference gives:

- Chamber center Bz: **0.749227 mT**.
- Chamber exit Bz: **0.708176 mT**.
- Scaling both coil currents by approximately **18.6859** would give **14 mT
  at the center** under this idealized model. This is not a volume-average fit,
  not a verified published operating point, and not a coil thermal design.

The reference has no finite outer boundary. A FEM comparison must account for
both mesh error and outer-domain truncation. The reference output is clearly
labeled and is not presented as a FEM solution.

## FEniCSx starter on x86 Ubuntu

Keep the entire project together when copying it to Ubuntu. Create a separate
Conda environment using `environment.yml` (installation recipe, not a lockfile):

```sh
conda env create -f environment.yml
conda activate helicon
python -m helicon.mesh
python -m helicon.solve --build build
```

**Status:** the mesh and reference workflow were executed and tested on the Mac.
The FEniCSx code was syntax-checked and reviewed against the 0.10 API, but the
installed Mac MPI runtime aborts before importing DOLFINx. Therefore the FEM
solve and its exports remain **unverified until run on Ubuntu**.

The first solver deliberately runs on one MPI rank: invoke `python` directly.
FEniCSx still uses an MPI runtime in this mode. Parallel distributed point
sampling and solver configuration are a later extension; this initial code
rejects multi-rank execution explicitly rather than reporting incomplete
magnetic-field diagnostics. No local MPI or OpenMP settings are changed.

The unknown is A_theta(r,z), with

```
B_r = -∂A_theta/∂z
B_z = ∂A_theta/∂r + A_theta/r
```

The weak form, with the common 2π factor canceled, is

```
∫ (r/μ₀) [ A_z v_z + (A_r + A/r)(v_r + v/r) ] dr dz
    = ∫ J_theta v r dr dz.
```

`A_theta=0` is imposed on AXIS (regularity) and FAR_BOUNDARY (finite-domain
approximation). Numerical quadrature uses interior points with r>0. On-axis
sampling uses the regular limit `B_z(0,z)=2 ∂A_theta/∂r`, not division by zero.
Chamber metrics use the 3D volume weight `2πr`.

Expected solver outputs:

- `fem_summary.json`: chamber mean Bz, Bz standard deviation, Br RMS, and
  on-axis relative L2 error versus the independent finite-pack reference.
- `fem_axis.csv`: chamber on-axis FEM/reference samples.
- `field.xdmf`/`.h5`: cell-center `(B_r,B_z)` visualization in ParaView.
- `potential.xdmf`/`.h5`: P1 visualization of the P2 potential. This visualization
  does not retain all P2 degrees of freedom and is not a restart file.

For the first Ubuntu verification, run the baseline and `--scale 0.5` mesh,
then enlarge the outer domain in a separate configuration while keeping local
mesh sizes fixed. Compare chamber metrics and on-axis errors. Do not treat one
successful solve as mesh- or domain-independent validation. The mesh-size study
should eventually bring the relevant field metrics below 1% change, followed by
an independent domain-size study.

## Checks already run

Four automated tests cover the exact Helmholtz-pair center field, the thin-pack
limit and current reversal/superposition, invalid geometry, and the generated
mesh partition. Mesh checks verify region areas, unique coordinates, nonzero
triangle areas, one connected mesh, one physical group per surface, and exact
agreement between exterior mesh edges and the AXIS/FAR_BOUNDARY tags.

The default mesh contains **2,494 nodes and 4,811 triangles** with no duplicate
interface nodes or untagged exterior edges. These checks do not validate FEM
accuracy; that remains an Ubuntu task.

## References for implementation APIs

- [Gmsh reference manual](https://gmsh.info/doc/texinfo/gmsh.html)
- [DOLFINx 0.10 Gmsh importer](https://docs.fenicsproject.org/dolfinx/v0.10.0.post3/python/generated/dolfinx.io.gmsh.html)
- [DOLFINx 0.10 PETSc LinearProblem](https://docs.fenicsproject.org/dolfinx/v0.10.0.post3/python/generated/dolfinx.fem.petsc.html)
