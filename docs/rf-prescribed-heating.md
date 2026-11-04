# Coarse RF to prescribed 2D plasma heating

This one-way experiment reuses an existing RF solution. It conservatively deposits
integrated tetrahedron powers into the axisymmetric transport grid, verifies the
original RF absorption before scaling, and explicitly normalizes the heating to
a user-selected **plasma absorbed power**. Uniform and RF-shaped heating are then
solved at the same power. No RF solve, voltage scaling, mesh generation or plasma
feedback is performed. NumPy and SciPy are sufficient with `--no-plots`;
matplotlib adds the field plots. DOLFINx/PETSc/MPI are not needed for this command.

## Linux command using the saved coarse AMS solve

From `~/Documents/HeliconThruster`, after transferring the Git changes:

```bash
python3 -m helicon.rf.prescribed \
  --rf-solution runs/rf-reference-20261027-121634/ams-main-20261102-164445/solution \
  --plasma-config config/plasma_spatial_coarse.json \
  --absorbed-power-W 100 \
  --allow-exploratory-interface \
  --out runs/coarse-prescribed-100W
```

Run Python directly, without `mpiexec` or `run.sh`. The output directory must be
new. Use `--no-plots` if matplotlib is unavailable. The 100 W value is a
prescribed-heating experiment, not a prediction of accepted antenna feed power.
The command does not alter the RF input directory. To freeze the experiment,
keep that directory intact: all input artifact hashes, full RF configuration,
original RF summary, resolved plasma configuration and source snapshot are saved
in the new run. No copy of the large RF mesh or cell exports is made.

`--allow-exploratory-interface` explicitly permits the small negative outward
flux already admitted by the current RF check (up to 1e-4 of input power). It
cannot admit failed RF checks, negative material/PML absorption beyond 1e-12 of
input power, or excessive interface trace disagreement. This option does not
certify a physical open boundary; the output records that limitation. It is
needed for the pasted November 2 AMS result. Omit it for strictly nonnegative
outward flux, subject to a 1e-12 roundoff allowance.

The source must contain `rf.json`, `rf.msh`, `mesh_report.json`,
`rf_summary.json`, `run.json`, and the complete `cells_rankNNNN.npz` set declared
by `mpi_ranks`. Artifact hashes must match the RF run record. Cell exports must
contain integrated `cell_absorbed_power_W` and `cell_volume_m3`; visualization
samples `q_abs_W_m3` are never substituted. Every transport bin must be occupied.
Power mismatch, missing ranks, changed inputs, geometry mismatch, or failed
transport conservation stop the experiment and retain diagnostics.

## Outputs and interpretation

- `report.html`: equal-power comparison, with field plots when enabled.
- `uniform/` and `rf-shaped/`: independently checked transport summaries and NPZ
  fields, with local ion/neutral/current/energy residuals and integrated balances.
- `heating.npz`: grid faces, original RF heating, and normalized prescribed
  `q_abs_W_m3`, also usable with `helicon.spatial --heating`.
- `transfer.json`: original integrated power, normalization factor, normalized
  power, occupied-bin fraction, and RF versus finite-volume chamber volumes.
- `experiment_summary.json`: volume-weighted means, extrema, exit ion flows,
  interface allowance and explicit one-way limitation.
- `run.json`: input hashes, stages, source snapshot, completion/failure and errors.

The RF distribution remains fixed at its original plasma state. The transport
response does not update its dielectric. Transport uses the configured uniform
axial B; an RF magnetic map is recorded but is not imported as nonuniform
transport. Azimuthal heating is summed into r-z bins by tetrahedron centroid.
Total power is conserved; subcell shape, azimuthal structure and mesh accuracy
are not certified. Faceted RF and ideal cylindrical transport volumes are
reported separately without a hidden volume correction.

## Local verification on November 4, 2026

The latest November 2 Linux AMS solution is not present on this Mac. The actual
saved RF solution used for local transfer/transport verification was
`runs/rf-compact-20261023-194156/mumps-ooc-4000-20261026-130445`: 1,485,118 RF
DOFs, four MPI export files, and 0.618913698 mW original plasma absorption.
This is an older reflecting-box diagnostic, not the latest 433,368-DOF solve.

On the supplied **2 radial by 4 axial** transport grid at 100 W, both uniform
and RF-shaped cases converged. Original RF-to-grid power mismatch was 1.17e-14,
all bins were occupied, and normalized power was 100 W within floating-point
roundoff. Maximum local residuals were below 1.1e-13 for both cases. RF-shaped
Te ranged from 3.270 to 6.612 eV. These are verification results for this coarse
experiment, not a validated plasma prediction.

The existing 4x8 grid failed for this same RF shape at 100, 50 and 10 W; the
100 and 50 W responses reached 7 eV. The coarser configuration is therefore
an executable workflow benchmark only. Its completion does not resolve the
transport-grid sensitivity, the heating-profile resolution, or the chemistry
limit. The default 4x8 configuration is unchanged. Run the same controlled
comparison with the latest Linux RF exports before interpreting its response.

A nonlinear solver termination is not acceptance. The 1–7 eV chemistry interval
and conservation checks remain enforced. If a run reaches 7 eV and fails, inspect
its saved fields/residuals; test a lower prescribed power in a new directory.
Do not extend the rate fits or weaken the checks merely to obtain completion.
Changing power changes the experiment and must be recorded explicitly.

Mesh and PML convergence, source calibration and transport closure validation
remain necessary before quantitative design conclusions or physical feedback.
