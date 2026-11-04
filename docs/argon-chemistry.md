# Selectable Maxwellian argon chemistry for spatial transport

The spatial solver accepts an optional `chemistry_model` key. Omission selects
`argon_ground_state_LL_1_to_7_eV_v1`, preserving the original fits and 1–7 eV
bounds. The added `argon_phelps_maxwellian_1_to_20_eV_v1` selects derived
Maxwellian rate tables from the three-process Phelps ground-state Ar cross
sections. The standalone 0D model retains its original backend and interval.

## Linux comparison

After transferring all changes, including `helicon/data/argon_phelps_maxwellian.json`:

```bash
cd ~/Documents/HeliconThruster
python3 -m helicon.rf.continuation \
  --rf-solution runs/rf-reference-20261027-121634/ams-main-20261102-164445/solution \
  --plasma-config config/plasma_spatial_4x4_phelps.json \
  --absorbed-power-W 100 \
  --steps 40 \
  --allow-exploratory-interface \
  --out runs/coarse-continuation-4x4-100W-phelps
```

The grid, RF field, power, ion mobility, neutral diffusion, magnetic field,
collision floor and acceptance tolerances match the earlier 4x4 experiment.
Only the selected electron collision/rate backend changes. This changes rates,
loss thresholds, elastic energy averaging and collision-dependent electron
mobility/conduction as well as the temperature bound. It is therefore not a
pure test of increasing a numerical bound. Original failed runs remain intact.
There is no RF solve, plasma feedback, kinetic solver or time integration here.

No new dependency is needed beyond the existing NumPy/SciPy environment.
Use `--no-plots` if matplotlib is unavailable. Inspect the console step lines
and `continuation_summary.json`; each saved step summary contains the backend,
range, provenance, method, limitations and rate-table hash. The actual table
is included in run source snapshots. The budget diagnostic and RF plasma-map
reader use the selected interval rather than a universal 7 eV bound.

## Data and calculation

Official source: [Phelps database, LXCat](https://www.lxcat.net/Phelps), retrieved
November 4, 2026. The dataset header attributes the compilation to Yamabe,
Buckman and Phelps, *Phys. Rev. A* **27**, 1345 (1983), with October 1997
revisions. Its process comments identify low-energy excitation with
Schapert–Scheibner and ionization with Rapp–Schram. Preserve the contributor
attribution in publications; consult the [Phelps compilation notes](https://us.lxcat.net/notes/index.php?download=phelps5)
and [LXCat reference guidance](https://nl.lxcat.net/instructions/how_reference.php).

This repository contains **calculated Maxwellian rates**, not a redistributed
copy/reformat of the raw cross-section dataset. Obtain raw input directly from
LXCat for regeneration: select complete cross-section sets, electron + Ar,
Phelps, and retain the supplied citation information. The raw input SHA-256 is
recorded in the generated table. LXCat's
[redistribution policy](https://nl.lxcat.net/instructions/redistribution.php)
applies to its downloaded data.

For kinetic energy E and electron temperature Te in eV:

    k(Te) = sqrt(8 e / (pi me)) Te^(-3/2)
            integral sigma(E) E exp(-E/Te) dE.

Raw cross sections are linearly interpolated on the union of their energy knots.
Sixteen-point Gauss–Legendre quadrature is applied on each interval. The common
collision-energy range reaches 10,000 eV. No raw cross-section extrapolation is
performed. The 1–20 eV interval is a deliberately selected numerical table
range covered by these integrals, not an assertion of experimental model
validity throughout that interval. All rate queries outside it fail.
A 401-point geometric Te grid uses positivity-preserving PCHIP interpolation
of log(rate) against log(Te), with extrapolation disabled.

The Phelps set contains EFFECTIVE total momentum transfer, EXCITATION and
IONIZATION. These types are defined in the
[BOLSIG+ input documentation](https://www.bolsig.laplace.univ-tlse.fr/wp-content/uploads/2016/03/bolsigdoc0316.pdf).
Elastic momentum transfer is recovered by subtracting the excitation and
ionization cross sections from EFFECTIVE. Negative recovered values are
rejected. Mobility uses the Maxwellian-averaged EFFECTIVE momentum-transfer
coefficient in the existing collision-frequency closure; it is not a
Boltzmann-computed swarm mobility. Elastic energy transfer uses the recovered
ELASTIC cross section, avoiding inelastic double counting:

    elastic loss coefficient [eV m3/s]
      = (2 me/MAr) <sigma_el v E> (1 - Tg_eV/Te).

This is the heavy-neutral Maxwellian elastic energy-exchange approximation.
The existing collision floor remains in the momentum frequency without a new
floor-related energy sink. The ionization energy loss is 15.8 eV per event;
the effective total excitation sink is 11.5 eV per event, matching the chosen
cross-section set. They replace the legacy 15.76/12.14 eV constants only when
this backend is selected. The 11.5 eV sink is an effective excitation model,
not a resolved population of all physical excited states.

## Verification and interpretation

`validation/argon-phelps-reference.json` contains independently evaluated
adaptive QUADPACK integrals at 1, 1.7, 3, 5.3, 7, 10, 13.7 and 20 eV. The
maximum relative difference of the interpolated rates was 2.51e-8. Analytic
constant-cross-section tests check Maxwellian normalization and the energy
moment `<sigma v E> = 2 Te <sigma v>`. Tests also check table bounds,
selected collision frequency, energy-channel accounting, hot plasma-map
support and budget provenance.

`validation/argon-chemistry-overlap.csv` compares legacy and new rates over
1–7 eV. At 7 eV, new/legacy ratios are about 1.064 for ionization, 0.892 for
excitation and 0.607 for elastic momentum-transfer rate. Different datasets
are not forced to agree. Agreement with independent quadrature establishes
numerical calculation accuracy, not independent validation of the cross
sections or the fluid model.

On the older RF case available locally
(`runs/rf-compact-20261023-194156/mumps-ooc-4000-20261026-130445`), the 4x4,
100 W experiment completed alpha=1 in 0.025 increments. Endpoint Te spanned
3.163–7.816 eV; maximum local residual was 1.82e-13 and relative power error
1.57e-15. This differs from the latest Linux RF case, which must be tested
with the command above. No physical validation or mesh convergence is claimed.

The model still omits metastable kinetics, stepwise ionization, superelastic
collisions, multiple charge states and recombination. Maxwellian EEDF, prescribed
transport coefficients, uniform axial B and boundary losses remain assumptions.
A wider table removes the previous fit boundary; it does not establish the
adequacy of those closures at higher temperature.

## Reproduce from an official raw download

Extract the Phelps Ar `Cross section.txt` locally; then use a new output path:

```bash
python3 -m helicon.argon_chemistry \
  --cross-sections '/path/to/Cross section.txt' \
  --out /tmp/argon-phelps-regenerated.json

python3 -m helicon.argon_chemistry --verify \
  --cross-sections '/path/to/Cross section.txt' \
  --out runs/argon-rate-verification
```

Verification requires the exact raw-input hash used for the bundled calculation.
A later raw database version is a new dataset requiring explicit review and
regeneration; it is not silently substituted into an archived experiment.
