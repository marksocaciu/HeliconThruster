# Global argon plasma model, version 1

This is a steady 0D chamber model: three coupled balances determine mean electron
density, neutral density and electron temperature. It is an initial engineering
model with explicit assumed boundary losses, not a reproduction or validation of
the Iannarelli thruster framework. It has no spatial solution or RF wave solver.

## Run on macOS or Linux

From `/Users/mark/Documents/HeliconThruster` on the Mac, or the project directory
on Linux:

```bash
python3 -m helicon.plasma --out runs/plasma-baseline --no-plots
```

The command above needs only Python's standard library. No MPI, FEniCS, NumPy,
Gmsh, or meshing step is required. To include the budget figure, use your existing
Python environment with matplotlib, or install the small plasma dependency file:

```bash
python3 -m pip install -r requirements-plasma.txt
python3 -m helicon.plasma --out runs/plasma-with-plots
```

Each command needs a NEW output directory. Open `report.html` there. The run also
saves the plasma/geometry inputs, complete result JSON, particle and power CSVs,
source snapshot, hashes, Python/platform details, and failures in `run.json`.

```bash
python3 -m helicon.plasma --config config/plasma_argon.json \
  --geometry runs/convergence/domain_02/geometry.json \
  --absorbed-power 500 --mass-flow-mg-s 1.5 --out runs/plasma-case-02
```

`--absorbed-power` means power delivered to the plasma, NOT generator power.
Only chamber radius and length from the geometry enter these equations; changing
coil current or mesh resolution does not change the plasma calculation yet.
The geometry snapshot preserves that distinction rather than pretending to
derive magnetic confinement from a chamber-averaged magnetic field.

The example is 500 W absorbed power, 1.5 mg/s argon, 400 K neutrals, R=3 cm and
L=12 cm. It is deliberately an illustrative case within the reaction-fit range,
not the paper's approximately 1 kW case. Its loss ratios are assumptions.

## Variables and closures

- `ne`, `nn`: volume-mean electron/ion and ground-state neutral density, m^-3.
- `Te`: Maxwellian electron temperature in eV; cold, singly charged ions.
- V=pi R^2 L; side area As=2 pi R L; back and exit areas Ab=Ae=pi R^2.
- cs=sqrt(e Te/mi), with mi=39.948 atomic mass units.
- hs, hb, he are boundary-to-VOLUME-MEAN density ratios. They are NOT the
  sheath-edge-to-centerline ratios in some published models. Default values
  0.1, 0.5, 0.5 are illustrative effective loss coefficients, not measurements
  or magnetic-field fits. They must be justified or calibrated for real use.
- Effective loss area Aeff=hs As+hb Ab+he Ae.
- Ion loss rates Lj=ne cs hj Aj, separately for side, back and exit.
- Neutrals escape into vacuum with conductance Cn=tau Ae vbar/4, where
  vbar=sqrt(8 kB Tn/(pi mi)). `neutral_exit_transmission` tau is in (0,1].
  This is a thermal-effusion assumption, not a solved neutral-flow field. It
  can be inaccurate for directed or collisional gas flow and finite chamber
  background pressure. Neutral temperature is prescribed, not solved.
- Wall recycling r in [0,1] immediately returns r times the wall ion rate as
  cold neutrals. The remaining fraction is removed/retained at the wall. Full
  recycling is the example default. Exit particles never recycle.

## Conservation equations

All Ndot/L/S terms below are particles per second; Q=mass_flow/mi.

    S = ne nn kiz(Te) V
    S = Lside + Lback + Lexit
    Q + r (Lside + Lback) = S + Cn nn

Adding the species balances gives the independently checked atom balance:

    Q = Lexit + Cn nn + (1-r) (Lside + Lback)

The power balance is:

    Pabs = Pion + Pexc + Pel + Pside + Pback + Pexit
    Pion = e S * 15.76
    Pexc = e ne nn V kexc * 12.14
    Pel  = e ne nn V kel * (3 me/mi) * (Te - kB Tn/e)
    Pside/back = e Lside/back gamma_wall Te
    gamma_wall = 2.5 + 0.5 ln(mi/(2 pi me))
    Pexit = e Lexit * 3 Te

Wall loss is the standard floating, non-emitting planar-sheath approximation:
2 Te escaping-electron energy plus 0.5 Te Bohm-ion energy plus the sheath drop.
Applying it at the side wall is a simplification for a magnetized source.
`Pion` already accounts for ionization binding energy; do not add recombination
energy again to the total budget. Wall-power entries are therefore kinetic/sheath
channels, not the total heat a physical wall would receive.

The OPEN exit has a deliberately different closure: 5/2 Te electron enthalpy
plus 1/2 Te ion directed kinetic energy, at sonic flow. No floating-wall sheath
is imposed there. Electron conductive heat flux through that plane is set to
zero. This is an assumed source energy boundary condition; a later nozzle model
must replace it with a consistent coupled electron-energy flux. It does not
calculate a nozzle potential drop or final beam energy.

Ionization and excitation remove electron energy; excitation is represented
as an effective irreversible loss, with no stored metastable population or
stepwise ionization. Electron-neutral elastic transfer is a sink into a gas
held at the specified Tn. Neutral injection/escape thermal enthalpy is not part
of the electron/plasma power budget. Ion-neutral momentum and energy transfer,
volume recombination, multiple charge states and secondary emission are omitted.

## Reaction data and provenance

The implementation uses these common ground-state argon Maxwellian fits,
with Te in eV and k in m^3/s:

    kiz  = 2.34e-14 Te^0.59 exp(-17.44/Te)
    kexc = 2.48e-14 Te^0.33 exp(-12.78/Te)
    kel  = 2.336e-14 Te^1.609 exp(0.0618 ln(Te)^2 - 0.1171 ln(Te)^3)

The Arrhenius parameters 17.44 and 12.78 are fit parameters, NOT the energy
thresholds used in the power balance (15.76 and effective 12.14 eV).

Sources:

1. Lieberman & Lichtenberg, *Principles of Plasma Discharges and Materials
   Processing*, 2nd ed., Wiley (2005), foundational global-model/rate reference.
2. [Acta Physica Sinica 73, 215201 (2024), Table 1](https://wulixb.iphy.ac.cn/article/pdf/preview/10.7498/aps.73.20240952.pdf)
   provides the three numerical rate fits and SI units. Only the ground-state
   reactions are used here, not that paper's full reaction network or geometry.
3. [University of Washington dissertation (2020), discussion around Eq. 2.39](https://digital.lib.washington.edu/server/api/core/bitstreams/74a04c0e-45de-4295-9b74-860aa16b8290/content)
   reproduces the fits, excitation energy and reported 1--7 eV accuracy range.
4. [Iannarelli et al. (2026)](https://doi.org/10.1007/s44205-026-00197-7), source
   power-accounting discussion, is a roadmap for the later model, not the
   source of the numerical rates here. Its axial flow, PIC-based radial loss
   fit, detailed excitation rates, and nozzle closure are NOT implemented.
5. [Guaita et al. (2022)](https://re.public.polimi.it/handle/11311/1201457)
   demonstrates a 0D source followed by a separate acceleration model. This
   implementation is not a reproduction of that model either.

No extrapolation outside 1--7 eV is permitted. A missing root means the requested
case has no solution WITHIN THIS MODEL'S SUPPORTED INTERVAL; it is not proof of
physical extinction. Broader-temperature use requires replacing the rate model
with documented cross-section integration or validated broader-range fits.

## Numerical method and verification

For positive power, ion balance gives nn=cs Aeff/(V kiz). Power balance then
gives ne=Pabs/(power loss per unit ne). Substituting both into atom balance
leaves one scalar equation in Te. The solver brackets it in 1--7 eV and bisects;
it rejects multiple detected roots and independently checks all balances.
It finds a steady branch, not its dynamic stability or ignition threshold.

At exactly zero absorbed power, it returns the neutral-only solution
ne=0, nn=Q/Cn, and undefined Te (JSON null). No artificial seed plasma or
meaningless zero-temperature rate evaluation is used.

Tests include independently constructed steady states, recovery across the
supported temperature interval, full/zero wall recycling, mass/power budgets,
the cold-gas limit, out-of-range failure, source/geometry isolation, and record
integrity. These verify equations and implementation. Experimental validation
and reproduction of a published operating curve remain future work.

## Interpret the result

- Propellant utilization = Lexit/Q. This differs from ne/(ne+nn), the local
  ionized number fraction: ions and neutrals leave at different speeds.
- Equivalent ion current = e Lexit; it is not the net current of the ambipolar
  plume, whose outgoing electrons are not shown as an external current draw.
- Wall loss fractions use total ion production as denominator. With recycling,
  ionization events can exceed injected atoms without violating conservation.
- Absorbed energy per exit ion includes energy spent on particles that hit
  walls. It is not exit ion kinetic energy and must not be converted to thrust.
- Density and Te are mean plasma-state predictions conditional on the closures.
  Check exit ion rate, utilization and power losses alongside them.

Next development should replace prescribed loss ratios with a justified
confinement/axial-flow model and couple the exit energy flux to a nozzle model.
Neither B-field optimization nor a thrust claim is justified by this 0D version.
