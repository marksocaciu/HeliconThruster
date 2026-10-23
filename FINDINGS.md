# Second-order RF diagnostic comparison

The 4 mm and 3 mm saved runs are not mesh-converged. No solver or tolerance changes were made during this inspection.

## Verified evidence

- All archived source files match between runs. Imported plasma summary and magnetic-map files are byte-identical. The effective configurations differ only in mesh.plasma_m (0.004 versus 0.003 m).
- Summing owned-cell integrated plasma powers across four ranks gives 0.0004914003052068105 W and 0.0007667495548052544 W, matching the respective summaries. No plasma cell has negative integrated absorption.
- Plasma cells: 24,662 versus 57,848. Integrated mesh volumes: 0.0003385285071790036 versus 0.0003388716815880265 m^3, a change of about 0.1014%, versus 56.03% in power.
- Cells with centres at radius 28--30 mm contribute 0.277603 versus 0.534930 mW. Their increase accounts for about 93.45% of the total power increase. These are centroid-based bins of whole-cell integrals, not exact geometric shell integrals; boundary-crossing cells limit spatial attribution.
- Cells with centres at radius 25--30 mm account for approximately 90.03% versus 92.91% of total absorption.
- The largest individual cell contributes only 0.1049% versus 0.02578% of total power. The increase is not an isolated-cell spike. Every 10 mm axial bin has increased integrated absorption.
- Re-evaluating the archived magnetic map at saved plasma cell centres gives |B| ranges 0.0133533--0.0141786 T and 0.0133406--0.0141815 T, with medians 0.0140860 and 0.0140863 T. This does not eliminate sensitivity to cellwise material sampling, but there is no changed map or gross field-range shift.

## Interface failure

The 3 mm physical-side interface flux is -5.388517029721042e-8 W, about -0.00702774% of accepted power. The sign tolerance is input power times 1e-8 (about 7.6675e-12 W); the separate agreement tolerance is input power times 0.001 (about 7.6675e-7 W). Therefore the sign check fails while the agreement check passes. This is consistent with the implementation, not a CSV/report contradiction. It is not a demonstrated cause of the 0.275349 mW absorption difference. Do not relax the gates to label the solution validated.

## Interpretation and limits

The evidence points to sensitivity of absorption near the plasma--wall interface; it does not establish a unique cause, a resolved physical skin layer, or the correct limiting power. The sweep remeshes the full geometry and uses first-order geometric tetrahedra even with degree-2 field elements. Fixed target sizes elsewhere do not guarantee identical surrounding meshes. The wall target is 6 mm while the physical wall is 3 mm thick.

Saved E and B arrays and XDMF fields are DG0 cell samples, not complete degree-2 finite-element coefficients. Consequently the exact high-order solution cannot be reconstructed for new quadrature or exact interface-trace evaluation from this archive. Integrated cell powers are suitable for the regional comparison performed here.

A useful next implementation is controlled refinement at the plasma--wall boundary, with adequate wall resolution, and export of full high-order solution coefficients for reproducible diagnostics. A fresh solve would still be required to validate such changes. No further Linux runs or physics fixes are claimed by this inspection. The nearly disabled PML still describes a reflecting PEC-box experiment, not the original open-boundary problem.

## Reproduce analysis

Run analyze.py and check_inputs.py in this folder using a Python environment with NumPy. Extracted inputs remain in plasma4mm and plasma3mm; numerical analysis is in analysis.json.
