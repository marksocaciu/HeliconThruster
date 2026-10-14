"""Ground-state argon rate fits, SI rates and electron temperatures in eV.

Lieberman & Lichtenberg (2005) fits, reproduced in Table 1 of
https://doi.org/10.7498/aps.73.20240952 and in the UW dissertation linked
in docs/plasma-model.md. Restricted to the reported 1--7 eV fit interval.
This is a lumped excitation energy sink, not a metastable population model.
"""
import math

E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23
M_E = 9.1093837015e-31
M_AR = 39.948 * 1.66053906660e-27
EPS_0 = 8.8541878128e-12
TE_MIN_EV, TE_MAX_EV = 1.0, 7.0
IONIZATION_EV, EXCITATION_EV = 15.76, 12.14
RATE_MODEL = 'argon_ground_state_LL_1_to_7_eV_v1'


def rates(temperature_eV):
    """Maxwellian rate coefficients in m^3/s; never extrapolate silently."""
    t = temperature_eV
    if isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t):
        raise ValueError('Electron temperature must be a finite number in eV')
    if not TE_MIN_EV <= t <= TE_MAX_EV:
        raise ValueError('Argon rate fits require 1 <= electron temperature <= 7 eV')
    log_t = math.log(t)
    return {
        'ionization_m3_s': 2.34e-14 * t**0.59 * math.exp(-17.44/t),
        'excitation_m3_s': 2.48e-14 * t**0.33 * math.exp(-12.78/t),
        'elastic_m3_s': 2.336e-14 * t**1.609 * math.exp(
            0.0618*log_t**2 - 0.1171*log_t**3),
    }
