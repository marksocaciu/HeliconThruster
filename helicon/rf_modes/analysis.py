"""Analytical parallel cold-plasma wave and phase-preserving reconstruction."""
import numpy as np
from ..argon import E_CHARGE, M_E
from ..rf.materials import EPS_0, MU0


def relative_error(value, reference):
    return float(np.linalg.norm(np.asarray(value)-reference)/max(np.linalg.norm(reference), 1e-300))


def cylinder_benchmark(config, m, amplitude, fields):
    omega = 2*np.pi*config['frequency_Hz']
    omega_p2 = config['electron_density_m3']*E_CHARGE**2/(M_E*EPS_0)
    omega_c = E_CHARGE*config['Bz_T']/M_E
    eps = 1 - omega_p2/(omega*(omega-m*omega_c+1j*config['collision_frequency_s']))
    k = np.sqrt(omega**2*MU0*EPS_0*eps)
    length = config['length_m']
    z = fields['z_edges_m'] - config['z_start_m']
    dz = z[1]-z[0]
    # Reflection from a PEC end. Use decaying exponentials for evanescent branches.
    reflection = np.expm1(2j*k*length)
    exact = amplitude*np.exp(1j*k*z)*np.expm1(2j*k*(length-z))/reflection
    profile = fields['Er_edge_V_m'].mean(axis=1)
    if np.linalg.norm(profile[1:-1]) == 0:
        raise ArithmeticError('Degenerate axial field profile')
    cos_kh = np.vdot(profile[1:-1], profile[:-2]+profile[2:])/(2*np.vdot(profile[1:-1], profile[1:-1]))
    measured = np.arccos(cos_kh)/dz
    if abs(-measured-k) < abs(measured-k):
        measured = -measured
    # Two transverse components, peak phasors. At z=0, E'(0)=-k cot(k L) E(0).
    admittance = -k/(1j*omega*MU0*np.tan(k*length))
    pin = np.pi*config['radius_m']**2 * abs(amplitude)**2 * admittance.real
    return (dict(k_analytic_per_m=[k.real, k.imag], k_measured_per_m=[measured.real, measured.imag],
                 k_relative_error=relative_error(measured, k), analytic_input_W=float(pin),
                 axial_field_relative_error=relative_error(fields['Er_edge_V_m'], np.broadcast_to(exact[:, None], fields['Er_edge_V_m'].shape))),
            dict(z_profile_m=fields['z_edges_m'], Er_numeric_V_m=profile, Er_analytic_V_m=exact))


def reconstruct(modal_fields, phi):
    """Cell-centre peak phasors in the local cylindrical basis, phi in radians."""
    e = sum(f['E_cyl_peak_V_m'] * np.exp(1j*m*phi) for m, f in modal_fields.items())
    b = sum(f['B_cyl_peak_T'] * np.exp(1j*m*phi) for m, f in modal_fields.items())
    return e, b


def cartesian(field, phi):
    r, azimuthal, z = np.moveaxis(field, -1, 0)
    return np.stack((r*np.cos(phi)-azimuthal*np.sin(phi),
                     r*np.sin(phi)+azimuthal*np.cos(phi), z), axis=-1)


def compare_fields(modal_fields, reference, stats, total, config):
    xy = reference['xy_vertices_m'][reference['triangles']].mean(axis=1)
    radii = next(iter(modal_fields.values()))['r_edges_m']
    e_samples, b_samples = [], []
    for x, y in xy:
        phi = np.arctan2(y, x)
        i = np.searchsorted(radii, np.hypot(x, y))-1
        e, b = reconstruct(modal_fields, phi)
        e_samples.append(cartesian(e[:, i], phi))
        b_samples.append(cartesian(b[:, i], phi))
    # The independent reference has a polygonal cross section. Compare power per area.
    area = np.pi*config['radius_m']**2
    return dict(E_relative_error=relative_error(np.stack(e_samples, axis=1), reference['E_cart_peak_V_m']),
                B_relative_error=relative_error(np.stack(b_samples, axis=1), reference['B_cart_peak_T']),
                absorbed_W_m2_relative_error=relative_error(total/area, stats['absorption_W']/stats['area_m2']))
