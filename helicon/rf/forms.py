"""UFL definitions shared by the RF solver and compiler-only checks."""
import numpy as np
from .materials import EPS_0,MU0


def pml_tensors(x,c):
    import ufl
    d=c['domain'];t=d['pml_thickness_m']
    bounds=[(-d['half_width_m'],d['half_width_m']),(-d['half_width_m'],d['half_width_m']),(d['z_min_m'],d['z_max_m'])]
    stretch=[]
    for i,(lo,hi) in enumerate(bounds):
        distance=ufl.max_value(ufl.max_value(ufl.real(lo-x[i]),ufl.real(x[i]-hi)),0)
        stretch.append(1+1j*d['pml_strength']*(distance/t)**2)
    det=stretch[0]*stretch[1]*stretch[2]
    return (ufl.diag(ufl.as_vector([det/s**2 for s in stretch])),
            ufl.diag(ufl.as_vector([s**2/det for s in stretch])))


def bilinear(domain,u,v,eps,dx,c):
    import ufl
    epml,invmu=pml_tensors(ufl.SpatialCoordinate(domain),c)
    k02=(2*np.pi*c['frequency_Hz'])**2*MU0*EPS_0
    form=(ufl.inner(ufl.curl(u),ufl.curl(v))-k02*ufl.inner(eps*u,v))*dx((1,2,4,6))
    form+=(ufl.inner(invmu*ufl.curl(u),ufl.curl(v))-k02*ufl.inner(epml*u,v))*dx(7)
    return form


def interface_flux(domain,E,frequency_Hz,center):
    """Legacy averaged trace retained for diagnostic comparison only."""
    import ufl
    w=2*np.pi*frequency_Hz
    H=ufl.curl(E)/(1j*w*MU0)
    x=ufl.SpatialCoordinate(domain);n=ufl.FacetNormal(domain)
    normal=ufl.conditional(ufl.gt(ufl.real(ufl.dot(n('+'),x('+')-ufl.as_vector(center))),0),n('+'),-n('+'))
    return .5*ufl.real(ufl.dot(ufl.cross(ufl.avg(E),ufl.conj(ufl.avg(H))),normal))


def interface_flux_traces(domain, E, frequency_Hz, physical_indicator):
    """Outward-from-core fluxes evaluated separately on the two interface sides.

    physical_indicator is DG0: one in physical cells, zero in PML cells,
    including ghost cells. Restricting by material avoids dependence on the
    arbitrary +/- facet orientation. H(curl) elements enforce tangential E
    continuity, but their discrete curl (and hence H) need not be continuous.
    The physical-side trace defines power leaving the physical domain.
    """
    import ufl
    H = ufl.curl(E)/(1j*2*np.pi*frequency_Hz*MU0)
    n = ufl.FacetNormal(domain)
    physical, pml = 0, 0
    for side in ('+', '-'):
        local = .5*ufl.real(ufl.dot(
            ufl.cross(E(side), ufl.conj(H(side))), n(side)))
        physical += physical_indicator(side)*local
        # PML's outward cell normal points back into the physical domain.
        pml -= (1-physical_indicator(side))*local
    return physical, pml


def positive_preconditioner(u, v, eps, dx, c, mass_scale=1.):
    """Experimental definite auxiliary operator, NOT a change to RF physics.

    beta = mass_scale * k0^2 * max(1, ||epsilon_r||_F/sqrt(3)).
    The normalized Frobenius norm is a positive scale heuristic, not a
    spectral equivalence guarantee. Use real identity permeability throughout,
    including PML cells; the complex PML and tensor plasma remain in A.
    """
    import ufl
    if not np.isfinite(mass_scale) or mass_scale <= 0:
        raise ValueError('HELICON_RF_PC_MASS_SCALE must be positive and finite')
    k02=(2*np.pi*c['frequency_Hz'])**2*MU0*EPS_0
    if not np.isfinite(k02) or k02 <= 0:
        raise ValueError('Positive preconditioner needs a positive finite frequency')
    weight=ufl.real(ufl.sqrt(ufl.max_value(1.,ufl.real(ufl.inner(eps,eps))/3.)))
    return (ufl.inner(ufl.curl(u),ufl.curl(v))+
            mass_scale*k02*weight*ufl.inner(u,v))*dx
