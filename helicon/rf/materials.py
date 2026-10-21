"""Passive cold-electron dielectric from the momentum equation, exp(-i wt).

Stationary ion background. nu is an EFFECTIVE momentum relaxation frequency
in s^-1, not an angular RF frequency. No fitting to the global-model rates.
"""
import numpy as np
from ..argon import E_CHARGE, M_E, EPS_0
MU0=1.25663706212e-6


def cold_epsilon(frequency_Hz, density_m3, collision_frequency_Hz, B_T):
    b=np.asarray(B_T,dtype=float)
    if b.shape[-1:] != (3,) or not np.all(np.isfinite(b)):
        raise ValueError('B must have final dimension 3 and finite components')
    density_m3 = np.asarray(density_m3, dtype=float)
    collision_frequency_Hz = np.asarray(collision_frequency_Hz, dtype=float)
    if (not np.isfinite(frequency_Hz) or frequency_Hz <= 0
            or not np.isfinite(density_m3).all() or np.any(density_m3 < 0)
            or not np.isfinite(collision_frequency_Hz).all() or np.any(collision_frequency_Hz < 0)):
        raise ValueError('Invalid cold-plasma inputs')
    w=2*np.pi*frequency_Hz
    cross=np.zeros(b.shape[:-1]+(3,3))
    cross[...,0,1]=-b[...,2]; cross[...,0,2]=b[...,1]
    cross[...,1,0]=b[...,2]; cross[...,1,2]=-b[...,0]
    cross[...,2,0]=-b[...,1]; cross[...,2,1]=b[...,0]
    # (nu-iw)v + (q/m) B x v = (q/m) E; q=-e.
    matrix=(collision_frequency_Hz[..., None, None]-1j*w)*np.eye(3)-E_CHARGE/M_E*cross
    sigma=density_m3[..., None, None]*E_CHARGE**2/M_E*np.linalg.inv(matrix)
    return np.eye(3)+1j*sigma/(EPS_0*w)


def dissipation_tensor(epsilon):
    return (epsilon-np.swapaxes(epsilon.conj(),-1,-2))/(2j)


def absorbed_density(E, epsilon, frequency_Hz):
    E=np.asarray(E)
    return .5*2*np.pi*frequency_Hz*EPS_0*np.einsum('...i,...ij,...j->...',E.conj(),dissipation_tensor(epsilon),E).real


class AxisymmetricField:
    """Bilinear interpolation of saved r-z maps; no out-of-bounds extrapolation."""
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.r=data['r_m'].copy(); self.z=data['z_m'].copy()
            self.br=data['Br_T'].copy(); self.bz=data['Bz_T'].copy()
        for a in (self.r,self.z):
            if a.ndim!=1 or len(a)<2 or not np.all(np.isfinite(a)) or not np.all(np.diff(a)>0):
                raise ValueError('Field-map coordinates must be finite and strictly increasing')
        if self.r[0] != 0: raise ValueError('Field map must include r=0')
        if any(a.shape!=(len(self.z),len(self.r)) or not np.all(np.isfinite(a)) for a in (self.br,self.bz)):
            raise ValueError('Invalid field-map shape or nonfinite values')
        if not np.allclose(self.br[:,0],0,atol=1e-10,rtol=0): raise ValueError('Br must vanish on the axis')

    def __call__(self,xyz):
        xyz=np.asarray(xyz,dtype=float); r=np.hypot(xyz[:,0],xyz[:,1]); z=xyz[:,2]
        if not np.all(np.isfinite(xyz)) or np.any(r>self.r[-1]) or np.any(z<self.z[0]) or np.any(z>self.z[-1]):
            raise ValueError('RF plasma points lie outside saved magnetic map; supply a larger map')
        ir=np.clip(np.searchsorted(self.r,r)-1,0,len(self.r)-2)
        iz=np.clip(np.searchsorted(self.z,z)-1,0,len(self.z)-2)
        ar=(r-self.r[ir])/(self.r[ir+1]-self.r[ir]); az=(z-self.z[iz])/(self.z[iz+1]-self.z[iz])
        def interp(f):
            return ((1-ar)*(1-az)*f[iz,ir]+ar*(1-az)*f[iz,ir+1]+(1-ar)*az*f[iz+1,ir]+ar*az*f[iz+1,ir+1])
        br,bz=interp(self.br),interp(self.bz)
        factor=np.divide(br,r,out=np.zeros_like(r),where=r>0)
        return np.column_stack((factor*xyz[:,0],factor*xyz[:,1],bz))
