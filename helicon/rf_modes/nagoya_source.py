"""Finite-width divergence-free Nagoya III sheet, peak current convention."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import spsolve
from ..rf.materials import MU0


@dataclass(frozen=True)
class Sheet:
    radius: float
    length: float
    width: float   # angular rail width, radians
    end_band: float
    current: float

    def axial_integrals(self, z):
        """Integral of the rail envelope per cell and of its derivative per nodal hat."""
        rails = np.zeros(len(z)-1)
        returns = np.zeros_like(z)
        for j, (lo, hi) in enumerate(zip(z[:-1], z[1:])):
            knots = [lo] + [a for a in (self.end_band, self.length-self.end_band) if lo<a<hi] + [hi]
            for a, b in zip(knots[:-1], knots[1:]):
                midpoint = (a+b)/2
                rails[j] += (b-a)*min(1., midpoint/self.end_band, (self.length-midpoint)/self.end_band)
                slope = (int(midpoint < self.end_band)-int(midpoint > self.length-self.end_band))/self.end_band
                weight = (midpoint-lo)/(hi-lo)
                returns[j:j+2] += slope*(b-a)*np.array([1-weight, weight])
        return rails, returns

    def modal_load(self, m, r, z):
        ring = np.flatnonzero(np.isclose(r, self.radius, atol=1e-12, rtol=0)).item()
        rails, returns = self.axial_integrals(z)
        # 2π times the Fourier coefficients of g' and g respectively.
        derivative = (1-(-1)**abs(m))*np.sinc(m*self.width/(2*np.pi))
        primitive = derivative/(1j*m) if m else 0.
        return ring, self.current*derivative*rails, -self.current*primitive*returns

    def segment_integrals(self, lo, hi):
        """Real-space integration on one unwrapped angular panel; split at strap edges."""
        h = self.width/2
        knots = [lo] + [a for a in (h, np.pi-h, np.pi+h, 2*np.pi-h) if lo<a<hi] + [hi]
        vertical = np.zeros(2)
        tangential = 0.
        for a, b in zip(knots[:-1], knots[1:]):
            phi = (a+b)/2
            if phi < h:
                g, gp = phi/self.width, 1/self.width
            elif phi < np.pi-h:
                g, gp = .5, 0.
            elif phi < np.pi+h:
                g, gp = (np.pi-phi)/self.width, -1/self.width
            elif phi < 2*np.pi-h:
                g, gp = -.5, 0.
            else:
                g, gp = (phi-2*np.pi)/self.width, 1/self.width
            weight = (phi-lo)/(hi-lo)
            vertical += gp*(b-a)*np.array([1-weight, weight])
            tangential += g*(b-a)/(hi-lo)
        return vertical, tangential


def solve_current(matrix, fixed, current_load, omega):
    free = ~fixed
    rhs = 1j*omega*MU0*current_load[free]
    x = np.zeros(matrix.shape[0], complex)
    a = matrix[free][:, free].tocsc()
    if np.any(rhs):
        x[free] = spsolve(a, rhs)
    residual = np.linalg.norm(a @ x[free] - rhs)/max(np.linalg.norm(rhs), 1e-300)
    work = -.5*np.vdot(x, current_load).real
    if not np.isfinite(x).all() or not np.isfinite(residual):
        raise ArithmeticError("Nonfinite prescribed-current solution")
    return x, float(work), float(residual)
