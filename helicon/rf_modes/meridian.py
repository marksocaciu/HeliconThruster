"""Fourier H(curl) elements on a tensor mesh in (r,z), exp(im phi-i omega t)."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix

from .fem import assemble, dirichlet_solve, local_operator
from ..rf.materials import MU0, absorbed_density


def basis_at(left, dr, dz, m, u, v):
    """Rows represent Er on two r-edges, Ez on two z-edges, r Ephi at four vertices."""
    r = left + dr*u
    scalar = np.array([(1-u)*(1-v), u*(1-v), (1-u)*v, u*v])
    du = np.array([v-1, 1-v, -v, v], dtype=float)
    dv = np.array([u-1, -u, 1-u, u], dtype=float)
    if left == 0 and m == 0:
        # Regular axisymmetric Ephi is O(r). Axis vertices are constrained to zero.
        scalar[[1, 3]] *= u
        du[[1, 3]] *= 2*u
        dv[[1, 3]] *= u
    basis = np.zeros((8, 3), dtype=complex)
    curl = np.zeros_like(basis)
    basis[:2, 0] = [1-v, v]
    basis[2:4, 2] = [1-u, u]
    basis[4:, 1] = scalar/r
    curl[:2, 1] = np.array([-1, 1])/dz
    curl[:2, 2] = -1j*m*basis[:2, 0]/r
    curl[2:4, 0] = 1j*m*basis[2:4, 2]/r
    curl[2:4, 1] = np.array([1, -1])/dr
    curl[4:, 0] = -dv/(dz*r)
    curl[4:, 2] = du/(dr*r)
    return basis, curl


def solve_mode(config, epsilon, m, amplitude, nr, nz):
    radius, length = config['radius_m'], config['length_m']
    omega = 2*np.pi*config['frequency_Hz']
    dr, dz = radius/nr, length/nz
    r = np.linspace(0, radius, nr+1)
    z = np.linspace(0, length, nz+1)
    er = np.arange((nz+1)*nr).reshape(nz+1, nr)
    ez = er.size + np.arange(nz*(nr+1)).reshape(nz, nr+1)
    angular = er.size + ez.size + np.arange((nz+1)*(nr+1)).reshape(nz+1, nr+1)
    size = er.size + ez.size + angular.size
    cells = np.stack([er[:-1], er[1:], ez[:, :-1], ez[:, 1:],
                      angular[:-1, :-1], angular[:-1, 1:],
                      angular[1:, :-1], angular[1:, 1:]], axis=-1)

    nodes, weights = leggauss(4)
    quadrature = [(u, v, wu*wv) for u, wu in zip((nodes+1)/2, weights/2)
                  for v, wv in zip((nodes+1)/2, weights/2)]
    blocks = np.zeros((nr, 8, 8), complex)
    for i in range(nr):
        for u, v, weight in quadrature:
            basis, curl = basis_at(r[i], dr, dz, m, u, v)
            blocks[i] += (2*np.pi*(r[i]+dr*u)*dr*dz*weight
                          * local_operator(basis, curl, epsilon, omega))
    matrix = assemble(cells.reshape(-1, 8), np.tile(blocks, (nz, 1, 1)), size)

    # Remove singular traces before solving. At the axis of nonzero modes,
    # d_r(r Ephi) = i m Er and Ez = 0 ensure finite curl energy.
    independent = np.ones(size, bool)
    independent[angular[:, 0]] = False
    if m:
        independent[er[:, 0]] = False
        independent[ez[:, 0]] = False
    rows = np.flatnonzero(independent)
    cols = np.cumsum(independent)-1
    values = np.ones(len(rows), complex)
    col_idx = cols[rows]
    if m:
        col_idx = np.r_[col_idx, cols[angular[:, 1]]]
        rows = np.r_[rows, er[:, 0]]
        values = np.r_[values, np.full(nz+1, 1/(1j*m*dr))]
    prolong = coo_matrix((values, (rows, col_idx)), shape=(size, independent.sum())).tocsr()
    reduced = prolong.conj().T @ matrix @ prolong

    # Longitudinal-electric / longitudinal-magnetic wall: Ez essential, Hz natural.
    fixed = np.zeros(size, bool)
    fixed[ez[:, -1]] = True
    fixed[er[[0, -1]]] = True
    fixed[angular[[0, -1]]] = True
    lift = np.zeros(size, complex)
    if m:
        power = abs(m)-1
        # Average Er along the edge, consistent with its line-integral DOF.
        lift[er[0]] = amplitude*nr*np.diff((r/radius)**(power+1))/(power+1)
        lift[angular[0]] = 1j*np.sign(m)*amplitude*r*(r/radius)**power
    else:
        lift[angular[0]] = amplitude*r*r/radius
    x, input_power, residual = dirichlet_solve(
        reduced, fixed[independent], lift[independent], omega)
    primal = prolong @ x
    coefficients = primal[cells]
    electric = np.empty((nz, nr, 3), complex)
    magnetic = np.empty_like(electric)
    watts = np.zeros((nz, nr))
    entry_flux = 0.
    for i in range(nr):
        basis, curl = basis_at(r[i], dr, dz, m, .5, .5)
        electric[:, i] = coefficients[:, i] @ basis
        magnetic[:, i] = coefficients[:, i] @ curl / (1j*omega)
        for u, v, weight in quadrature:
            basis, _ = basis_at(r[i], dr, dz, m, u, v)
            q = absorbed_density(coefficients[:, i] @ basis, epsilon, config['frequency_Hz'])
            watts[:, i] += 2*np.pi*(r[i]+dr*u)*dr*dz*weight*q
        for u, weight in zip((nodes+1)/2, weights/2):
            basis, curl = basis_at(r[i], dr, dz, m, u, 0)
            e = coefficients[0, i] @ basis
            h = coefficients[0, i] @ curl/(1j*omega*MU0)
            entry_flux += np.real(np.cross(e, h.conj())[2])*np.pi*(r[i]+dr*u)*dr*weight
    fields = dict(r_edges_m=r, z_edges_m=z + config['z_start_m'],
                  Er_edge_V_m=primal[er], Ez_edge_V_m=primal[ez], rEphi_vertex_V=primal[angular],
                  E_cyl_peak_V_m=electric, B_cyl_peak_T=magnetic, cell_absorbed_W=watts)
    return fields, dict(input_reaction_W=input_power, input_poynting_W=float(entry_flux),
                        absorption_W=float(watts.sum()), free_relative_residual=residual,
                        free_dofs=int(np.count_nonzero(~fixed[independent])))
