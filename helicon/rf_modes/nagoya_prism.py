"""3D prism Maxwell reference with real-space antenna loads and discrete axial transforms."""
import numpy as np
from scipy.fft import dst, dct, idst, idct
from scipy.sparse import bmat, coo_matrix
from ..rf.materials import EPS_0, MU0, dissipation_tensor
from .nagoya_source import solve_current


def disk(r, sectors):
    theta = np.arange(sectors)*2*np.pi/sectors
    circles = r[1:, None, None]*np.column_stack((np.cos(theta), np.sin(theta)))
    xy = np.vstack(([0., 0.], circles.reshape(-1, 2)))
    rings = np.arange(1, len(xy)).reshape(-1, sectors)
    triangles, bands = [], []
    for i, outer in enumerate(rings):
        for j, b in enumerate(outer):
            k = (j+1) % sectors
            if i:
                inner = rings[i-1]
                triangles.extend(((inner[j], b, outer[k]), (inner[j], outer[k], inner[k])))
                bands.extend((i, i))
            else:
                triangles.append((0, b, outer[k]))
                bands.append(0)
    return xy, rings, np.asarray(triangles), np.asarray(bands)


def sparse_block(a, rows, cols, shape):
    rid = np.broadcast_to(rows[:, :, None], a.shape)
    cid = np.broadcast_to(cols[:, None, :], a.shape)
    return coo_matrix((a.ravel(), (rid.ravel(), cid.ravel())), shape=shape).tocsr()


def solve_prisms(c, r, eps, sheet, nz, sectors):
    omega = 2*np.pi*c['frequency_Hz']
    k2 = omega**2*MU0*EPS_0
    xy, rings, tri, bands = disk(r, sectors)
    pairs = np.array([[0, 1], [1, 2], [2, 0]])
    edge_vertices = np.sort(tri[:, pairs], axis=-1)
    edges, ids = np.unique(edge_vertices.reshape(-1, 2), axis=0, return_inverse=True)
    ids = ids.reshape(-1, 3)
    nh, nv = len(edges), len(xy)
    coords = xy[tri]
    jac = (coords[:, 1:] - coords[:, :1]).transpose(0, 2, 1)
    area = np.linalg.det(jac)/2
    grad = np.array([[-1., -1.], [1., 0.], [0., 1.]]) @ np.linalg.inv(jac)
    a, b = pairs.T
    sign = np.where(tri[:, a]<tri[:, b], 1, -1)
    curl = 2*sign*(grad[:, a, 0]*grad[:, b, 1] - grad[:, a, 1]*grad[:, b, 0])
    materials = eps[bands]
    hmass = np.zeros((len(tri), 3, 3))
    hmaterial = np.zeros_like(hmass, dtype=complex)
    coupling = np.zeros_like(hmass)
    for bary in np.eye(3)*.5+1/6:
        w = sign[..., None]*(bary[a, None]*grad[:, b]-bary[b, None]*grad[:, a])
        wt = w.transpose(0, 2, 1)
        hmass += area[:, None, None]/3*(w @ wt)
        hmaterial += area[:, None, None]/3*(w @ materials[:, :2, :2] @ wt)
        coupling -= area[:, None, None]/3*(w @ grad.transpose(0, 2, 1))
    vmass = area[:, None, None]*(1+np.eye(3))/12
    vmaterial = vmass*materials[:, 2, 2, None, None]
    curl2 = area[:, None, None]*curl[:, :, None]*curl[:, None, :]
    vgrad = area[:, None, None]*(grad @ grad.transpose(0, 2, 1))
    stiffness = sparse_block(curl2-k2*hmaterial, ids, ids, (nh, nh))
    mass = sparse_block(hmass, ids, ids, (nh, nh))
    mixed = sparse_block(coupling, ids, tri, (nh, nv))
    z = np.linspace(0, sheet.length, nz+1)
    dz = sheet.length/nz
    axial = dz*sparse_block(vgrad-k2*vmaterial, tri, tri, (nv, nv))
    rail_z, return_z = sheet.axial_integrals(z)
    ring_index = np.flatnonzero(np.isclose(r[1:], sheet.radius, atol=1e-12, rtol=0)).item()
    conductor = rings[ring_index]
    edge_id = {tuple(e): i for i, e in enumerate(edges)}
    hdrive, vdrive = np.zeros(nh), np.zeros(nv)
    for j, start in enumerate(conductor):
        end = conductor[(j+1) % sectors]
        vertical, tangential = sheet.segment_integrals(j*2*np.pi/sectors, (j+1)*2*np.pi/sectors)
        vdrive[[start, end]] += sheet.current*vertical
        hdrive[edge_id[tuple(sorted((start, end)))]] = -sheet.current*tangential*np.sign(end-start)
    # Orthogonal transforms diagonalize this z-invariant 3D prism matrix.
    # All axial harmonics are retained, with independent 3D angular edge DOFs.
    hsource = dst(return_z[1:-1], type=1, norm='ortho')
    vsource = dct(rail_z, type=2, norm='ortho')
    fixed_h = np.isin(edges, rings[-1]).all(axis=1)
    fixed_v = np.isin(np.arange(nv), rings[-1])
    fixed = np.r_[fixed_h, fixed_v]
    hs = np.zeros((nz-1, nh), complex)
    vs = np.zeros((nz, nv), complex)
    residual_sq, rhs_sq, work = 0., 0., 0.
    for k in range(nz):
        if k:
            angle = np.pi*k/nz
            diff = 2*np.sin(angle/2)
            hh = dz*(2+np.cos(angle))/3*stiffness + diff**2/dz*mass
            hv = diff*mixed
            matrix = bmat([[hh, hv], [hv.T, axial]], format='csr')
            source = np.r_[hsource[k-1]*hdrive, vsource[k]*vdrive]
            x, pin, res = solve_current(matrix, fixed, source, omega)
            hs[k-1], vs[k] = x[:nh], x[nh:]
            norm2 = np.vdot(source[~fixed], source[~fixed]).real
        else:
            source = vsource[0]*vdrive
            vs[0], pin, res = solve_current(axial, fixed_v, source, omega)
            norm2 = np.vdot(source[~fixed_v], source[~fixed_v]).real
        residual_sq += res**2*norm2
        rhs_sq += norm2
        work += pin
    h = np.pad(idst(hs, type=1, norm='ortho', axis=0), ((1, 1), (0, 0)))
    v = idct(vs, type=2, norm='ortho', axis=0)
    h_loss, v_loss = dissipation_tensor(hmaterial), dissipation_tensor(vmaterial)
    watts = np.zeros((nz, len(r)-1))
    for j in range(nz):
        ha, hb, vz = h[j, ids], h[j+1, ids], v[j, tri]
        def energy(left, matrix, right):
            return np.einsum('ti,tij,tj->t', left.conj(), matrix, right).real
        q = (energy(ha, h_loss, ha)+energy(hb, h_loss, hb)+energy(ha, h_loss, hb))/3
        q += energy(vz, v_loss, vz)
        watts[j] = np.bincount(bands, weights=.5*omega*EPS_0*dz*q, minlength=len(r)-1)
    return watts, dict(source_work_W=work, free_relative_residual=float(np.sqrt(residual_sq/rhs_sq)),
                       free_dofs=int((nz-1)*(~fixed_h).sum()+nz*(~fixed_v).sum()),
                       sectors=sectors, volume_ratio_polygon_to_circle=sectors*np.sin(2*np.pi/sectors)/(2*np.pi))
