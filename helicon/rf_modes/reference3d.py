"""Independent small 3D prism edge-element solve for transverse plane-wave drives."""
import numpy as np

from .fem import assemble, dirichlet_solve, local_operator
from ..rf.materials import absorbed_density


def prism_basis(grad, pairs, bary, v, dz):
    basis = np.zeros((9, 3))
    curl = np.zeros_like(basis)
    for edge, (a, b) in enumerate(pairs):
        whitney = bary[a]*grad[b] - bary[b]*grad[a]
        curl2 = 2*(grad[a, 0]*grad[b, 1] - grad[a, 1]*grad[b, 0])
        for offset, shape, derivative in ((0, 1-v, -1/dz), (3, v, 1/dz)):
            row = edge + offset
            basis[row, :2] = shape*whitney
            curl[row] = [-derivative*whitney[1], derivative*whitney[0], shape*curl2]
    basis[6:, 2] = bary
    curl[6:, 0] = grad[:, 1]
    curl[6:, 1] = -grad[:, 0]
    return basis, curl


def solve_reference(config, epsilon, drives, nz, sectors):
    """Polygonal cylinder, horizontal Whitney edges x linear z, vertical nodal x DG0."""
    omega = 2*np.pi*config['frequency_Hz']
    dz = config['length_m']/nz
    theta = 2*np.pi*np.arange(sectors)/sectors
    vertices = np.vstack(([0, 0], config['radius_m']*np.c_[np.cos(theta), np.sin(theta)]))
    triangles = np.array([(0, j+1, (j+1) % sectors+1) for j in range(sectors)])
    pairs = [(0, 1), (1, 2), (2, 0)]
    edges = np.sort(triangles[:, pairs], axis=-1)
    global_edges, edge_ids = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
    horizontal = np.arange((nz+1)*len(global_edges)).reshape(nz+1, -1)
    vertical = horizontal.size + np.arange(nz*len(vertices)).reshape(nz, -1)
    ids = edge_ids.reshape(sectors, 3)
    cells = np.concatenate([horizontal[:-1, ids], horizontal[1:, ids], vertical[:, triangles]], axis=2)
    # Degree-two triangle quadrature and two-point Gauss in z.
    bary_nodes = (np.eye(3)*.5 + 1/6)
    z_nodes = (.5 - .5/np.sqrt(3), .5 + .5/np.sqrt(3))
    blocks = np.zeros((sectors, 9, 9), complex)
    elements = []
    areas = []
    for i, tri in enumerate(triangles):
        coords = vertices[tri]
        jac = (coords[1:]-coords[0]).T
        area = np.linalg.det(jac)/2
        grad = np.array([[-1., -1.], [1., 0.], [0., 1.]]) @ np.linalg.inv(jac)
        oriented = [(a, b) if tri[a] < tri[b] else (b, a) for a, b in pairs]
        samples = []
        for bary in bary_nodes:
            for v in z_nodes:
                basis, curl = prism_basis(grad, oriented, bary, v, dz)
                blocks[i] += area*dz/6 * local_operator(basis, curl, epsilon, omega)
                samples.append(basis)
        elements.append((samples, prism_basis(grad, oriented, [1/3]*3, .5, dz)))
        areas.append(area)
    size = horizontal.size+vertical.size
    matrix = assemble(cells.reshape(-1, 9), np.tile(blocks, (nz, 1, 1)), size)
    fixed = np.zeros(size, bool)
    fixed[horizontal[[0, -1]]] = True
    fixed[vertical[:, 1:]] = True
    lift = np.zeros(size, complex)
    drive = sum(a*np.array([1., 1j*m]) for m, a in drives.items())
    lift[horizontal[0]] = np.diff(vertices[global_edges], axis=1)[:, 0] @ drive
    solution, pin, residual = dirichlet_solve(matrix, fixed, lift, omega)
    coefficients = solution[cells]
    electric = np.empty((nz, sectors, 3), complex)
    magnetic = np.empty_like(electric)
    watts = np.zeros((nz, sectors))
    for i, (samples, (basis, curl)) in enumerate(elements):
        electric[:, i] = coefficients[:, i] @ basis
        magnetic[:, i] = coefficients[:, i] @ curl/(1j*omega)
        for basis in samples:
            q = absorbed_density(coefficients[:, i] @ basis, epsilon, config['frequency_Hz'])
            watts[:, i] += areas[i]*dz*q/6
    return (dict(xy_vertices_m=vertices, triangles=triangles,
                 z_edges_m=config['z_start_m'] + np.linspace(0, config['length_m'], nz+1),
                 E_cart_peak_V_m=electric, B_cart_peak_T=magnetic, cell_absorbed_W=watts),
            dict(area_m2=sum(areas), absorption_W=float(watts.sum()),
                 input_reaction_W=pin, free_relative_residual=residual,
                 free_dofs=int(np.count_nonzero(~fixed))))
