"""Sample the axisymmetric input material of a completed 3D RF solution."""
from pathlib import Path

import numpy as np

from ..records import digest, read_json
from ..rf.config import load
from ..rf.materials import AxisymmetricField, cold_epsilon
from ..rf.spatial_maps import PlasmaMap


def sample_rf_material(directory, config, r, z, plasma_cells, radial_epsilon):
    directory = Path(directory)
    record = read_json(directory / 'run.json')
    if record.get('status') != 'completed':
        raise ValueError('Material source must be a completed 3D RF solution')

    inputs = {}
    for name, key in [('rf.json', 'config_sha256'), ('input_plasma_fields.npz', 'plasma_map_sha256'),
                      ('background_B.npz', 'magnetic_map_sha256')]:
        path = directory / name
        if key not in record:
            raise ValueError(f'Missing RF input provenance: {key}')
        expected = record[key]
        if expected is None and name != 'rf.json' and not path.exists():
            continue  # The 3D solve used the uniform value in rf.json.
        if not path.is_file() or digest(path) != expected or (
                'artifacts_sha256' in record and record['artifacts_sha256'].get(name) != expected):
            raise ValueError(f'RF input hash mismatch: {name}')
        inputs[name] = expected

    source = load(directory / 'rf.json')
    if source['material_case'] != 'plasma':
        raise ValueError('Material source must be a plasma RF solution')
    if source['frequency_Hz'] != config['frequency_Hz']:
        raise ValueError('RF material source and modal frequencies differ')
    for key in ('radius_m', 'length_m', 'z_start_m'):
        if not np.isclose(source['chamber'][key], config['chamber'][key], atol=1e-12, rtol=0):
            raise ValueError(f'RF material source and modal chambers differ: {key}')

    rc = (r[:plasma_cells] + r[1:plasma_cells+1]) / 2
    zc = (z[:-1] + z[1:]) / 2  # Physical z faces, including the chamber offset.
    zz, rr = np.meshgrid(zc, rc, indexing='ij')
    xyz = np.column_stack((rr.ravel(), np.zeros(rr.size), zz.ravel()))
    plasma = source['plasma']
    if 'input_plasma_fields.npz' in inputs:
        state = PlasmaMap(directory / 'input_plasma_fields.npz')(xyz)
        density, collision = state['electron_density_m3'], state['collision_frequency_s']
    else:
        density, collision = plasma['electron_density_m3'], plasma['collision_frequency_Hz']
    if 'background_B.npz' in inputs:
        # At phi=0 Cartesian components coincide with the (r, phi, z) basis.
        field = AxisymmetricField(directory / 'background_B.npz')(xyz)
    else:
        if any(plasma['uniform_B_T'][:2]):
            raise ValueError('A uniform Cartesian transverse B is not axisymmetric')
        field = np.broadcast_to(plasma['uniform_B_T'], xyz.shape)

    tensor = cold_epsilon(source['frequency_Hz'], density, collision, field)
    epsilon = np.broadcast_to(radial_epsilon, (len(zc), *radial_epsilon.shape)).copy()
    epsilon[:, :plasma_cells] = tensor.reshape(len(zc), plasma_cells, 3, 3)
    return epsilon, {'directory': str(directory.resolve()), 'input_sha256': inputs,
                     'sampling': 'RF plasma cell centers at phi=0 in physical coordinates',
                     'basis': ['r', 'phi', 'z']}
