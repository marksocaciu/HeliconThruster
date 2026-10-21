"""Conservative RF heating binning and bounded axisymmetric material lookup."""
from pathlib import Path
import numpy as np
from .mesh import CELL


class PlasmaMap:
    """Piecewise-constant FV state, including axis and boundary half cells.

    Cell faces define support. Points outside are rejected, never extrapolated.
    """
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.rf, self.zf = data['r_faces_m'].copy(), data['z_faces_m'].copy()
            self.fields = {key: data[key].copy() for key in (
                'electron_density_m3', 'neutral_density_m3',
                'electron_temperature_eV', 'collision_frequency_s')}
        for faces in (self.rf, self.zf):
            if faces.ndim != 1 or len(faces) < 3 or not np.isfinite(faces).all() or np.any(np.diff(faces) <= 0):
                raise ValueError('Invalid plasma-map cell faces')
        if self.rf[0] != 0:
            raise ValueError('Plasma-map radial faces must start at zero')
        shape = (len(self.zf)-1, len(self.rf)-1)
        for key, values in self.fields.items():
            if values.shape != shape or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f'Invalid plasma-map field: {key}')
        te = self.fields['electron_temperature_eV']
        if np.any(te < 1) or np.any(te > 7):
            raise ValueError('Plasma-map Te must lie in 1–7 eV')

    def __call__(self, xyz):
        ir, iz = bin_indices(xyz, self.rf, self.zf)
        return {key: values[iz, ir] for key, values in self.fields.items()}


def bin_indices(xyz, rf, zf):
    xyz = np.asarray(xyz, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Points must be finite Cartesian coordinates')
    r, z = np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2]
    tol = 1e-12
    if np.any(r > rf[-1]+tol) or np.any(z < zf[0]-tol) or np.any(z > zf[-1]+tol):
        raise ValueError('RF plasma cells lie outside the spatial plasma domain')
    ir = np.clip(np.searchsorted(rf, r, side='right')-1, 0, len(rf)-2)
    iz = np.clip(np.searchsorted(zf, z, side='right')-1, 0, len(zf)-2)
    return ir, iz


def heating_from_rf(directory, grid, expected_power_W, tolerance=1e-6):
    """Deposit quadrature-integrated tetrahedron powers into r-z control volumes.

    Centroid assignment preserves total power, not subcell shape. No global
    rescaling conceals a transfer error. Unoccupied bins are explicitly reported.
    """
    power = np.zeros(grid.shape)
    sampled_volume = np.zeros(grid.shape)
    paths = sorted(Path(directory).glob('cells_rank*.npz'))
    if not paths or not np.isfinite(expected_power_W) or expected_power_W <= 0:
        raise ValueError('Require RF cell exports and positive plasma absorption')
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if not {'cell_absorbed_power_W', 'cell_volume_m3'} <= set(data.files):
                raise ValueError('RF export lacks integrated cell powers; rerun the updated RF solver')
            mask = data['region'] == CELL['PLASMA']
            ir, iz = bin_indices(data['xyz_m'][mask], grid.rf, grid.zf)
            p, v = data['cell_absorbed_power_W'][mask], data['cell_volume_m3'][mask]
            if not np.isfinite(p).all() or not np.isfinite(v).all() or np.any(v <= 0) or np.any(p < 0):
                raise ValueError('Invalid/nonpassive RF plasma cell integrals')
            np.add.at(power, (iz, ir), p)
            np.add.at(sampled_volume, (iz, ir), v)
    error = abs(power.sum()-expected_power_W)/expected_power_W
    if error > tolerance:
        raise ValueError(f'RF-to-plasma heating transfer does not conserve power: {error:g}')
    # Compare faceted RF volume with ideal cylindrical FV volume, without correction.
    info = {'integrated_power_W': float(power.sum()), 'relative_power_error': float(error),
            'occupied_bin_fraction': float(np.mean(sampled_volume > 0)),
            'rf_plasma_volume_m3': float(sampled_volume.sum()),
            'fv_plasma_volume_m3': float(grid.volumes.sum()),
            'method': 'integrated tetrahedron power assigned by centroid; azimuthally summed'}
    return power/grid.volumes, info
