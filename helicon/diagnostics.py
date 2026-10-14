"""Metrics independent of FEM, with explicit normalization and SI units."""
import math
import numpy as np


def positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def design_metrics(mean, std, radial_rms, target_t):
    positive(target_t, 'target field')
    # Equal weighting of axial target error and radial field, volume averaged.
    score = math.sqrt((mean-target_t)**2 + std**2 + radial_rms**2) / target_t
    return {'target_Bz_T': target_t, 'target_Bz_G': target_t*1e4,
            'chamber_mean_Bz_G': mean*1e4,
            'chamber_Bz_std_G': std*1e4, 'chamber_Br_rms_G': radial_rms*1e4,
            'target_mean_error_percent': 100*(mean-target_t)/target_t,
            'chamber_Bz_nonuniformity_percent': 100*std/abs(mean) if abs(mean)>1e-30 else None,
            'chamber_Br_fraction_percent': 100*radial_rms/abs(mean) if abs(mean)>1e-30 else None,
            'normalized_rms_field_error': score}


def reconstruct_field(radius, potential, gradient):
    """Gradient columns are dA/dr,dA/dz. Apply regular limits at r=0."""
    radius = np.asarray(radius)
    potential = np.asarray(potential)
    gradient = np.asarray(gradient)
    br = -gradient[:, 1].copy()
    bz = gradient[:, 0].copy()
    axis = radius == 0
    bz[~axis] += potential[~axis] / radius[~axis]
    bz[axis] = 2*gradient[axis, 0]
    br[axis] = 0
    return br, bz


def disk_mean(values, radii, gauss_weights, radius):
    """Disk mean with Gauss nodes mapped to [0,R]; weights remain on [-1,1]."""
    return float(np.sum(np.asarray(gauss_weights)*np.asarray(radii)*np.asarray(values))/radius)
