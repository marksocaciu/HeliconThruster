"""Strict, SI-unit RF inputs independent of the legacy geometry configuration."""
import json
import math
from pathlib import Path
from ..config import ROOT


def validate(c):
    keys = {'model','frequency_Hz','voltage_peak_V','material_case','chamber','antenna',
            'domain','mesh','dielectric','plasma','solver','description'}
    if set(c) != keys or c['model'] != 'rf_gap_3d_v1':
        raise ValueError('Unexpected RF configuration keys or model version')
    def fields(section, positive, nonnegative=()):
        obj=c[section]
        if set(obj) != set(positive)|set(nonnegative):
            raise ValueError(f'Unexpected {section} keys')
        for key in obj:
            number(obj[key], f'{section}.{key}', key in positive)
    def number(x, name, positive=True):
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 or (positive and x==0):
            raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    number(c['frequency_Hz'],'frequency_Hz'); number(c['voltage_peak_V'],'voltage_peak_V')
    if c['material_case'] not in ('vacuum','dielectric','plasma'):
        raise ValueError('material_case must be vacuum, dielectric or plasma')
    for section, signed in [('chamber',('z_start_m',)),('domain',('z_min_m','z_max_m'))]:
        for key in signed:
            x=c[section][key]
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
                raise ValueError(f'{section}.{key} must be finite')
    expected={'chamber':{'radius_m','length_m','z_start_m','wall_thickness_m'},
              'domain':{'half_width_m','z_min_m','z_max_m','pml_thickness_m','pml_strength'}}
    for section in expected:
        if set(c[section]) != expected[section]: raise ValueError(f'Unexpected {section} keys')
        for key in c[section]:
            if key not in ('z_start_m','z_min_m','z_max_m'): number(c[section][key],key)
    fields('antenna',('inner_radius_m','radial_thickness_m','ring_width_m','leg_angle_rad','gap_m'))
    mesh_required = {'bulk_m','plasma_m','wall_m','antenna_m','gap_m'}
    mesh_optional = {'interface_m','interface_band_m','interface_transition_m'}
    antenna_optional = {'antenna_band_m','antenna_transition_m'}
    if set(c['mesh']) & antenna_optional and not antenna_optional <= set(c['mesh']):
        raise ValueError('Both antenna band parameters are required together')
    if not mesh_required <= set(c['mesh']) or set(c['mesh']) - mesh_required - mesh_optional - antenna_optional:
        raise ValueError('Unexpected mesh keys')
    if set(c['mesh']) & mesh_optional and not mesh_optional <= set(c['mesh']):
        raise ValueError('All three interface refinement parameters are required together')
    for key, value in c['mesh'].items(): number(value, 'mesh.'+key)
    if antenna_optional <= set(c['mesh']) and c['mesh']['antenna_transition_m'] <= c['mesh']['antenna_band_m']:
        raise ValueError('antenna_transition_m must exceed antenna_band_m')
    if mesh_optional <= set(c['mesh']):
        if c['mesh']['interface_transition_m'] <= c['mesh']['interface_band_m']:
            raise ValueError('interface_transition_m must exceed interface_band_m')
        if c['mesh']['interface_m'] > min(c['mesh']['plasma_m'], c['mesh']['wall_m']):
            raise ValueError('interface_m must not exceed plasma_m or wall_m')
    fields('dielectric',('wall_epsilon_r','test_epsilon_r'),('wall_loss_tangent','test_loss_tangent'))
    if set(c['plasma']) != {'electron_density_m3','electron_temperature_eV','neutral_density_m3','collision_frequency_Hz','uniform_B_T'}:
        raise ValueError('Unexpected plasma keys')
    for key in ('electron_density_m3','electron_temperature_eV','neutral_density_m3','collision_frequency_Hz'):
        number(c['plasma'][key],key)
    b=c['plasma']['uniform_B_T']
    if not isinstance(b,list) or len(b)!=3 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in b):
        raise ValueError('uniform_B_T must contain three finite Cartesian components')
    fields('solver',('degree','power_balance_tolerance','relative_residual_tolerance'))
    if c['solver']['degree'] not in (1,2): raise ValueError('Only Nedelec degree 1 or 2 is supported')
    if c['solver']['power_balance_tolerance'] >= .1 or c['solver']['relative_residual_tolerance'] >= .01:
        raise ValueError('Solver tolerances are too loose')
    h,a,d=c['chamber'],c['antenna'],c['domain']
    if a['inner_radius_m'] <= h['radius_m']+h['wall_thickness_m']: raise ValueError('Antenna must clear the dielectric tube')
    if a['inner_radius_m']+a['radial_thickness_m'] >= d['half_width_m']: raise ValueError('Antenna must lie inside the non-PML domain')
    if 2*a['ring_width_m']+a['gap_m'] >= h['length_m']: raise ValueError('Rings and gap overlap')
    if a['leg_angle_rad'] >= math.pi/2: raise ValueError('Antenna legs must be separated')
    if d['z_min_m'] >= h['z_start_m'] or d['z_max_m'] <= h['z_start_m']+h['length_m']: raise ValueError('Chamber must lie inside the non-PML domain')
    if not isinstance(c['description'],str): raise ValueError('description must be text')


def load(path=None):
    c=json.loads(Path(path or ROOT/'config/rf_nagoya.json').read_text())
    validate(c)
    return c
