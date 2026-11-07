"""Validated interpolation of an accepted uniform-heating state to a new grid.

This transfers an initial guess only; the target solution must pass normal checks.
"""
from pathlib import Path
import shutil
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import spatial,argon_chemistry as chemistry
from .records import read_json,write_json,digest
from .spatial_budget import reconstruct


def interpolate(fields,source_grid,target_grid,config):
    """Linear r-z interpolation; log positive fields, constant edge-center extension."""
    z,r=np.meshgrid(target_grid.z,target_grid.r,indexing='ij')
    points=np.column_stack((np.clip(z.ravel(),source_grid.z[0],source_grid.z[-1]),
                            np.clip(r.ravel(),source_grid.r[0],source_grid.r[-1])))
    state={'r_m':target_grid.r,'z_m':target_grid.z,'r_faces_m':target_grid.rf,
           'z_faces_m':target_grid.zf,'cell_volume_m3':target_grid.volumes,
           'q_abs_W_m3':np.full(target_grid.shape,config['absorbed_power_W']/target_grid.volumes.sum())}
    positive=('electron_density_m3','neutral_density_m3','electron_temperature_eV')
    for key in (*positive,'potential_V'):
        values=np.asarray(fields[key],float)
        if values.shape!=source_grid.shape or not np.isfinite(values).all():
            raise ValueError('Invalid source state field '+key)
        if key in positive:
            if np.any(values<=0):raise ValueError('Nonpositive source state field '+key)
            values=np.log(values)
        value=RegularGridInterpolator((source_grid.z,source_grid.r),values,
            method='linear',bounds_error=True)(points).reshape(target_grid.shape)
        state[key]=np.exp(value) if key in positive else value
    state['potential_V']-=state['potential_V'].flat[0]
    _,nu,_=spatial.coefficients(state['electron_temperature_eV'].ravel(),
                                state['neutral_density_m3'].ravel(),config)
    state['collision_frequency_s']=nu.reshape(target_grid.shape)
    if chemistry.model(config)!=chemistry.LEGACY:state['chemistry_model']=np.array(chemistry.model(config))
    low,high=chemistry.bounds(config)
    te=state['electron_temperature_eV']
    if np.any(te<low) or np.any(te>high) or np.any(abs(state['potential_V'])>500):
        raise ValueError('Interpolated state is outside target solver bounds')
    for key,scale in (('electron_density_m3',1e18),('neutral_density_m3',1e19)):
        if np.any(abs(np.log(state[key]/scale))>20):raise ValueError('Interpolated density exceeds solver bounds')
    return state


def residuals(grid,config,state):
    eq=spatial.Equations(grid,config,state['q_abs_W_m3'])
    x=np.array([np.log(state['electron_density_m3']/1e18).ravel(),
        np.log(state['neutral_density_m3']/1e19).ravel(),
        state['electron_temperature_eV'].ravel(),state['potential_V'].ravel()])
    a=eq.evaluate(x)
    return {key:float(np.max(abs(a[key]/scale))) for key,scale in (
        ('ion_res',eq.particle_scale),('neutral_res',eq.particle_scale),
        ('current_res',eq.particle_scale),('energy_res',eq.energy_scale))}


def load_seed(directory,geometry,config,out):
    """Require a hash-verified accepted alpha=0 state with matching physical inputs."""
    directory=Path(directory).resolve();out=Path(out)
    parent=next((p for p in directory.parents if (p/'continuation_summary.json').is_file()),None)
    if parent is None:raise ValueError('Initial state must be a saved continuation step directory')
    record=read_json(parent/'continuation_summary.json')
    relative=str(directory.relative_to(parent))
    entry=next((row for row in record['history'] if row['directory']==relative),None)
    if entry is None or entry['accepted'] is not True or entry['alpha']!=0:
        raise ValueError('Initial state must be the accepted alpha=0 uniform-heating step')
    paths=[parent/'plasma_config.json',parent/'geometry.json',directory/'plasma_fields.npz',
           directory/'spatial_summary.json',parent/'continuation_summary.json']
    hashes={str(p):digest(p) for p in paths}
    manifest=record.get('artifacts_sha256',{})
    for path in paths[:-1]:
        if manifest.get(str(path.relative_to(parent)))!=hashes[str(path)]:
            raise ValueError('Initial-state source artifact hash mismatch: '+str(path))
    source_config=spatial.load(parent/'plasma_config.json')
    source_geometry=read_json(parent/'geometry.json')
    if source_geometry!=geometry:raise ValueError('Initial-state and target geometry differ')
    def strip(c):
        result={k:v for k,v in c.items() if k not in ('nr','nz')}
        result['chemistry_model']=chemistry.model(c)
        return result
    if strip(source_config)!=strip(config):
        raise ValueError('Initial-state and target configuration differ beyond grid dimensions')
    summary=read_json(directory/'spatial_summary.json')
    with np.load(directory/'plasma_fields.npz',allow_pickle=False) as data:
        fields={key:data[key].copy() for key in data.files}
    g,_,checks,_=reconstruct(geometry,source_config,summary,fields)
    tol=source_config['residual_tolerance']
    if summary['status']!='converged' or max(*checks['reconstructed_max_local_residuals'].values(),
            checks['relative_mass_error'],checks['relative_power_error'])>=tol:
        raise ValueError('Initial state fails reconstructed source conservation checks')
    expected=source_config['absorbed_power_W']/g.volumes.sum()
    if not np.allclose(fields['q_abs_W_m3'],expected,rtol=1e-12,atol=0):
        raise ValueError('Initial state does not have uniform heating at the requested power')
    grid=spatial.Grid(geometry,config['nr'],config['nz'])
    if not (np.array_equal(g.rf[[0,-1]],grid.rf[[0,-1]]) and
            np.array_equal(g.zf[[0,-1]],grid.zf[[0,-1]])):
        raise ValueError('Initial and target domains differ')
    state=interpolate(fields,g,grid,config)
    info={'source_step':str(directory),'source_alpha':0.,'source_grid':[source_config['nr'],source_config['nz']],
        'target_grid':[config['nr'],config['nz']],'input_sha256':hashes,
        'method':'Linear interpolation of log(ne), log(nn), log(Te); linear potential; gauge restored.',
        'edge_treatment':'Constant extension from nearest coarse cell center inside the same physical domain.',
        'collision_frequency':'Recomputed from target chemistry and interpolated thermochemical state.',
        'note':'Initial guess only; it is not a target-grid accepted solution or a conservative state remap.',
        'source_checks':checks['reconstructed_max_local_residuals'],
        'target_initial_relative_residuals':residuals(grid,config,state)}
    out.mkdir(parents=True,exist_ok=False)
    names=('plasma_config.json','geometry.json','source_plasma_fields.npz',
           'source_spatial_summary.json','source_continuation_summary.json')
    for path,name in zip(paths,names):shutil.copy2(path,out/name)
    verify_source(info)
    np.savez_compressed(out/'interpolated_initial_state.npz',**state)
    write_json(out/'initialization.json',info)
    return state,info


def verify_source(info):
    if any(digest(path)!=sha for path,sha in info['input_sha256'].items()):
        raise ValueError('Initial-state source changed during the experiment')
