"""Reconstruct local plasma budgets from saved continuation states; no solve."""
import argparse
import csv
import html
from pathlib import Path
import sys

import numpy as np

from . import spatial
from .argon import TE_MIN_EV, TE_MAX_EV
from .records import digest, read_json, write_json, utc_now

ENERGY_LOSSES = ('ionization_loss_W', 'excitation_loss_W', 'elastic_exchange_W',
                 'conduction_out_W', 'enthalpy_out_W', 'boundary_out_W')


def reconstruct(geometry, config, summary, fields):
    spatial.validate(config)
    if summary.get('model') != spatial.MODEL or summary.get('input') != config:
        raise ValueError('Saved summary and resolved transport configuration differ')
    g = spatial.Grid(geometry, config['nr'], config['nz'])
    for key, expected in [('r_faces_m',g.rf), ('z_faces_m',g.zf), ('cell_volume_m3',g.volumes)]:
        actual = fields[key]
        if actual.shape != expected.shape or not np.allclose(actual,expected,rtol=1e-12,atol=1e-15):
            raise ValueError('Saved grid differs from configured geometry: '+key)
    keys = ('electron_density_m3','neutral_density_m3','electron_temperature_eV',
            'potential_V','q_abs_W_m3','collision_frequency_s')
    for key in keys:
        if fields[key].shape != g.shape or not np.isfinite(fields[key]).all():
            raise ValueError('Invalid saved cell field: '+key)
    ne, nn, te, phi = [fields[k].ravel() for k in keys[:4]]
    if np.any(ne <= 0) or np.any(nn <= 0): raise ValueError('Nonpositive saved density')
    x = np.array([np.log(ne/1e18),np.log(nn/1e19),te,phi])
    eq = spatial.Equations(g,config,fields['q_abs_W_m3'])
    a = eq.evaluate(x,diagnostics=True)
    e, p = a['energy_budget'],a['particle_budget']
    if not np.allclose(a['nu'].reshape(g.shape), fields['collision_frequency_s'],rtol=1e-10,atol=0):
        raise ValueError('Saved collision state differs from diagnostic equations')
    normalized = {key: a[key]/scale for key,scale in (
        ('ion_res',eq.particle_scale), ('neutral_res',eq.particle_scale),
        ('current_res',eq.particle_scale), ('energy_res',eq.energy_scale))}
    recomputed = {k:float(np.max(np.abs(v))) for k,v in normalized.items()}
    saved = summary['balances']['max_local_relative_residuals']
    if any(not np.isclose(recomputed[k],saved[k],rtol=1e-6,atol=1e-9) for k in recomputed):
        raise ValueError('Reconstructed residuals differ from saved solve; check code/config provenance')
    sums = {k:float(v.sum()) for k,v in e.items()}
    b = summary['balances']
    for value,expected in ((eq.power,b['absorbed_power_W']),
        (a['inelastic'].sum(),b['inelastic_power_W']),
        (a['boundary_energy'].sum(),b['electron_boundary_power_W']),
        (a['work'].sum(),b['electron_electrostatic_work_W'])):
        if not np.isclose(value,expected,rtol=1e-10,atol=eq.power*1e-12):
            raise ValueError('Reconstructed integrated budget differs from saved solve')
    exit_faces = g.boundary_kind == 2
    mass_error = abs(eq.feed-a['boundary_neutrals'][exit_faces].sum()
        -a['boundary_ions'][exit_faces].sum()
        -(1-config['wall_recycling_fraction'])*a['boundary_ions'][~exit_faces].sum())/eq.feed
    power_error = abs(eq.power-a['inelastic'].sum()-a['boundary_energy'].sum()+a['work'].sum())/eq.power
    if any(not np.isclose(value,expected,rtol=1e-6,atol=1e-9) for value,expected in (
            (mass_error,b['relative_mass_error']),(power_error,b['relative_power_error']))):
        raise ValueError('Reconstructed integrated conservation residuals differ from saved solve')
    residual = sum(e[k] for k in ENERGY_LOSSES)-e['rf_heating_W']-e['electrostatic_work_W']
    # Scaled errors remain meaningful for both accepted and rejected states.
    energy_error = float(np.max(np.abs(residual-a['energy_res']))/eq.power)
    particle_error = max(float(np.max(np.abs(p['ion_internal_out_s']+p['boundary_ions_out_s']-
        p['ionization_events_s']-p['ion_residual_s']))/eq.feed),
        float(np.max(np.abs(p['neutral_internal_out_s']+p['boundary_neutrals_out_s']+
        p['ionization_events_s']-p['neutral_residual_s']))/eq.feed),
        float(np.max(np.abs(p['ion_internal_out_s']-p['electron_internal_out_s']-
        p['current_residual_particles_s']))/eq.feed))
    cancellation = {k:abs(sums[k])/max(float(np.sum(np.abs(e[k]))),eq.power)
                    for k in ('conduction_out_W','enthalpy_out_W')}
    if max(energy_error,particle_error,*cancellation.values()) > 1e-10:
        raise ValueError('Local budget reconstruction or internal face cancellation failed')
    rows = []
    for i in range(g.size):
        iz,ir = divmod(i,config['nr'])
        row = {'cell':i,'iz':iz,'ir':ir,'r_m':float(g.r[ir]),'z_m':float(g.z[iz]),
               'Te_eV':float(te[i]),'ne_m3':float(ne[i]),'nn_m3':float(nn[i]),
               'potential_V':float(phi[i]),'volume_m3':float(g.volumes.ravel()[i]),
               'Te_upper_margin_eV':float(TE_MAX_EV-te[i]),
               'Te_lower_margin_eV':float(te[i]-TE_MIN_EV),
               'Te_near_bound':bool(min(TE_MAX_EV-te[i],te[i]-TE_MIN_EV)<.01),
               'potential_bound_margin_V':float(500-abs(phi[i])),
               'ne_log_bound_margin':float(20-abs(x[0,i])),
               'nn_log_bound_margin':float(20-abs(x[1,i]))}
        row.update({k:float(v[i]) for k,v in e.items()})
        row.update({k:float(v[i]) for k,v in p.items()})
        row.update({k+'_relative':float(v[i]) for k,v in normalized.items()})
        rows.append(row)
    info = {'saved_status':summary['status'], 'diagnostic_only':True,
            'reconstructed_max_local_residuals':recomputed,
            'global_energy_terms_W':sums,
            'relative_mass_error':float(mass_error),'relative_power_error':float(power_error),
            'energy_reconstruction_error_relative_to_power':energy_error,
            'particle_reconstruction_error_relative_to_feed':particle_error,
            'internal_energy_cancellation_relative':cancellation,
            'hottest_cells':sorted(rows,key=lambda r:r['Te_eV'],reverse=True)[:3],
            'worst_cells_by_equation':{k: max(rows,key=lambda r:abs(r[k+'_relative']))
                                       for k in normalized}}
    return g,rows,info,a


def save_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def plot(directory,grid,fields,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entries = [('Te [eV]',fields['electron_temperature_eV'])]
    entries += [(k,v.reshape(grid.shape)) for k,v in a['energy_budget'].items()]
    fig,axes=plt.subplots(2,5,figsize=(19,7),constrained_layout=True)
    for ax,(label,values) in zip(axes.flat,entries):
        im=ax.pcolormesh(grid.zf*100,grid.rf*100,values.T,shading='flat',cmap='coolwarm')
        ax.set(title=label,xlabel='z [cm]',ylabel='r [cm]')
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Reconstructed saved-state energy budget: W per cell except Te')
    fig.savefig(directory/'energy_budget.png',dpi=140);plt.close(fig)


def compare(continuation,out,plots=True):
    continuation,out=Path(continuation).resolve(),Path(out)
    parent=read_json(continuation/'continuation_summary.json')
    history=parent.get('history',[])
    rejected=next((i for i,row in enumerate(history) if row['accepted'] is False),None)
    if rejected is None or rejected==0:
        raise ValueError('Require a rejected candidate preceded by an accepted state')
    pair=history[rejected-1:rejected+1]
    if pair[0]['accepted'] is not True:raise ValueError('Previous state was not accepted')
    config=read_json(continuation/'plasma_config.json')
    geometry=read_json(continuation/'geometry.json')
    paths=[continuation/name for name in ('continuation_summary.json','plasma_config.json','geometry.json')]
    for row in pair:
        folder=(continuation/row['directory']).resolve()
        if continuation not in folder.parents:raise ValueError('State directory escapes continuation run')
        paths.extend([folder/'spatial_summary.json',folder/'plasma_fields.npz'])
    hashes={str(p.relative_to(continuation)):digest(p) for p in paths}
    # Completed continuation commands hash state exports even when a candidate failed.
    manifest=parent.get('artifacts_sha256',{})
    for name,sha in hashes.items():
        if name != 'continuation_summary.json' and manifest.get(name) != sha:
            raise ValueError('Saved continuation artifact hash missing or changed: '+name)
    out.mkdir(parents=True,exist_ok=False)
    record={'model':'spatial_budget_comparison_v1','status':'running','started_utc':utc_now(),
            'continuation':str(continuation),'input_sha256':hashes,
            'note':'Re-evaluation of saved states with current diagnostic equations; no plasma or RF solve. '
                   'Rejected state budgets describe an unbalanced iterate, not a physical solution.'}
    write_json(out/'run.json',record)
    try:
        source=out/'source';source.mkdir()
        import shutil
        for path in (Path(__file__),Path(spatial.__file__),Path(__file__).with_name('argon.py')):
            shutil.copy2(path,source/path.name)
        record['diagnostic_source_sha256']={p.name:digest(p) for p in source.iterdir()}
        record['original_solver_source_sha256']=parent.get('source',{}).get('sha256',{})
        cases={};allrows={}
        for label,row in zip(('accepted','rejected'),pair):
            folder=continuation/row['directory']
            summary=read_json(folder/'spatial_summary.json')
            with np.load(folder/'plasma_fields.npz',allow_pickle=False) as d:
                fields={k:d[k].copy() for k in d.files}
            grid,rows,info,a=reconstruct(geometry,config,summary,fields)
            accepted=summary['status']=='converged' and max(
                *info['reconstructed_max_local_residuals'].values(),
                info['relative_mass_error'],info['relative_power_error'])<config['residual_tolerance']
            if accepted != row['accepted']:
                raise ValueError('History acceptance disagrees with reconstructed transport checks')
            directory=out/label;directory.mkdir()
            save_csv(directory/'cells.csv',rows)
            np.savez_compressed(directory/'cell_budget.npz',**{
                **a['energy_budget'],**a['particle_budget'],'r_faces_m':grid.rf,'z_faces_m':grid.zf})
            np.savez_compressed(directory/'face_budget.npz',**a['face_budget'])
            info['alpha']=row['alpha'];cases[label]=info;allrows[label]=rows
            write_json(directory/'budget_summary.json',info)
            if plots:plot(directory,grid,fields,a)
        changes=[]
        for old,new in zip(allrows['accepted'],allrows['rejected']):
            change={k:new[k] for k in ('cell','iz','ir','r_m','z_m')}
            change.update({k+'_delta':new[k]-old[k] for k in new
                           if k not in change and not isinstance(new[k],bool)})
            changes.append(change)
        save_csv(out/'cell_changes.csv',changes)
        if any(digest(continuation/name)!=sha for name,sha in hashes.items()):
            raise ValueError('Saved inputs changed during diagnostic reconstruction')
        selected=cases['rejected']['hottest_cells'][0]['cell']
        limiting={label:rows[selected] for label,rows in allrows.items()}
        result={'status':'completed','cases':cases,'note':record['note'],
                'limiting_cell_comparison':limiting,
                'sign_convention':'Residual = ionization + excitation + elastic + net conduction out '
                 '+ net enthalpy out + boundary out - RF heating - electrostatic work. '
                 'Positive residual means excess modeled loss/net outflow; negative means excess source. '
                 'Internal transport can be negative (net inflow). Particle residuals are particles/s; '
                 'current residual uses ion minus electron particle flux, not amperes.',
                'bound_reporting_tolerance_eV':.01}
        write_json(out/'comparison_summary.json',result)
        table=''
        for label in cases:
            hottest=cases[label]['hottest_cells'][0]
            table+=f'<tr><td>{label}</td><td>{cases[label]["alpha"]}</td><td>{hottest["Te_eV"]:.7g}</td>'
            table+=f'<td>{hottest["r_m"]*1000:.4g}</td><td>{hottest["z_m"]*1000:.4g}</td></tr>'
        figures=''.join(f'<h2>{label}</h2><img src="{label}/energy_budget.png" alt="{label} energy budget">'
                        for label in cases) if plots else ''
        budget_table=''.join(f'<tr><td>{key}</td><td>{limiting["accepted"][key]:.7g}</td>'
            f'<td>{limiting["rejected"][key]:.7g}</td>'
            f'<td>{limiting["rejected"][key]-limiting["accepted"][key]:.7g}</td></tr>'
            for key in ('rf_heating_W',*ENERGY_LOSSES,'electrostatic_work_W','energy_residual_W'))
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8">'
            '<title>Local plasma budget comparison</title><style>body{font:16px system-ui;'
            'margin:30px auto;max-width:1500px;padding:20px}td,th{padding:8px}img{width:100%}</style>'
            '<h1>Local plasma budget comparison</h1><p>'+html.escape(result['note'])+'</p><p>'
            +html.escape(result['sign_convention'])+'</p><table><tr><th>State</th><th>Alpha</th>'
            '<th>Hottest Te [eV]</th><th>r [mm]</th><th>z [mm]</th></tr>'+table+'</table>'
            '<h2>Energy budget at the rejected state\'s hottest cell</h2>'
            '<table><tr><th>Term [W per cell]</th><th>Accepted</th><th>Rejected</th>'
            '<th>Change</th></tr>'+budget_table+'</table>'
            '<p><a href="comparison_summary.json">Budget summaries and limiting cells</a> · '
            '<a href="accepted/cells.csv">Accepted cell table</a> · '
            '<a href="rejected/cells.csv">Rejected cell table</a> · '
            '<a href="cell_changes.csv">Cell-by-cell changes</a></p>'+figures)
        record.update(status='completed',finished_utc=utc_now())
        write_json(out/'run.json',record)
        return result
    except (Exception,KeyboardInterrupt) as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_utc=utc_now())
        write_json(out/'run.json',record);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continuation',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    try:result=compare(args.continuation,args.out,not args.no_plots)
    except (Exception,KeyboardInterrupt) as exc:
        print(f'Budget diagnostic stopped: {exc}',file=sys.stderr);return 2
    for label,case in result['cases'].items():
        cell=case['hottest_cells'][0]
        print(f"{label}: alpha={case['alpha']} hottest Te={cell['Te_eV']:.7g} eV "
              f"at r={cell['r_m']*1000:.4g}, z={cell['z_m']*1000:.4g} mm")
        for equation,worst in case['worst_cells_by_equation'].items():
            print(f"  {equation}: max |relative residual|={abs(worst[equation+'_relative']):.5g} "
                  f"at r={worst['r_m']*1000:.4g}, z={worst['z_m']*1000:.4g} mm")
    print('Energy at rejected-state hottest cell [W]: accepted -> rejected')
    pair=result['limiting_cell_comparison']
    for key in ('rf_heating_W',*ENERGY_LOSSES,'electrostatic_work_W','energy_residual_W'):
        print(f"  {key}: {pair['accepted'][key]:.7g} -> {pair['rejected'][key]:.7g}")
    print(f'Budget diagnostic completed: {args.out}/report.html')
    return 0


if __name__=='__main__':sys.exit(main())
