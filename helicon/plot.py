"""Plot saved FEM samples and write a readable run report, without MPI."""
import argparse
import html
from pathlib import Path
import numpy as np
from .config import load, rectangles
from .records import digest, read_json


def render(out):
    out=Path(out)
    c=load(out/'geometry.json'); m=read_json(out/'fem_summary.json')
    for filename,key in [('geometry.json','configuration_sha256'),('helicon.msh','mesh_sha256')]:
        if m[key] != digest(out/filename):
            raise ValueError(f'{filename} changed since the solve; rerun FEM before plotting')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    target=m['target_Bz_G']
    axis=np.atleast_1d(np.genfromtxt(out/'fem_axis.csv',delimiter=',',names=True))
    exit_data=np.atleast_1d(np.genfromtxt(out/'fem_exit.csv',delimiter=',',names=True))
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    axs[0].plot(axis['z_m']*1e3,axis['Bz_FEM_T']*1e4,label='FEM')
    axs[0].plot(axis['z_m']*1e3,axis['Bz_free_space_pack_T']*1e4,'--',label='Free-space reference')
    axs[0].axhline(target,color='gray',ls=':',label='Chamber-average target')
    axs[0].set(xlabel='z [mm]',ylabel='On-axis Bz [G]',title='Chamber centerline')
    for key,label in [('Bz_T','Bz'),('Br_T','Br'),('B_magnitude_T','|B|')]:
        axs[1].plot(exit_data['r_m']*1e3,exit_data[key]*1e4,label=label)
    axs[1].set(xlabel='r [mm]',ylabel='Field [G]',title='Chamber exit: radial profile')
    for ax in axs: ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out/'profiles.png',dpi=170); plt.close(fig)
    for name in ('device','full'):
        with np.load(out/f'field_{name}.npz') as f:
            r=f['r_m']*1e3; z=f['z_m']*1e3
            br=f['Br_T']; bz=f['Bz_T']; psi=f['psi_Wb_per_rad']
        fig,ax=plt.subplots(figsize=(7,8),layout='constrained')
        image=ax.pcolormesh(r,z,np.hypot(br,bz)*1e4,cmap='viridis',shading='auto')
        fig.colorbar(image,ax=ax,label='|B| [G]')
        span=float(np.ptp(psi))
        if span>1e-30:
            levels=np.linspace(float(psi.min()),float(psi.max()),25)[1:-1]
            ax.contour(r,z,psi,levels=levels,colors='white',linewidths=.55,alpha=.85)
        for label,tag,(x,y,w,h) in rectangles(c):
            color='#ffb347' if tag>=101 else '#ef4444' if tag==1 else '#cbd5e1'
            ax.add_patch(Rectangle((x*1e3,y*1e3),w*1e3,h*1e3,fill=False,
                                   edgecolor=color,linewidth=1.1,linestyle='--' if tag==3 else '-'))
        ax.set(xlabel='r [mm]',ylabel='z [mm]',xlim=(r[0],r[-1]),ylim=(z[0],z[-1]),
               title=f'FEM magnetic field — {name} view\nWhite curves: constant r Aθ (field lines)')
        ax.set_aspect('equal')
        fig.savefig(out/f'field_{name}.png',dpi=170); plt.close(fig)
    rows=[('Target chamber mean [G]',target),('Computed chamber mean [G]',m['chamber_mean_Bz_G']),
          ('On-axis exit Bz [G]',m['axis_exit_Bz_G']),('Exit area-average Bz [G]',m['exit_area_mean_Bz_G']),
          ('Chamber Bz standard deviation [G]',m['chamber_Bz_std_G']),
          ('Chamber Br RMS [G]',m['chamber_Br_rms_G']),
          ('Bz nonuniformity: std / |mean| [%]',m['chamber_Bz_nonuniformity_percent']),
          ('Target mean error [%]',m['target_mean_error_percent']),
          ('Axis reference error [%]',m['axis_reference_error_percent']),
          ('Normalized RMS field error (lower is better)',m['normalized_rms_field_error'])]
    fmt=lambda value: 'undefined (zero denominator)' if value is None else f'{value:.6g}'
    coils='\n'.join(f"- {k['name']}: {k['turns']} turns × {k['current']:g} A = {k['turns']*k['current']:g} A-turn; z={k['z_center']:g} m" for k in c['coils'])
    table='\n'.join(f'| {label} | {fmt(value)} |' for label,value in rows)
    notes=('The target is a chamber volume average. The on-axis exit value and exit area average are different metrics. '
           'Field lines are contours of r Aθ sampled on a visualization grid. Report metrics use the FEM solution, '
           'not image pixels. Positive current produces positive axial field. An axis reference check alone does not '
           'establish convergence throughout the chamber. No RF, plasma, coil heating, or thrust is modeled.')
    report=f'# Magnetostatic run: {out.name}\n\n{coils}\n\n| Quantity | Result |\n|---|---:|\n{table}\n\n{notes}\n\n'
    report+='![Profiles](profiles.png)\n\n![Device field](field_device.png)\n\n![Full field](field_full.png)\n'
    (out/'report.md').write_text(report)
    trs=''.join(f'<tr><td>{html.escape(label)}</td><td>{fmt(value)}</td></tr>' for label,value in rows)
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Helicon magnetic run</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#172435}table{border-collapse:collapse;width:100%}'
        'td{padding:9px;border-bottom:1px solid #ddd}td:last-child{text-align:right}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        f'<h1>Magnetostatic run: {html.escape(out.name)}</h1><pre>{html.escape(coils)}</pre><table>{trs}</table>'
        f'<p>{html.escape(notes)}</p><img src="profiles.png" alt="Axial and radial profiles">'
        '<img src="field_device.png" alt="Device field map"><img src="field_full.png" alt="Full-domain field map"></html>')
    print(f'Chamber mean: {m["chamber_mean_Bz_G"]:.3f} G; axis exit: {m["axis_exit_Bz_G"]:.3f} G; '
          f'exit area mean: {m["exit_area_mean_Bz_G"]:.3f} G. Report: {out/"report.html"}')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--build',default='build')
    render(p.parse_args().build)

if __name__=='__main__': main()
