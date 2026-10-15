"""Summarize completed RF runs and plot saved cell samples on Mac or Linux."""
import argparse
from html import escape
from pathlib import Path
import numpy as np
from ..records import read_json,digest


def render(out):
    out=Path(out); r=read_json(out/'rf_summary.json'); record=read_json(out/'run.json')
    for name,sha in record['artifacts_sha256'].items():
        if digest(out/name)!=sha: raise ValueError('Changed RF result artifact: '+name)
    arrays=[]
    for path in sorted(out.glob('cells_rank*.npz')):
        with np.load(path,allow_pickle=False) as d: arrays.append({k:d[k].copy() for k in d.files})
    if not arrays: raise ValueError('Missing RF cell samples')
    points=np.concatenate([a['xyz_m'] for a in arrays]); tag=np.concatenate([a['region'] for a in arrays])
    electric=np.concatenate([a['E_peak_V_m'] for a in arrays]); heating=np.concatenate([a['q_abs_W_m3'] for a in arrays])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=read_json(out/'rf.json'); width=c['mesh']['plasma_m']*read_json(out/'mesh_report.json')['mesh_scale']
    use=(np.abs(points[:,1])<width/2)&(tag!=7)
    if np.count_nonzero(use)<3: raise ValueError('Too few cell samples for central slab plot')
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    values=[np.linalg.norm(electric,axis=1),heating]
    for ax,val,title in zip(axes,values,('Electric phasor norm (V/m)','Material absorption (W/m³)')):
        idx=use if ax is axes[0] else use&(tag==1)
        color=ax.scatter(points[idx,2],points[idx,0],c=val[idx],s=5,cmap='viridis')
        fig.colorbar(color,ax=ax);ax.set(xlabel='z (m)',ylabel='x (m)',title=title);ax.set_aspect('equal')
    fig.suptitle('RF cell samples in a central slab — not a converged field map')
    fig.savefig(out/'rf_fields.png',dpi=160);plt.close(fig)
    rows=[('Case',r['case']),('Frequency (MHz)',r['frequency_Hz']/1e6),('Input power (W)',r['input_power_W']),
          ('Plasma/test-medium absorption (W)',r['material_absorption_W']['PLASMA']),('PML absorption (W)',r['pml_absorption_W']),
          ('Outward interface flux (W)',r['outward_interface_flux_W']),('Power balance relative error',r['power_balance_relative_error']),
          ('Free equation residual',r['free_equation_relative_residual']),('Numerical checks passed',r['numerical_checks_passed'])]
    table='<table>'+''.join('<tr><th>'+escape(str(k))+'</th><td>'+escape(str(v))+'</td></tr>' for k,v in rows)+'</table>'
    html='<!doctype html><html lang="en"><meta charset="utf-8"><title>RF prototype report</title><style>body{max-width:1100px;margin:40px auto;font:16px system-ui;padding:20px}th,td{padding:8px;text-align:left}img{width:100%}</style><h1>Frequency-domain RF prototype</h1>'
    html+='<p>Numerical checks are not mesh/PML convergence or experimental validation.</p>'+table+'<img src="rf_fields.png" alt="RF field and absorption cell samples">'
    html+='<ul>'+''.join('<li>'+escape(x)+'</li>' for x in r['limitations'])+'</ul><a href="rf_summary.json">Full result</a></html>'
    (out/'report.html').write_text(html)
    print(out/'report.html')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args();render(a.run)

if __name__=='__main__': main()
