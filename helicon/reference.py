"""Independent on-axis Biot–Savart reference, including finite winding packs."""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from .config import load

MU0 = 4e-7*np.pi


def loop_bz(z, radius, z_center, ampere_turns):
    z = np.asarray(z,dtype=float)
    return MU0*ampere_turns*radius**2/(2*(radius**2+(z-z_center)**2)**1.5)


def pack_bz(z, coils, order=16):
    """Integrate circular loops over a uniform rectangular winding cross-section."""
    z = np.asarray(z,dtype=float)
    result = np.zeros_like(z)
    nodes, weights = leggauss(order)
    for coil in coils:
        for x,wx in zip(nodes,weights):
            radius = coil['radius'] + x*coil['radial_thickness']/2
            for y,wy in zip(nodes,weights):
                center = coil['z_center'] + y*coil['axial_width']/2
                result += loop_bz(z,radius,center,coil['turns']*coil['current']*wx*wy/4)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--out',default='build')
    a=p.parse_args(); c=load(a.config); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    h=c['chamber']; end=h['z_start']+h['length']
    z=np.linspace(h['z_start']-.06,end+c['downstream']['length'],501)
    b=pack_bz(z,c['coils'])
    thin=sum(loop_bz(z,k['radius'],k['z_center'],k['turns']*k['current']) for k in c['coils'])
    np.savetxt(out/'reference_axis.csv',np.column_stack([z,b,thin]),delimiter=',',
               header='z_m,Bz_finite_pack_T,Bz_thin_loop_T',comments='')
    mid=(h['z_start']+end)/2
    stats={'model':'free-space Biot-Savart; uniform finite winding packs; not a FEM solution',
           'center_Bz_T':float(pack_bz(mid,c['coils'])),
           'exit_Bz_T':float(pack_bz(end,c['coils'])),
           'current_scale_for_14mT_at_center':float(.014/pack_bz(mid,c['coils'])) if abs(pack_bz(mid,c['coils']))>1e-30 else None}
    (out/'reference_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.plot(z*1000,b*1000,label='Finite winding packs')
    ax.plot(z*1000,thin*1000,'--',label='Thin-loop approximation')
    ax.axvspan(h['z_start']*1000,end*1000,color='#dbeafe',alpha=.7,label='Chamber')
    ax.set(xlabel='Axial z [mm]',ylabel='On-axis Bz [mT]',title='Independent free-space reference • configured coil currents')
    ax.legend(); ax.grid(alpha=.25); fig.savefig(out/'reference_axis.png',dpi=180); plt.close(fig)
    print(json.dumps(stats,indent=2))

if __name__=='__main__': main()
