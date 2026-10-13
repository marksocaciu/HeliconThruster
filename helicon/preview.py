"""Render physical regions and the actual generated triangular mesh."""
import argparse
from pathlib import Path
import numpy as np
from .mesh import gmsh_module


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--build',default='build'); a=p.parse_args()
    out=Path(a.build)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    g=gmsh_module(); g.initialize()
    try:
        g.open(str(out/'helicon.msh'))
        tags,xyz,_=g.model.mesh.getNodes(); xyz=np.asarray(xyz).reshape(-1,3)[:,:2]*1000
        lookup={int(t):i for i,t in enumerate(tags)}
        fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained')
        colors={1:'#79bcea',2:'#cab3e6',3:'#a2d7ce',4:'#edf1f5'}
        legend=[]
        for dim,tag in g.model.getPhysicalGroups(2):
            triangles=[]
            for surface in g.model.getEntitiesForPhysicalGroup(dim,tag):
                types,_,connect=g.model.mesh.getElements(2,int(surface))
                for typ,conn in zip(types,connect):
                    if typ != 2: raise ValueError('Preview expects first-order triangles')
                    triangles.extend([[lookup[int(n)] for n in tri] for tri in np.asarray(conn).reshape(-1,3)])
            color=colors.get(tag,'#e69a50')
            name=g.model.getPhysicalName(dim,tag)
            legend.append(Patch(facecolor=color,label=name))
            for ax in axes:
                ax.add_collection(PolyCollection(xyz[np.asarray(triangles)],facecolors=color,
                                                 edgecolors='#64748b',linewidths=.17))
                ax.autoscale_view(); ax.set_aspect('equal'); ax.set(xlabel='r [mm]',ylabel='z [mm]')
        axes[0].set_title('Full magnetic domain')
        from .config import load
        c=load(out/'geometry.json'); h=c['chamber']
        xmax=max(k['radius']+k['radial_thickness']/2 for k in c['coils'])
        axes[1].set(xlim=(-3,xmax*1000+15),ylim=(h['z_start']*1000-25,(h['z_start']+h['length'])*1000+60),
                    title='Chamber and coil detail')
        fig.legend(handles=legend,loc='outside lower center',ncol=3)
        fig.suptitle('Helicon thruster • axisymmetric Gmsh mesh',fontsize=16)
        fig.savefig(out/'mesh_preview.png',dpi=180); plt.close(fig)
    finally: g.finalize()

if __name__=='__main__': main()
