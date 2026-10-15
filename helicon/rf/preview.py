"""Draw the meshed PEC antenna and feed sheet, with a chamber outline."""
import argparse
from pathlib import Path
import numpy as np
from ..mesh import gmsh_module
from ..records import read_json,digest
from .mesh import FACE


def render(build):
    build=Path(build); report=read_json(build/'mesh_report.json');c=read_json(build/'rf.json')
    if digest(build/'rf.msh')!=report['mesh_sha256']: raise ValueError('Changed RF mesh')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    g=gmsh_module();g.initialize()
    try:
        g.open(str(build/'rf.msh'))
        ids,coords,_=g.model.mesh.getNodes();xyz=np.asarray(coords).reshape(-1,3)
        lookup={int(k):v for k,v in zip(ids,xyz)}
        fig=plt.figure(figsize=(8,7));ax=fig.add_subplot(projection='3d')
        for tag,color in [(FACE['PEC_ANTENNA'],'#b87333'),(FACE['FEED_SHEET'],'#238fcd')]:
            triangles=[]
            for surface in g.model.getEntitiesForPhysicalGroup(2,tag):
                types,_,connections=g.model.mesh.getElements(2,int(surface))
                for typ,conn in zip(types,connections):
                    if typ!=2: raise ValueError('Preview expects linear triangle faces')
                    triangles.extend([[lookup[int(i)] for i in tri] for tri in np.asarray(conn).reshape(-1,3)])
            ax.add_collection3d(Poly3DCollection(triangles,facecolors=color,edgecolors='#382f27',linewidths=.08,alpha=.95))
        h=c['chamber'];theta=np.linspace(0,2*np.pi,120);r=h['radius_m']+h['wall_thickness_m']
        for z in (h['z_start_m'],h['z_start_m']+h['length_m']): ax.plot(r*np.cos(theta),r*np.sin(theta),theta*0+z,color='gray',alpha=.6)
        ax.set(xlim=(-.05,.05),ylim=(-.05,.05),zlim=(h['z_start_m']-.01,h['z_start_m']+h['length_m']+.01),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
        ax.set_box_aspect((1,1,1.4));ax.view_init(elev=20,azim=-35)
        ax.set_title('Meshed ring-and-rail antenna\nCopper: PEC boundary · Blue: ideal feed sheet')
        fig.tight_layout();fig.savefig(build/'antenna_mesh.png',dpi=160);plt.close(fig)
    finally:g.finalize()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',required=True);a=p.parse_args();render(a.build)
if __name__=='__main__': main()
