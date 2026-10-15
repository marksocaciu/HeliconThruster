"""3D ring-and-rail antenna, dielectric tube, vacuum box and Cartesian PML."""
import argparse
import math
from pathlib import Path
from ..mesh import gmsh_module
from ..records import write_json, digest
from .config import load, validate
CELL={'PLASMA':1,'WALL':2,'AIR':4,'FEED_GAP':6,'PML':7}
FACE={'PEC_ANTENNA':11,'OUTER':12,'FEED_SHEET':13,'PML_INTERFACE':14}


def generate(c,out,scale=1.):
    validate(c)
    if not math.isfinite(scale) or scale<=0: raise ValueError('Mesh scale must be positive')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    write_json(out/'rf.json',c)
    try: g=gmsh_module()
    except SystemExit as exc: raise RuntimeError(str(exc)) from exc
    g.initialize()
    try:
        o=g.model.occ; g.model.add('rf_3d')
        h,a,d=c['chamber'],c['antenna'],c['domain']
        z0=h['z_start_m']; length=h['length_m']; ar=a['inner_radius_m']; thick=a['radial_thickness_m']
        def shell(radius,t,z,l,angle=2*math.pi,rotation=0):
            outer=o.addCylinder(0,0,z,0,0,l,radius+t,angle=angle)
            inner=o.addCylinder(0,0,z,0,0,l,radius,angle=angle)
            result,_=o.cut([(3,outer)],[(3,inner)])
            if rotation: o.rotate(result,0,0,0,0,0,1,rotation)
            return result
        width=a['ring_width_m']; theta=a['leg_angle_rad']; gap=a['gap_m']; center=z0+length/2
        metal=shell(ar,thick,z0,width)+shell(ar,thick,z0+length-width,width)
        metal+=shell(ar,thick,z0,length,theta,-theta/2)
        metal+=shell(ar,thick,z0,length,theta,math.pi-theta/2)
        metal,_=o.fuse(metal[:1],metal[1:])
        cutter=shell(ar,thick,center-gap/2,gap,theta,-theta/2)
        metal,_=o.cut(metal,cutter)
        port=shell(ar,thick,center-gap/2,gap,theta,-theta/2)
        plasma=[(3,o.addCylinder(0,0,z0,0,0,length,h['radius_m']))]
        wall=shell(h['radius_m'],h['wall_thickness_m'],z0,length)
        b=d['half_width_m']; t=d['pml_thickness_m']; low=d['z_min_m']; high=d['z_max_m']
        outer=[(3,o.addBox(-b-t,-b-t,low-t,2*(b+t),2*(b+t),high-low+2*t))]
        core=[(3,o.addBox(-b,-b,low,2*b,2*b,high-low))]
        _,maps=o.fragment(outer,core+plasma+wall+metal+port); o.synchronize()
        def vols(items): return {tag for dim,tag in items if dim==3}
        allvol=vols(maps[0]); corevol=vols(maps[1]); pv=vols(maps[2]); wv=vols(maps[3])
        mv=set().union(*(vols(x) for x in maps[4:4+len(metal)]))
        gv=set().union(*(vols(x) for x in maps[4+len(metal):]))
        def boundary(volumes):
            return {tag for dim,tag in g.model.getBoundary([(3,v) for v in sorted(volumes)],combined=True,oriented=False) if dim==2}
        pec=boundary(mv); feed=boundary(gv)-pec; exterior=boundary(allvol); interface=boundary(corevol)
        o.remove([(3,v) for v in mv],recursive=False); o.synchronize()
        regions={'PLASMA':pv,'WALL':wv,'AIR':corevol-pv-wv-mv-gv,'FEED_GAP':gv,'PML':allvol-corevol}
        surfaces={'PEC_ANTENNA':pec,'OUTER':exterior,'FEED_SHEET':feed,'PML_INTERFACE':interface}; groups={}
        for dim,collection,ids in ((3,regions,CELL),(2,surfaces,FACE)):
            for name,entities in collection.items():
                if not entities: raise RuntimeError(f'Missing RF group {name}')
                g.model.addPhysicalGroup(dim,sorted(entities),ids[name],name)
                groups[name]={'dim':dim,'tag':ids[name],'entities':sorted(entities)}
        volumes={name:sum(o.getMass(3,v) for v in ids) for name,ids in regions.items()}
        annulus=math.pi*((ar+thick)**2-ar**2)
        expected_metal=annulus*(2*width+(theta/math.pi)*(length-2*width)-theta/(2*math.pi)*gap)
        expectedvol=4*(b+t)**2*(high-low+2*t)
        if abs(sum(volumes.values())+expected_metal-expectedvol)>1e-9*expectedvol:
            raise RuntimeError('RF volume partition check failed')
        if abs(volumes['PLASMA']-math.pi*h['radius_m']**2*length)>1e-10: raise RuntimeError('Plasma volume mismatch')
        size=c['mesh']; fields=[]
        for name,key in [('PLASMA','plasma_m'),('WALL','wall_m'),('FEED_GAP','gap_m')]:
            f=g.model.mesh.field.add('MathEval'); g.model.mesh.field.setString(f,'F',str(size[key]*scale))
            restrict=g.model.mesh.field.add('Restrict'); g.model.mesh.field.setNumber(restrict,'InField',f)
            g.model.mesh.field.setNumbers(restrict,'VolumesList',sorted(regions[name])); fields.append(restrict)
        distance=g.model.mesh.field.add('Distance'); g.model.mesh.field.setNumbers(distance,'SurfacesList',sorted(pec|feed))
        g.model.mesh.field.setNumber(distance,'Sampling',80)
        near=g.model.mesh.field.add('Threshold'); g.model.mesh.field.setNumber(near,'InField',distance)
        for key,val in {'SizeMin':min(size['antenna_m'],size['gap_m'])*scale,'SizeMax':size['bulk_m']*scale,'DistMin':.003,'DistMax':.014}.items():
            g.model.mesh.field.setNumber(near,key,val)
        fields.append(near)
        f=g.model.mesh.field.add('Min'); g.model.mesh.field.setNumbers(f,'FieldsList',fields); g.model.mesh.field.setAsBackgroundMesh(f)
        for key,val in {'Mesh.MeshSizeMax':size['bulk_m']*scale,'Mesh.MeshSizeFromPoints':0,'Mesh.MeshSizeFromCurvature':0,
                        'Mesh.MeshSizeExtendFromBoundary':0,'Mesh.MshFileVersion':4.1,'Mesh.ElementOrder':1,'Mesh.Algorithm3D':1}.items():
            g.option.setNumber(key,val)
        g.write(str(out/'rf.brep')); g.model.mesh.generate(3); g.write(str(out/'rf.msh'))
        (out/'rf.geo').write_text('Merge "rf.msh";\n')
        report={'groups':groups,'volumes_m3':volumes,'metal_volume_m3':expected_metal,'mesh_scale':scale,
                'nodes':len(g.model.mesh.getNodes()[0]),'tetrahedra':sum(len(x) for x in g.model.mesh.getElements(3)[1]),
                'gmsh_version':g.__version__,'mesh_sha256':digest(out/'rf.msh'),'config_sha256':digest(out/'rf.json'),
                'note':'Geometry smoke mesh only; electromagnetic and PML convergence must be checked.'}
        write_json(out/'mesh_report.json',report); return report
    finally: g.finalize()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config'); p.add_argument('--out',required=True); p.add_argument('--scale',type=float,default=1.)
    a=p.parse_args(); print(generate(load(a.config),a.out,a.scale))

if __name__=='__main__': main()
