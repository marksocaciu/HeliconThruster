"""Conforming OpenCASCADE geometry with persistent physical-region IDs."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from .config import load, rectangles


def gmsh_module():
    try:
        return importlib.import_module('gmsh')
    except ModuleNotFoundError:
        # Homebrew ships the Python API next to libgmsh, outside site-packages.
        for p in ['/opt/homebrew/opt/gmsh/lib', '/usr/local/opt/gmsh/lib']:
            if (Path(p) / 'gmsh.py').exists():
                sys.path.insert(0, p)
                return importlib.import_module('gmsh')
        raise SystemExit('Install gmsh: python -m pip install gmsh')


def generate(c, out, scale=1.0, gui=False):
    if not 0 < scale < float('inf'):
        raise ValueError('mesh scale must be finite and positive')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g = gmsh_module()
    g.initialize()
    try:
        g.model.add('helicon_rz')
        occ = g.model.occ
        d = c['domain']
        outer = occ.addRectangle(0, d['z_min'], 0, d['radius'], d['z_max']-d['z_min'])
        regions = rectangles(c)
        parts = [occ.addRectangle(r,z,0,w,h) for _,_,(r,z,w,h) in regions]
        _, mapping = occ.fragment([(2,outer)], [(2,t) for t in parts])
        occ.synchronize()
        assigned = set()
        groups = {}
        sizes = ['chamber','wall','downstream'] + ['coil']*len(c['coils'])
        fields = []
        for (name, tag, box), entities, size in zip(regions, mapping[1:], sizes):
            ids = [t for dim,t in entities if dim == 2]
            if not ids or assigned.intersection(ids):
                raise RuntimeError('Missing or multiply assigned material region')
            assigned.update(ids)
            g.model.addPhysicalGroup(2, ids, tag, name)
            groups[name] = {'dim':2, 'tag':tag, 'entities':ids}
            r,z,w,h = box
            f = g.model.mesh.field.add('Box')
            for key,value in {'VIn':c['mesh'][size]*scale, 'VOut':c['mesh']['far']*scale,
                              'XMin':r, 'XMax':r+w, 'YMin':z, 'YMax':z+h,
                              'ZMin':-1, 'ZMax':1, 'Thickness':c['mesh']['far']*scale}.items():
                g.model.mesh.field.setNumber(f,key,value)
            fields.append(f)
        air = sorted({t for _,t in g.model.getEntities(2)} - assigned)
        g.model.addPhysicalGroup(2, air, 4, 'SURROUNDING_AIR')
        groups['SURROUNDING_AIR'] = {'dim':2,'tag':4,'entities':air}
        boundary = g.model.getBoundary(g.model.getEntities(2), combined=True, oriented=False)
        axis, far = [], []
        for dim,t in boundary:
            xmin,_,_,xmax,_,_ = g.model.getBoundingBox(dim,t)
            (axis if abs(xmin)<1e-6 and abs(xmax)<1e-6 else far).append(t)
        for name,tag,ids in [('AXIS',11,axis),('FAR_BOUNDARY',12,far)]:
            if not ids:
                raise RuntimeError(f'Missing {name}')
            g.model.addPhysicalGroup(1,ids,tag,name)
            groups[name] = {'dim':1,'tag':tag,'entities':ids}
        f = g.model.mesh.field.add('Min')
        g.model.mesh.field.setNumbers(f, 'FieldsList', fields)
        g.model.mesh.field.setAsBackgroundMesh(f)
        for key,value in {'Mesh.MeshSizeFromPoints':0, 'Mesh.MeshSizeFromCurvature':0,
                          'Mesh.MeshSizeExtendFromBoundary':0, 'Mesh.Algorithm':6,
                          'Mesh.MshFileVersion':4.1, 'Mesh.ElementOrder':1}.items():
            g.option.setNumber(key,value)
        g.model.mesh.generate(2)
        g.write(str(out/'helicon.msh'))
        g.write(str(out/'helicon.brep'))
        # Mesh-backed GUI entry preserves tags; BREP alone has no physical groups.
        (out/'helicon.geo').write_text('Merge "helicon.msh";\n')
        areas = {name:sum(occ.getMass(2,t) for t in item['entities'])
                 for name,item in groups.items() if item['dim']==2}
        expected = d['radius']*(d['z_max']-d['z_min'])
        if abs(sum(areas.values())-expected) > 1e-10:
            raise RuntimeError('Subdomain areas do not cover the outer domain')
        element_ids = g.model.mesh.getElements(2)[1]
        report = {'coordinate_mapping':'Gmsh x = r, Gmsh y = axial z, Gmsh z = 0; SI metres',
                  'physical_groups':groups,'areas_m2':areas,
                  'nodes':len(g.model.mesh.getNodes()[0]),
                  'triangles':sum(len(ids) for ids in element_ids), 'mesh_scale':scale}
        (out/'mesh_report.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'geometry.json').write_text(json.dumps(c,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ['nodes','triangles','areas_m2']},indent=2))
        if gui:
            g.fltk.run()
        return report
    finally:
        g.finalize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--out', default='build')
    p.add_argument('--scale',type=float,default=1)
    p.add_argument('--gui',action='store_true')
    a = p.parse_args()
    generate(load(a.config),a.out,a.scale,a.gui)

if __name__ == '__main__':
    main()
