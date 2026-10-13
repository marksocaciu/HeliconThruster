import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest
import numpy as np
from helicon.config import load, validate, rectangles
from helicon.mesh import generate, gmsh_module
from helicon.reference import MU0, loop_bz, pack_bz


class ModelTests(unittest.TestCase):
    def test_reference_helmholtz_center(self):
        R=.12; NI=100
        expected=(4/5)**1.5*MU0*NI/R
        actual=loop_bz(0,R,-R/2,NI)+loop_bz(0,R,R/2,NI)
        self.assertAlmostEqual(float(actual/expected),1,places=13)

    def test_finite_pack_limit_and_superposition(self):
        c=load(); coils=copy.deepcopy(c['coils']); z=np.linspace(-.12,0,17)
        for k in coils: k.update(radial_thickness=1e-7,axial_width=1e-7)
        exact=sum(loop_bz(z,k['radius'],k['z_center'],k['turns']*k['current']) for k in coils)
        np.testing.assert_allclose(pack_bz(z,coils),exact,rtol=1e-10)
        base=pack_bz(z,c['coils'])
        for k in c['coils']: k['current'] *= -2
        np.testing.assert_allclose(pack_bz(z,c['coils']),-2*base,rtol=1e-13)

    def test_invalid_geometry(self):
        for change in ['overlap','outside','nan','duplicate','negative']:
            c=load()
            if change=='overlap': c['coils'][1]['z_center']=c['coils'][0]['z_center']
            if change=='outside': c['domain']['radius']=.05
            if change=='nan': c['coils'][0]['current']=float('nan')
            if change=='duplicate': c['coils'][1]['name']=c['coils'][0]['name']
            if change=='negative': c['chamber']['radius']=-1
            with self.subTest(change=change),self.assertRaises(ValueError): validate(c)

    def test_mesh_partition_and_interfaces(self):
        c=load()
        with tempfile.TemporaryDirectory() as folder:
            report=generate(c,folder)
            for name,_,(_,_,w,h) in rectangles(c):
                self.assertAlmostEqual(report['areas_m2'][name],w*h,places=12)
            g=gmsh_module(); g.initialize()
            try:
                g.open(str(Path(folder)/'helicon.msh'))
                tags,coords,_=g.model.mesh.getNodes(); coords=np.asarray(coords).reshape(-1,3)
                self.assertEqual(len(np.unique(np.round(coords,12),axis=0)),len(tags))
                points={int(t):v for t,v in zip(tags,coords)}
                edges=Counter(); cells=0
                for _,surface in g.model.getEntities(2):
                    self.assertEqual(len(g.model.getPhysicalGroupsForEntity(2,surface)),1)
                    types,_,connections=g.model.mesh.getElements(2,surface)
                    for typ,connection in zip(types,connections):
                        self.assertEqual(typ,2)
                        for tri in np.asarray(connection).reshape(-1,3):
                            cells += 1
                            a,b,d=[points[int(t)][:2] for t in tri]
                            self.assertGreater(abs(np.linalg.det(np.stack([b-a,d-a]))),1e-16)
                            for u,v in zip(tri,np.roll(tri,-1)):
                                edges[tuple(sorted((int(u),int(v))))] += 1
                self.assertEqual(cells,report['triangles'])
                self.assertTrue(all(n in (1,2) for n in edges.values()))
                tagged=set()
                for physical in (11,12):
                    for curve in g.model.getEntitiesForPhysicalGroup(1,physical):
                        _,_,connections=g.model.mesh.getElements(1,int(curve))
                        for connection in connections:
                            for pair in np.asarray(connection).reshape(-1,2):
                                tagged.add(tuple(sorted(map(int,pair))))
                self.assertEqual(tagged,{e for e,n in edges.items() if n==1})
                # One connected triangle graph: an interface crack produces extra components.
                adjacency={int(t):set() for t in tags}
                for a,b in edges: adjacency[a].add(b); adjacency[b].add(a)
                seen=set(); todo=[int(tags[0])]
                while todo:
                    t=todo.pop()
                    if t not in seen: seen.add(t); todo.extend(adjacency[t]-seen)
                self.assertEqual(len(seen),len(tags))
            finally: g.finalize()

if __name__=='__main__': unittest.main()
