import unittest
from types import SimpleNamespace
from helicon.rf.config import load, validate
from helicon.rf.mesh import transition_end, quality_report

class MeshGrading(unittest.TestCase):
    def test_legacy_transition_is_unchanged(self):
        self.assertEqual(transition_end({'bulk_m':.07},.00125,.0015,.0045),.0045)

    def test_growth_bound_and_existing_long_transition(self):
        size={'bulk_m':.07,'max_size_gradient':.8}
        end=transition_end(size,.00125,.0015,.0045)
        self.assertAlmostEqual((.07-.00125)/(end-.0015),.8)
        self.assertEqual(transition_end(size,.00125,.0015,.2),.2)

    def test_config_rejects_invalid_gradient(self):
        c=load();c['mesh']['max_size_gradient']=.8;validate(c)
        for val in (0,-1,1.1,float('nan'),True):
            c['mesh']['max_size_gradient']=val
            with self.assertRaises(ValueError):validate(c)

    def test_quality_reports_bad_fraction_and_rejects_inversion(self):
        mesh=SimpleNamespace(getElements=lambda dim,entity:([4],[[1,2,3]],[]),
                             getElementQualities=lambda tags,metric:[.05,.5,.9])
        g=SimpleNamespace(model=SimpleNamespace(mesh=mesh))
        report=quality_report(g,{'AIR':{1}})['AIR']
        self.assertAlmostEqual(report['fraction_below_0p1'],1/3)
        self.assertEqual(report['minSICN_median'],.5)
        mesh.getElementQualities=lambda tags,metric:[-.1,.5,.9]
        with self.assertRaisesRegex(RuntimeError,'Invalid tetrahedra'):quality_report(g,{'AIR':{1}})
