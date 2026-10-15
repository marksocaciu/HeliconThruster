"""Optional compiler-only checks; they do not require PETSc or a FEM solve."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('ffcx') is not None,'optional FFCx compiler is not installed')
class ComplexForms(unittest.TestCase):
    def test_actual_complex_weak_form_and_power_diagnostics_compile(self):
        import ufl
        import basix.ufl
        import ffcx.compiler
        import ffcx.options
        import numpy as np
        from helicon.rf.config import load
        from helicon.rf.forms import bilinear,interface_flux
        from helicon.rf.materials import EPS_0
        c=load()
        domain=ufl.Mesh(basix.ufl.element('Lagrange','tetrahedron',1,shape=(3,)))
        for degree in (1,2):
            V=ufl.FunctionSpace(domain,basix.ufl.element('N1curl','tetrahedron',degree))
            T=ufl.FunctionSpace(domain,basix.ufl.element('DG','tetrahedron',0,shape=(3,3)))
            u,v=ufl.TrialFunction(V),ufl.TestFunction(V);eps=ufl.Coefficient(T);E=ufl.Coefficient(V)
            dx=ufl.Measure('dx',domain=domain,metadata={'quadrature_degree':6});ds=ufl.Measure('dS',domain=domain)
            a=bilinear(domain,u,v,eps,dx,c)
            flux=interface_flux(domain,E,c['frequency_Hz'],(0,0,-.06))*ds(14)
            heat=.5*2*np.pi*c['frequency_Hz']*EPS_0*ufl.imag(ufl.inner(eps*E,E))*dx(1)
            header,source=ffcx.compiler.compile_ufl_objects([a,flux,heat],options=ffcx.options.get_options({'scalar_type':'complex128'}))
            self.assertTrue(header);self.assertTrue(source)

if __name__=='__main__':unittest.main()
