"""Steady axisymmetric quasineutral argon drift-diffusion benchmark.

Four independent cell fields: ion/electron density, neutral density, electron
temperature, and electrostatic potential. See docs/spatial-plasma.md.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .argon import (E_CHARGE as E, K_B, M_E, M_AR, IONIZATION_EV,
                    EXCITATION_EV, TE_MIN_EV, TE_MAX_EV, rates)
from .config import ROOT, load as load_geometry
from .records import write_json

MODEL = 'argon_axisymmetric_dd_v1'


def load(path=None):
    c = json.loads(Path(path or ROOT/'config/plasma_spatial_argon.json').read_text())
    validate(c)
    return c


def validate(c):
    expected = {'model', 'nr', 'nz', 'absorbed_power_W', 'mass_flow_kg_s',
                'neutral_temperature_K', 'ion_temperature_eV', 'ion_mobility_m2_V_s',
                'neutral_diffusivity_m2_s', 'uniform_B_T', 'collision_floor_s-1',
                'wall_recycling_fraction', 'max_nfev', 'residual_tolerance'}
    if set(c) != expected or c['model'] != MODEL:
        raise ValueError('Unexpected spatial argon configuration keys/model')
    for key in expected - {'model', 'uniform_B_T'}:
        x = c[key]
        if isinstance(x, bool) or not isinstance(x, (float, int)) or not np.isfinite(x) or x <= 0:
            if key != 'wall_recycling_fraction' or x != 0 or isinstance(x, bool):
                raise ValueError(f'{key} must be finite and positive')
    for key in ('nr', 'nz', 'max_nfev'):
        if not isinstance(c[key], int) or c[key] < (2 if key != 'max_nfev' else 1):
            raise ValueError(f'{key} must be an integer (grid dimensions >= 2)')
    if not 0 <= c['wall_recycling_fraction'] <= 1:
        raise ValueError('wall_recycling_fraction must be in [0, 1]')
    if not c['residual_tolerance'] < .01:
        raise ValueError('residual_tolerance must be below .01')
    if K_B*c['neutral_temperature_K']/E >= TE_MIN_EV:
        raise ValueError('Neutral temperature exceeds rate interval')
    b = np.asarray(c['uniform_B_T'], dtype=float)
    if b.shape != (3,) or not np.isfinite(b).all() or np.any(b[:2] != 0):
        raise ValueError('Benchmark transport requires a uniform axial B: [0, 0, Bz]')


class Grid:
    def __init__(self, geometry, nr, nz):
        h = geometry['chamber']
        self.rf = np.linspace(0, h['radius'], nr+1)
        self.zf = np.linspace(h['z_start'], h['z_start']+h['length'], nz+1)
        if h['radius'] <= 0 or h['length'] <= 0:
            raise ValueError('Chamber dimensions must be positive')
        self.r = .5*(self.rf[:-1]+self.rf[1:])
        self.z = .5*(self.zf[:-1]+self.zf[1:])
        self.shape = (nz, nr)
        self.dr, self.dz = self.rf[1], self.zf[1]-self.zf[0]
        annulus = np.pi*np.diff(self.rf**2)
        self.volumes = np.broadcast_to(annulus*self.dz, self.shape).copy()
        ids = np.arange(nr*nz).reshape(self.shape)
        # Face orientation is L -> R, axial first then radial.
        self.left = np.r_[ids[:-1].ravel(), ids[:, :-1].ravel()]
        self.right = np.r_[ids[1:].ravel(), ids[:, 1:].ravel()]
        self.axis = np.r_[np.ones((nz-1)*nr, dtype=int), np.zeros(nz*(nr-1), dtype=int)]
        self.area = np.r_[np.tile(annulus, nz-1),
                          np.tile(2*np.pi*self.rf[1:-1]*self.dz, nz)]
        self.distance = np.where(self.axis == 1, self.dz, self.dr)
        self.boundary = np.r_[ids[:, -1], ids[0], ids[-1]]
        self.boundary_kind = np.r_[np.zeros(nz, int), np.ones(nr, int), np.full(nr, 2, int)]
        self.boundary_area = np.r_[np.full(nz, 2*np.pi*self.rf[-1]*self.dz), annulus, annulus]
        self.size = nr*nz

    def scatter(self, flux):
        out = np.zeros(self.size)
        np.add.at(out, self.left, flux)
        np.add.at(out, self.right, -flux)
        return out

    def boundary_scatter(self, flux):
        out = np.zeros(self.size)
        np.add.at(out, self.boundary, flux)
        return out


def bernoulli(x):
    """Stable Scharfetter-Gummel Bernoulli function, including large drift."""
    x = np.asarray(x)
    small = np.abs(x) < 1e-4
    y = np.empty_like(x, dtype=float)
    y[small] = 1-x[small]/2+x[small]**2/12-x[small]**4/720
    mid = ~small & (np.abs(x) < 50)
    y[mid] = x[mid]/np.expm1(x[mid])
    pos = x >= 50
    y[pos] = x[pos]*np.exp(-x[pos])
    neg = x <= -50
    y[neg] = -x[neg]
    return y


def coefficients(te, nn, c):
    k = {key: np.array([rates(float(t))[key] for t in te]) for key in
         ('ionization_m3_s', 'excitation_m3_s', 'elastic_m3_s')}
    nu = nn*k['elastic_m3_s']+c['collision_floor_s-1']
    mu = E/(M_E*nu)
    beta = E*abs(c['uniform_B_T'][2])/(M_E*nu)
    return k, nu, np.column_stack((mu/(1+beta**2), mu))


class Equations:
    def __init__(self, grid, c, heating):
        self.g, self.c = grid, c
        q = np.asarray(heating, float)
        if q.shape != grid.shape or not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError('Heating must be a finite nonnegative (nz, nr) W/m3 array')
        self.heating = q.ravel()
        self.power = float(np.dot(self.heating, grid.volumes.ravel()))
        if self.power <= 0:
            raise ValueError('A sustained benchmark requires positive absorbed power')
        self.feed = c['mass_flow_kg_s']/M_AR
        # Fixed per-cell scales: small cells cannot hide large local residuals.
        self.particle_scale = self.feed*grid.volumes.ravel()/grid.volumes.sum()
        self.energy_scale = self.power*grid.volumes.ravel()/grid.volumes.sum()

    def evaluate(self, x):
        g, c = self.g, self.c
        n, nn = np.exp(x[0])*1e18, np.exp(x[1])*1e19
        te, phi = x[2], x[3]
        k, nu, mu = coefficients(te, nn, c)
        l, r = g.left, g.right
        tf = .5*(te[l]+te[r])
        muf = 2*mu[l, g.axis]*mu[r, g.axis]/(mu[l, g.axis]+mu[r, g.axis])
        dphi, dt = phi[r]-phi[l], te[r]-te[l]
        di = c['ion_mobility_m2_V_s']*c['ion_temperature_eV']
        pi = dphi/c['ion_temperature_eV']
        pe = (-dphi+dt)/tf
        ion = di/g.distance*(n[l]*bernoulli(pi)-n[r]*bernoulli(-pi))*g.area
        electron = muf*tf/g.distance*(n[l]*bernoulli(pe)-n[r]*bernoulli(-pe))*g.area
        neutral = c['neutral_diffusivity_m2_s']/g.distance*(nn[l]-nn[r])*g.area
        # Upwind electron enthalpy plus anisotropic Fourier heat flux.
        nf = 2*n[l]*n[r]/(n[l]+n[r])
        conduction = -2.5*E*nf*muf*tf*dt/g.distance*g.area
        energy = 2.5*E*np.where(electron >= 0, te[l], te[r])*electron+conduction
        # Electron electrostatic work: Gamma_e . grad(phi), split over adjacent cells.
        work = np.zeros(g.size)
        np.add.at(work, l, .5*E*electron*dphi)
        np.add.at(work, r, .5*E*electron*dphi)
        b = g.boundary
        bohm = n[b]*np.sqrt(E*te[b]/M_AR)*g.boundary_area
        wall = g.boundary_kind != 2
        thermal = math.sqrt(K_B*c['neutral_temperature_K']/(2*np.pi*M_AR))
        neutral_b = np.where(wall, -c['wall_recycling_fraction']*bohm,
                             nn[b]*thermal*g.boundary_area)
        back = g.boundary_kind == 1
        neutral_b[back] -= self.feed*g.boundary_area[back]/sum(g.boundary_area[back])
        sheath = .5*math.log(M_AR/(2*np.pi*M_E))
        # Electron loss at a floating sheath edge includes the potential barrier.
        energy_b = E*te[b]*bohm*np.where(wall, 2+sheath, 2.5)
        volume = g.volumes.ravel()
        source = n*nn*k['ionization_m3_s']*volume
        inelastic = E*n*nn*volume*(k['ionization_m3_s']*IONIZATION_EV
                    +k['excitation_m3_s']*EXCITATION_EV
                    +k['elastic_m3_s']*3*M_E/M_AR*(te-K_B*c['neutral_temperature_K']/E))
        ion_res = g.scatter(ion)+g.boundary_scatter(bohm)-source
        neutral_res = g.scatter(neutral)+g.boundary_scatter(neutral_b)+source
        current_res = g.scatter(ion-electron)
        energy_res = g.scatter(energy)+g.boundary_scatter(energy_b)+inelastic-self.heating*volume-work
        return dict(n=n, nn=nn, te=te, phi=phi, nu=nu, ion_res=ion_res,
                    neutral_res=neutral_res, current_res=current_res, energy_res=energy_res,
                    boundary_ions=bohm, boundary_neutrals=neutral_b, boundary_energy=energy_b,
                    source=source, inelastic=inelastic, work=work)

    def residual(self, flat):
        x = flat.reshape(4, self.g.size)
        a = self.evaluate(x)
        res = np.array([a['ion_res']/self.particle_scale,
                        a['neutral_res']/self.particle_scale,
                        a['energy_res']/self.energy_scale,
                        a['current_res']/self.particle_scale])
        # Pure-Neumann current equation: one redundant row becomes the gauge.
        res[3, 0] = x[3, 0]/5
        return res.ravel()

    def jacobian(self):
        """Color local stencils to assemble a dense small-benchmark Jacobian.

        Sparse iterative least-squares was ill-conditioned for electron/ion
        mobility ratios. Coloring keeps the robust dense trust-region solve
        without perturbing every cell independently.
        """
        structure = self.sparsity().tocsc()
        rows = [structure.indices[structure.indptr[j]:structure.indptr[j+1]]
                for j in range(structure.shape[1])]
        groups, occupied = [], []
        for j, support in enumerate(rows):
            support_set = set(support)
            for group, used in zip(groups, occupied):
                if not support_set & used:
                    group.append(j); used.update(support_set)
                    break
            else:
                groups.append([j]); occupied.append(support_set)

        def jac(x):
            base = self.residual(x)
            matrix = np.zeros(structure.shape)
            steps = np.sqrt(np.finfo(float).eps)*np.maximum(np.abs(x), 1.)
            tslice = slice(2*self.g.size, 3*self.g.size)
            steps[tslice] *= np.where(x[tslice]+steps[tslice] > TE_MAX_EV, -1., 1.)
            for group in groups:
                point = x.copy(); point[group] += steps[group]
                delta = self.residual(point)-base
                for j in group:
                    matrix[rows[j], j] = delta[rows[j]]/steps[j]
            return matrix
        return jac

    def sparsity(self):
        n = self.g.size
        s = lil_matrix((4*n, 4*n), dtype=int)
        neighbors = [{i} for i in range(n)]
        for l, r in zip(self.g.left, self.g.right):
            neighbors[l].add(r)
            neighbors[r].add(l)
        for i, adjacent in enumerate(neighbors):
            cols = [j+k*n for j in adjacent for k in range(4)]
            for k in range(4):
                s[i+k*n, cols] = 1
        return s.tocsr()


def solve(geometry, c, heating=None, initial=None):
    validate(c)
    g = Grid(geometry, c['nr'], c['nz'])
    if heating is None:
        heating = np.full(g.shape, c['absorbed_power_W']/g.volumes.sum())
    if g.size > 512:
        raise ValueError('Dense benchmark solver is limited to 512 cells; use a scalable Newton solver for larger studies')
    eq = Equations(g, c, heating)
    if initial is None:
        x = np.zeros((4, g.size))
        x[0] = math.log(2.)
        x[1] = math.log(3.)
        x[2] = 4.5
    else:
        x = np.array([np.log(np.asarray(initial['electron_density_m3']).ravel()/1e18),
                      np.log(np.asarray(initial['neutral_density_m3']).ravel()/1e19),
                      np.asarray(initial['electron_temperature_eV']).ravel(),
                      np.asarray(initial['potential_V']).ravel()])
        if x.shape != (4, g.size) or not np.isfinite(x).all():
            raise ValueError('Initial state must match the spatial grid')
        x[3] -= x[3, 0]
    lower = np.r_[np.full(2*g.size, -20.), np.full(g.size, TE_MIN_EV), np.full(g.size, -500.)]
    upper = np.r_[np.full(2*g.size, 20.), np.full(g.size, TE_MAX_EV), np.full(g.size, 500.)]
    result = least_squares(eq.residual, x.ravel(), bounds=(lower, upper),
                           jac=eq.jacobian(), x_scale='jac',
                           ftol=1e-11, xtol=1e-11, gtol=1e-11,
                           max_nfev=c['max_nfev'])
    a = eq.evaluate(result.x.reshape(4, g.size))
    residuals = {key: float(np.max(np.abs(a[key]/scale))) for key, scale in (
        ('ion_res', eq.particle_scale), ('neutral_res', eq.particle_scale),
        ('current_res', eq.particle_scale), ('energy_res', eq.energy_scale))}
    boundary = a['boundary_ions']
    wall = g.boundary_kind != 2
    exit_ions = float(boundary[~wall].sum())
    wall_ions = float(boundary[wall].sum())
    neutral_exit = float((a['boundary_neutrals'][~wall]).sum())
    power_out = float(a['inelastic'].sum()+a['boundary_energy'].sum()-a['work'].sum())
    balances = {
        'ionization_events_s': float(a['source'].sum()),
        'wall_ions_s': wall_ions, 'exit_ions_s': exit_ions,
        'neutral_exit_atoms_s': neutral_exit,
        'wall_retained_atoms_s': (1-c['wall_recycling_fraction'])*wall_ions,
        'absorbed_power_W': eq.power, 'inelastic_power_W': float(a['inelastic'].sum()),
        'electron_boundary_power_W': float(a['boundary_energy'].sum()),
        'electron_electrostatic_work_W': float(a['work'].sum()),
        'relative_mass_error': abs(eq.feed-neutral_exit-exit_ions-(1-c['wall_recycling_fraction'])*wall_ions)/eq.feed,
        'relative_power_error': abs(eq.power-power_out)/eq.power,
        'max_local_relative_residuals': residuals,
    }
    passed = max(*residuals.values(), balances['relative_mass_error'],
                 balances['relative_power_error']) < c['residual_tolerance']
    fields = dict(r_m=g.r, z_m=g.z, r_faces_m=g.rf, z_faces_m=g.zf,
                  cell_volume_m3=g.volumes, q_abs_W_m3=np.asarray(heating),
                  electron_density_m3=a['n'].reshape(g.shape),
                  neutral_density_m3=a['nn'].reshape(g.shape),
                  electron_temperature_eV=a['te'].reshape(g.shape),
                  potential_V=a['phi'].reshape(g.shape),
                  collision_frequency_s=a['nu'].reshape(g.shape))
    summary = {'model': MODEL, 'status': 'converged' if passed else 'not_converged',
               'balances': balances, 'solver': {'nfev': result.nfev, 'message': result.message},
               'input': c, 'limitations': [
                   'Quasineutral steady drift-diffusion benchmark; ion inertia and nozzle acceleration omitted.',
                   'Prescribed ion mobility, neutral diffusivity and gas temperature; validate transport against pressure and size.',
                   'Uniform axial magnetic field; diagonal parallel/perpendicular electron transport, no Hall/azimuthal dynamics.',
                   'Bohm losses at cell boundary, floating non-emitting dielectric walls; no resolved Debye sheath.',
                   'Maxwellian ground-state argon, existing rate fits restricted to 1–7 eV.',
                   'Potential gauge is phi[0,0]=0; the potential is not an absolute wall-referenced voltage.',
                   'Electron energy balance is reported; ion kinetic energy and total thrust efficiency are not modeled.',
                   'Numerical convergence is not physical or experimental validation.']}
    return summary, fields


def save(out, summary, fields):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'spatial_summary.json', summary)
    np.savez_compressed(out/'plasma_fields.npz', **fields)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config'); p.add_argument('--geometry'); p.add_argument('--out', required=True)
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--heating', help='NPZ containing q_abs_W_m3 and matching r/z face arrays')
    a = p.parse_args()
    c, geometry = load(a.config), load_geometry(a.geometry)
    q = None
    if a.heating:
        grid = Grid(geometry, c['nr'], c['nz'])
        with np.load(a.heating, allow_pickle=False) as data:
            for key, expected in [('r_faces_m', grid.rf), ('z_faces_m', grid.zf)]:
                if data[key].shape != expected.shape or not np.allclose(data[key], expected, rtol=0, atol=1e-12):
                    raise ValueError('Heating grid does not match plasma grid')
            q = data['q_abs_W_m3'].copy()
    summary, fields = solve(geometry, c, q)
    save(a.out, summary, fields)
    from .rf.run import snapshot
    from .records import digest, utc_now
    record = {'status': summary['status'], 'finished_utc': utc_now(),
              'source': snapshot(a.out)}
    write_json(Path(a.out)/'geometry.json', geometry)
    if not a.no_plots:
        from .spatial_report import render
        render(a.out, summary, fields)
    record['artifacts_sha256'] = {path.name: digest(path) for path in Path(a.out).iterdir() if path.is_file()}
    write_json(Path(a.out)/'run.json', record)
    print(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'converged' else 2


if __name__ == '__main__':
    raise SystemExit(main())
