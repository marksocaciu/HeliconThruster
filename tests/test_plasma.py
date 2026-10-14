"""Global-model conservation and manufactured-state tests; no MPI dependency."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from helicon.argon import E_CHARGE as E, K_B, M_AR, M_E, rates
from helicon.config import load as geometry
from helicon.plasma import load, solve, run_case, NoSteadyStateError, validate
from helicon.records import digest


class GlobalPlasmaTests(unittest.TestCase):
    def test_manufactured_states_recover_temperature_density_and_balances(self):
        # Construct input power/flow from a chosen state without calling trial_state,
        # evaluate or geometry_terms. This checks root elimination, units, recycling,
        # and distinct wall/open-exit energy terms against the unreduced balances.
        for t in (1.1, 3.0, 6.9):
            for recycling in (0., .4, 1.):
                with self.subTest(temperature=t, recycling=recycling):
                    c = load(); geo = geometry()
                    c['losses']['wall_recycling_fraction'] = recycling
                    r, length = .03, .12
                    area, volume = math.pi*r*r, math.pi*r*r*length
                    effective_wall = .1*2*math.pi*r*length + .5*area
                    effective_exit = .5*area
                    cs = math.sqrt(E*t/M_AR)
                    k = rates(t)
                    ne = 7.3e17
                    nn = cs*(effective_wall+effective_exit)/(volume*k['ionization_m3_s'])
                    wall, exit_ions = ne*cs*effective_wall, ne*cs*effective_exit
                    exit_neutrals = nn*area*math.sqrt(K_B*400/(2*math.pi*M_AR))
                    feed = exit_neutrals+exit_ions+(1-recycling)*wall
                    pair = ne*nn*volume
                    power = E*(pair*(k['ionization_m3_s']*15.76+k['excitation_m3_s']*12.14
                              +k['elastic_m3_s']*3*M_E/M_AR*(t-K_B*400/E))
                              +wall*t*(2.5+.5*math.log(M_AR/(2*math.pi*M_E)))+exit_ions*t*3)
                    c.update(mass_flow_kg_s=feed*M_AR, absorbed_power_W=power)
                    result = solve(geo, c)
                    actual = result['metrics']
                    self.assertAlmostEqual(actual['electron_temperature_eV'], t, places=8)
                    self.assertAlmostEqual(actual['electron_density_m3']/ne, 1., places=8)
                    self.assertAlmostEqual(actual['neutral_density_m3']/nn, 1., places=8)
                    self.assertLess(max(map(abs, result['balances']['relative_residuals'].values())), 1e-9)

    def test_example_global_atom_and_energy_conservation(self):
        c=load(); result=solve(geometry(), c); b=result['balances']; m=result['metrics']
        q=c['mass_flow_kg_s']/M_AR
        self.assertAlmostEqual((b['neutral_exit_atoms_s']+m['exit_ion_rate_s']+b['wall_retained_atoms_s'])/q, 1., places=10)
        self.assertAlmostEqual(sum(b['power_budget_W'].values())/c['absorbed_power_W'], 1., places=12)
        self.assertGreater(b['ionization_events_s'], q)  # Recycling can re-ionize an atom.
        self.assertGreater(m['propellant_utilization'], m['ionized_number_fraction'])
        self.assertTrue(0 < m['propellant_utilization'] < 1)

    def test_zero_power_is_neutral_gas_with_undefined_temperature(self):
        c=load(); c['absorbed_power_W']=0
        result=solve(geometry(), c)
        self.assertEqual(result['status'], 'neutral_only')
        self.assertEqual(result['metrics']['electron_density_m3'], 0.)
        self.assertIsNone(result['metrics']['electron_temperature_eV'])
        self.assertEqual(sum(result['balances']['power_budget_W'].values()), 0.)
        self.assertAlmostEqual(result['balances']['neutral_exit_atoms_s']*M_AR/c['mass_flow_kg_s'], 1.)

    def test_no_root_is_not_silent_rate_extrapolation(self):
        c=load(); c['absorbed_power_W']=1e6
        with self.assertRaisesRegex(NoSteadyStateError, 'supported 1--7 eV'):
            solve(geometry(), c)
        for t in (0., .99, 7.01, float('nan'), True):
            with self.assertRaises(ValueError): rates(t)

    def test_invalid_config_and_unknown_keys_are_rejected(self):
        for key, bad in [('mass_flow_kg_s', 0), ('absorbed_power_W', -1),
                         ('absorbed_power_W', True), ('neutral_temperature_K', float('nan'))]:
            c=load(); c[key]=bad
            with self.assertRaises(ValueError): validate(c)
        for key, bad in [('side_density_ratio', -1), ('exit_density_ratio', 0),
                         ('neutral_exit_transmission', 0), ('wall_recycling_fraction', 1.01)]:
            c=load(); c['losses'][key]=bad
            with self.assertRaises(ValueError): validate(c)
        c=load(); c['absorbed_powre_W']=1
        with self.assertRaises(ValueError): validate(c)

    def test_lossless_walls_and_no_recycling_limit(self):
        c=load(); c['losses'].update(side_density_ratio=0., back_density_ratio=0.)
        c['absorbed_power_W']=100.
        result=solve(geometry(), c)
        self.assertEqual(result['metrics']['exit_ion_loss_fraction'], 1.)
        c=load(); c['losses']['wall_recycling_fraction']=0; c['absorbed_power_W']=100.
        result=solve(geometry(), c); m=result['metrics']
        self.assertAlmostEqual(m['propellant_utilization']+m['wall_retained_feed_fraction']+m['neutral_exit_feed_fraction'],1.,places=10)

    def test_magnetic_and_mesh_inputs_do_not_fake_plasma_coupling(self):
        c=load(); geo=geometry(); before=copy.deepcopy(c)
        baseline=solve(geo, c)['metrics']
        geo['coils'][0]['current'] *= 100
        geo['mesh']['chamber'] *= .01
        self.assertEqual(baseline, solve(geo, c)['metrics'])
        self.assertEqual(c, before)

    def test_rate_values_against_independent_tabulation(self):
        # Independently evaluated with 50-digit Decimal arithmetic at 3 eV,
        # pinned to prevent coefficient/unit drift (not experimental rate data).
        k=rates(3.)
        self.assertAlmostEqual(k['ionization_m3_s']/1.3366534065823124e-16, 1., places=12)
        self.assertAlmostEqual(k['excitation_m3_s']/5.0327713742788654e-16, 1., places=12)


class PlasmaWorkflowTests(unittest.TestCase):
    def test_report_and_source_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'run'
            run_case(geometry(), load(), out, plots=False)
            record=json.loads((out/'run.json').read_text())
            self.assertEqual(record['status'], 'completed')
            for name, sha in record['artifacts_sha256'].items():
                self.assertEqual(digest(out/name), sha)
            for name, sha in record['source']['sha256'].items():
                self.assertEqual(digest(out/'source'/name), sha)
            for name in ('report.html','report.md','particle_budget.csv','power_budget.csv'):
                self.assertGreater((out/name).stat().st_size, 50)
            with self.assertRaises(FileExistsError): run_case(geometry(), load(), out, plots=False)

    def test_failed_solution_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'failed'; c=load(); c['absorbed_power_W']=1e6
            with self.assertRaises(NoSteadyStateError): run_case(geometry(), c, out, plots=False)
            record=json.loads((out/'run.json').read_text())
            self.assertEqual(record['status'],'failed')
            self.assertIn('NoSteadyStateError', record['error'])
            self.assertFalse((out/'plasma_summary.json').exists())

    def test_cli_without_site_packages(self):
        # -S disables site-packages: the full no-plots workflow needs only stdlib.
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'cli'
            result=subprocess.run([sys.executable,'-S','-m','helicon.plasma',
                                   '--out',str(out),'--absorbed-power','0','--no-plots'],
                                  text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIsNone(json.loads(result.stdout)['electron_temperature_eV'])


if __name__ == '__main__':
    unittest.main()
