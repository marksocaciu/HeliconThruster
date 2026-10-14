"""Portable reports for the global plasma model; plotting is optional."""
import csv
from html import escape
from pathlib import Path


def render(out, result, plots=True):
    out = Path(out)
    c, m, g = result['input'], result['metrics'], result['geometry']
    balance = result['balances']
    power = balance['power_budget_W']
    flow = balance['ion_rates_s']
    plot_runtime = None
    if plots:
        try:
            import matplotlib
        except ImportError as exc:
            raise ImportError('Install requirements-plasma.txt for figures, or rerun in a new folder with --no-plots') from exc
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plot_runtime = {'matplotlib': matplotlib.__version__}
        labels = {'ionization': 'Ionization', 'excitation': 'Excitation',
                  'elastic_gas_heating': 'Gas heating', 'side_wall': 'Side wall',
                  'back_wall': 'Back wall', 'exit': 'Open exit'}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        axes[0].barh([labels[k] for k in power], list(power.values()), color='#247a91')
        axes[0].set(xlabel='Power (W)', title='Absorbed-power budget')
        particle_values = [flow['side'], flow['back'], flow['exit'], balance['neutral_exit_atoms_s']]
        axes[1].barh(['Side-wall ions', 'Back-wall ions', 'Exit ions', 'Exit neutrals'],
                     [v/1e19 for v in particle_values], color=['#b56645', '#b56645', '#247a91', '#80947c'])
        axes[1].set(xlabel=r'Particle rate ($10^{19}$ s$^{-1}$)', title='Loss and exhaust rates')
        for ax in axes:
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=.2)
        fig.suptitle('Argon global source — prescribed losses, illustrative case', fontsize=12)
        fig.savefig(out/'plasma_budget.png', dpi=170)
        plt.close(fig)

    with (out/'power_budget.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['channel', 'power_W', 'fraction_of_absorbed_power'])
        for name, value in power.items():
            writer.writerow([name, value, value/c['absorbed_power_W'] if c['absorbed_power_W'] else ''])
    with (out/'particle_budget.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['channel', 'particles_per_second'])
        writer.writerows([('feed_neutrals', g['injected_atoms_s']),
                         ('side_wall_ions', flow['side']), ('back_wall_ions', flow['back']),
                         ('exit_ions', flow['exit']), ('exit_neutrals', balance['neutral_exit_atoms_s']),
                         ('wall_recycled_neutrals', balance.get('wall_recycled_atoms_s', 0.)),
                         ('wall_retained_atoms', balance.get('wall_retained_atoms_s', 0.))])
    def value(x):
        return 'undefined (no plasma)' if x is None else f'{x:.6g}'
    lines = ['# Global argon discharge', '',
             '**Illustrative model with prescribed loss coefficients; not experimental validation.**', '',
             f"State: {result['status']}. Absorbed power: {c['absorbed_power_W']:g} W. "
             f"Argon feed: {c['mass_flow_kg_s']*1e6:g} mg/s.", '',
             '## Plasma state', '', '| Quantity | Value |', '|---|---:|']
    quantities = [('Mean electron density (m^-3)', m['electron_density_m3']),
                  ('Electron temperature (eV)', m['electron_temperature_eV']),
                  ('Neutral density (m^-3)', m['neutral_density_m3']),
                  ('Neutral pressure (Pa)', m['neutral_pressure_Pa']),
                  ('Ionized number fraction (%)', 100*m['ionized_number_fraction']),
                  ('Exit propellant utilization (%)', 100*m['propellant_utilization']),
                  ('Exit ion rate (s^-1)', m['exit_ion_rate_s']),
                  ('Equivalent exit ion current (A)', m['exit_equivalent_ion_current_A'])]
    lines += [f'| {name} | {value(x)} |' for name, x in quantities]
    lines += ['', 'The equivalent ion current counts outgoing ions; it is not the net plume current.', '',
              '## Power accounting', '', '| Channel | W |', '|---|---:|']
    lines += [f'| {name} | {value(x)} |' for name, x in power.items()]
    lines += ['', 'Wall-power entries contain kinetic/sheath losses. Recombination energy is already '
              'accounted for in the ionization sink; these entries are not a complete wall thermal-load model.', '',
              '## Conservation', '', '| Relative residual | Value |', '|---|---:|']
    lines += [f'| {name} | {x:.3e} |' for name, x in balance['relative_residuals'].items()]
    lines += ['', '## Assumed boundary-to-volume-mean density ratios', '',
              '| Boundary | Ratio |', '|---|---:|']
    lines += [f"| {name} | {c['losses'][name+'_density_ratio']:g} |" for name in ('side', 'back', 'exit')]
    lines += ['', f"Wall recycling fraction: {c['losses']['wall_recycling_fraction']:g}. "
              f"Neutral exit transmission: {c['losses']['neutral_exit_transmission']:g}.", '',
              'Changing a coil current alone will not change this model: magnetic confinement is not yet coupled.', '',
              '## Scope', '']
    lines += ['- '+item for item in result['limitations']]
    lines += ['- '+item for item in result['diagnostics'].get('warnings', [])]
    lines += ['', 'Source equations, rate references, and assumptions are saved in '
              '[the model specification](source/docs/plasma-model.md).', '']
    (out/'report.md').write_text('\n'.join(lines))
    # Use explicit HTML tables rather than adding a Markdown-renderer dependency.
    def table(rows):
        return '<table>'+''.join('<tr><th>'+escape(str(a))+'</th><td>'+escape(str(b))+'</td></tr>'
                                for a, b in rows)+'</table>'
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    html += '<title>Global argon discharge</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#24333c}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{text-align:left;padding:8px;border-bottom:1px solid #dde3e6}td{text-align:right}img{width:100%}.note{background:#f4eee0;padding:16px}li{margin:8px 0}</style>'
    html += '<h1>Global argon discharge</h1><p class="note">Illustrative model with prescribed losses. '
    html += 'Conservation checks verify the calculation; they do not establish experimental accuracy.</p>'
    html += f"<p>{c['absorbed_power_W']:g} W absorbed · {c['mass_flow_kg_s']*1e6:g} mg/s argon · {c['neutral_temperature_K']:g} K neutrals</p>"
    html += table([(a, value(b)) for a, b in quantities])
    if plots:
        html += '<img src="plasma_budget.png" alt="Power and particle budgets">'
    html += '<h2>Power budget (W)</h2>'+table([(k, value(v)) for k, v in power.items()])
    html += '<p>Equivalent ion current is not net plume current. Wall-power entries exclude the binding energy already counted in ionization.</p>'
    html += '<h2>Conservation residuals</h2>'+table([(k, f'{v:.3e}') for k, v in balance['relative_residuals'].items()])
    html += '<h2>Prescribed loss inputs</h2>'+table(c['losses'].items())
    html += '<h2>Model scope</h2><ul>'+''.join('<li>'+escape(x)+'</li>' for x in result['limitations'])+'</ul>'
    html += '<ul>'+''.join('<li>'+escape(x)+'</li>' for x in result['diagnostics'].get('warnings', []))+'</ul>'
    html += '<p><a href="plasma_summary.json">Full result</a> · <a href="report.md">Text report</a> · <a href="source/docs/plasma-model.md">Equations and references</a></p></html>'
    (out/'report.html').write_text(html)
    return plot_runtime
