"""Portable plots of actual finite-volume benchmark fields."""
import html
from pathlib import Path
import numpy as np


def render(out, summary, fields):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    entries = [('electron_density_m3', 'Electron density [m⁻³]'),
               ('neutral_density_m3', 'Neutral density [m⁻³]'),
               ('electron_temperature_eV', 'Electron temperature [eV]'),
               ('potential_V', 'Potential relative to first cell [V]'),
               ('q_abs_W_m3', 'Absorbed RF heating [W/m³]'),
               ('collision_frequency_s', 'Momentum collision frequency [s⁻¹]')]
    for ax, (key, title) in zip(axes.flat, entries):
        image = ax.pcolormesh(fields['z_faces_m']*100, fields['r_faces_m']*100,
                              np.asarray(fields[key]).T, shading='flat')
        fig.colorbar(image, ax=ax, shrink=.8)
        ax.set(title=title, xlabel='z [cm]', ylabel='r [cm]')
    fig.suptitle('Argon spatial fluid benchmark — '+summary['status'])
    fig.savefig(out/'spatial_fields.png', dpi=150)
    plt.close(fig)
    b = summary['balances']
    items = ''.join('<li>'+html.escape(text)+'</li>' for text in summary['limitations'])
    (out/'report.html').write_text(f'''<!doctype html><html><meta charset="utf-8">
<title>Spatial argon benchmark</title><style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:20px}}img{{width:100%}}</style>
<h1>Spatial argon benchmark: {html.escape(summary['status'])}</h1>
<p>Absorbed power: {b['absorbed_power_W']:.6g} W. Mass residual: {b['relative_mass_error']:.3g}.
Electron-energy residual: {b['relative_power_error']:.3g}.</p>
<img src="spatial_fields.png" alt="Density, temperature, potential, heating and collision-rate maps">
<p>Cell fields: <a href="plasma_fields.npz">NPZ</a>. Full diagnostics: <a href="spatial_summary.json">JSON</a>.</p>
<ul>{items}</ul></html>''')
