"""Plot plasma heating profiles exported by the prescribed-current comparison."""
import argparse
from pathlib import Path
import numpy as np
from ..records import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    record = read_json(args.run/'run.json')
    if record.get('model') != 'nagoya_prescribed_closed_v1' or record['status'] != 'completed':
        parser.error('Supply a completed Nagoya comparison')
    if args.out.exists():
        parser.error('Choose a fresh figure name')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), layout='constrained')
    cutoff = min(7, record['max_m'])
    for ax, axis, symbol in zip(axes, ('radial', 'axial'), ('r', 'z')):
        data = np.genfromtxt(args.run/f'{axis}_profiles.csv', names=True, delimiter=',')
        edges = 1000*np.r_[data['lower_m'], data['upper_m'][-1]]
        def line(key, label, color, style):
            ax.stairs(1000*data[key]/np.diff(edges), edges, label=label, color=color, linestyle=style, linewidth=1.8)
        line('through_m1_W', 'Modes ±1', '#bf6431', '-')
        if cutoff > 1:
            line(f'through_m{cutoff}_W', f'Odd modes through ±{cutoff}', '#147d79', '-')
        if record['references']:
            n = record['references'][-1]['sectors']
            line(f'prism_{n}_W', f'3D, {n} sectors', '#322d49', '--')
        ax.set(xlabel=f'{symbol} (mm)', ylabel=f'dP/d{symbol} (mW/mm)', title=f'{axis.capitalize()} deposition',
               xlim=(edges[0], edges[-1]), ylim=(0, None))
        ax.grid(axis='y', alpha=.24)
        ax.spines[['right','top']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Plasma heating at {record['rf_inputs']['frequency_Hz']/1e6:g} MHz and "
                 f"{record['current_peak_A']:g} A peak", fontsize=14)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
