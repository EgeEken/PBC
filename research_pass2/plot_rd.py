import json
import os
import sys

from common import ROOT, OUT, SCREEN

os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(OUT / '__pycache__')
import matplotlib.pyplot as plt
from analyze import frontier


def plot():
    rows = json.loads((OUT / 'rd_raw_points.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, image in zip(axes.ravel(), SCREEN):
        points = [r for r in rows if r['image'] == image and r['dataset'] == 'hpt_data']
        for source, color, marker in (('release', '0.5', 's'), ('current', 'black', 'o')):
            curve = frontier([r for r in points if r['kind'] == 'default_control' and r['source'] == source])
            ax.plot([r['bpp'] for r in curve], [r['mse'] for r in curve], color=color, marker=marker, linestyle='--', label=source + ' presets')
        for limit, color in ((500000, '#167b9f'), (1000000, '#c56a16')):
            curve = frontier([r for r in points if r.get('config', {}).get('auto_downsample_max_pixels') == limit and r['source'] == 'current'])
            if curve:
                ax.plot([r['bpp'] for r in curve], [r['mse'] for r in curve], color=color, marker='o', label=f'root {limit // 1000}k measured')
        ablations = [r for r in points if r.get('experiment') == 'allocation_screen']
        ax.scatter([r['bpp'] for r in ablations], [r['mse'] for r in ablations], marker='x', color='#973b8c', label='allocation ablations')
        ax.set_title(image)
        ax.set_xlabel('Complete file bits / original pixel')
        ax.set_ylabel('Original-size decoded RGB MSE')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9)
    fig.suptitle('PBC single-stage measured RD points; lines connect observed nondominated neighbors')
    fig.tight_layout(rect=(0, .055, 1, .96))
    fig.savefig(OUT / 'quality_rd_screen.png', dpi=180)
    fig.savefig(OUT / 'quality_rd_screen.svg')
    plt.close(fig)


if __name__ == '__main__':
    plot()
