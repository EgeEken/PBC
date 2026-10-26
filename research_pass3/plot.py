import os
import json

from support import OUT, PASS2, SCREEN, read
from analyze import frontier

os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(OUT / '__pycache__')
import matplotlib.pyplot as plt


def draw(dataset, names, filename, columns):
    points = read('allocation_points')
    old = json.loads((PASS2 / 'rd_raw_points.json').read_text())
    nrows = (len(names) + columns - 1) // columns
    fig, axes = plt.subplots(nrows, columns, figsize=(5 * columns, 4.3 * nrows), squeeze=False)
    for ax, image in zip(axes.ravel(), names):
        selected = [r for r in points if r['dataset'] == dataset and r['image'] == image]
        for source, color in (('release', '0.6'), ('current', 'black')):
            defaults = [r for r in selected if r['label'].startswith(source + '_') and r.get('experiment') is None]
            curve = frontier(defaults)
            ax.plot([r['bpp'] for r in curve], [r['mse'] for r in curve], marker='s', linestyle='--', color=color, label=source + ' presets')
        existing = frontier([r for r in old if r['dataset'] == dataset and r['image'] == image])
        ax.plot([r['bpp'] for r in existing], [r['mse'] for r in existing], linestyle=':', color='0.4', linewidth=1, label='previous measured envelope')
        ordinary = next((r for r in selected if r['label'] == 'q500p35_none'), None)
        if ordinary:
            ax.scatter([ordinary['bpp']], [ordinary['mse']], color='black', marker='D', label='500k/35 control')
        for mode, color, marker in (('fixed', '#9c3e8b', 'x'), ('integrated', '#167d9a', 'o')):
            chosen = [r for r in selected if r['label'].startswith('q500p35_' + mode)]
            curve = frontier(chosen)
            ax.plot([r['bpp'] for r in curve], [r['mse'] for r in curve], marker=marker, color=color, label='500k/35 ' + mode)
        ax.set_title(image, fontsize=10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Complete-file bpp / original pixel', fontsize=9)
        ax.set_ylabel('Decoded RGB MSE', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.2)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('Measured PBC operating points; connecting lines are guides, no extrapolation', fontsize=12)
    fig.tight_layout(rect=(0, .07 if nrows < 3 else .035, 1, .96 if nrows < 3 else .98))
    fig.savefig(OUT / (filename + '.png'), dpi=160)
    fig.savefig(OUT / (filename + '.svg'))
    plt.close(fig)


if __name__ == '__main__':
    draw('hpt_data', SCREEN, 'rd_screen', 3)
    full = read('allocation_full')
    if full:
        for dataset, columns in (('hpt_data', 4), ('hpt_data_val', 3)):
            names = sorted({r['image'] for r in full if r['dataset'] == dataset})
            draw(dataset, names, 'rd_' + dataset, columns)
