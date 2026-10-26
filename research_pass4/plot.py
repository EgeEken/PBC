import math
import os

from support import OUT, read
os.environ['MPLCONFIGDIR']=str(OUT/'.mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,LogFormatterSciNotation,NullFormatter
from analyze import frontier


def main():
    points=read('matched_rate_points')
    for dataset in ('hpt_data','hpt_data_val'):
        images=sorted({r['image'] for r in points if r['dataset']==dataset})
        columns=4 if dataset=='hpt_data' else 3
        rows=math.ceil(len(images)/columns)
        fig,axes=plt.subplots(rows,columns,figsize=(columns*4.2,rows*3.4),squeeze=False)
        for ax,name in zip(axes.flat,images):
            group=[r for r in points if r['dataset']==dataset and r['image']==name]
            prior=[r for r in group if r['experiment']=='prior']
            fresh=[r for r in group if r['experiment']=='fresh_control']
            candidate=next(r for r in group if r['experiment']=='frozen_allocator')
            f=frontier(prior)
            ax.scatter([r['bpp'] for r in prior],[r['mse'] for r in prior],s=10,color='.65',label='Prior ordinary points')
            ax.plot([r['bpp'] for r in f],[r['mse'] for r in f],color='.4',linestyle=':',linewidth=1,label='Prior frontier guide')
            ax.scatter([r['bpp'] for r in fresh],[r['mse'] for r in fresh],s=27,color='#24668c',marker='o',label='Fresh ordinary controls')
            ax.scatter([candidate['bpp']],[candidate['mse']],s=115,color='#af3d48',marker='*',zorder=6,label='Frozen allocator 0.4')
            ax.axvspan(candidate['bpp']*.95,candidate['bpp']*1.05,color='#af3d48',alpha=.09,label='5% rate tolerance')
            ax.set(xscale='log',yscale='log',title=name,xlabel='Complete-file bpp / original pixel',ylabel='Decoded original RGB MSE')
            ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
            ax.xaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False,minor_thresholds=(float('inf'),float('inf'))))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='x',labelsize=8)
            ax.grid(alpha=.15)
        for ax in list(axes.flat)[len(images):]:
            ax.set_visible(False)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=9)
        fig.suptitle('Fresh ordinary encodes versus frozen allocator; lines are interpolation guides',fontsize=12)
        fig.tight_layout(rect=(0,.065,1,.965))
        for suffix in ('png','svg'):
            fig.savefig(OUT/f'rd_{dataset}.{suffix}',dpi=180)
        plt.close(fig)


if __name__=='__main__':
    main()
