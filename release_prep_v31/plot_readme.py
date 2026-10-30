from pathlib import Path
import difflib
import hashlib
import json
import shutil
import statistics as stats
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'artifacts/ui-polish'
OUT.mkdir(parents=True,exist_ok=True)
summary=json.loads((HERE/'artifacts/summary.json').read_text())
assert summary['measured_pairs']==900 and summary['exact_cases']==100
assets=ROOT/'docs/assets/v31'
assets.mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#ddd4d0',
    'axes.labelcolor':'#70665f','text.color':'#292624','xtick.color':'#524a46',
    'ytick.color':'#70665f','figure.facecolor':'#faf8f6','axes.facecolor':'#faf8f6'})
presets=('compression','balanced','quality','high_quality')
labels=('Compression','Balanced','Quality','High Quality')
colors=('#cf514a','#902e2d')
values={}
for field,title,expected in (('encode_seconds','Encoding',41.03),('decode_seconds','Decoding',53.59)):
    rows=[]
    for preset in presets:
        cases=[c for c in summary['cases'] if c['preset']==preset]
        assert len(cases)==25
        a=stats.mean(c['variants']['original'][field]['median'] for c in cases)
        b=stats.mean(c['variants']['candidate'][field]['median'] for c in cases)
        assert a==summary['presets'][preset][field]['mean_case_medians']['original']
        assert b==summary['presets'][preset][field]['mean_case_medians']['candidate']
        rows.append(dict(preset=preset,original_seconds=a,final_seconds=b,reduction_percent=100*(1-b/a)))
    overall=100*(1-stats.mean(r['final_seconds'] for r in rows)/stats.mean(r['original_seconds'] for r in rows))
    assert round(overall,2)==expected
    fig,ax=plt.subplots(figsize=(12,5.6),dpi=180)
    fig.subplots_adjust(top=.72,bottom=.18,left=.08,right=.97)
    fig.text(.08,.91,title.upper(),fontsize=22,weight='bold')
    fig.text(.97,.91,f'{overall:.2f}% lower time',fontsize=21,weight='bold',color=colors[1],ha='right')
    fig.text(.08,.85,'PBC V3.0 versus final PBC V3.1',fontsize=11,color='#70665f')
    x=np.arange(4)
    a=np.array([r['original_seconds'] for r in rows])
    b=np.array([r['final_seconds'] for r in rows])
    ax.bar(x-.16,a,.29,color=colors[0],label='PBC V3.0')
    ax.bar(x+.16,b,.29,color=colors[1],label='PBC V3.1')
    ax.set_ylim(0,max(a)*1.32)
    ax.set_ylabel('Execution time (seconds)')
    ax.set_xticks(x,labels)
    ax.tick_params(axis='both',length=0,pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#e8e1dd')
    for i,row in enumerate(rows):
        for offset,value,color in ((-.16,a[i],colors[0]),(.16,b[i],colors[1])):
            ax.text(i+offset,value+max(a)*.025,f'{value:.4f}',ha='center',fontsize=10,weight='bold',color=color)
        ax.text(i,a[i]+max(a)*.16,f"−{row['reduction_percent']:.2f}%",ha='center',fontsize=12,weight='bold',color=colors[1])
    ax.legend(loc='upper left',bbox_to_anchor=(-.01,-.14),ncol=2,frameon=False)
    fig.text(.97,.035,'25 images · 4 presets · median of 3 repetitions per case',ha='right',fontsize=9,color='#70665f')
    name='encode-time.png' if field=='encode_seconds' else 'decode-time.png'
    fig.savefig(assets/name)
    plt.close(fig)
    values[field]=dict(overall_reduction=overall,presets=rows)
fig,axes=plt.subplots(1,2,figsize=(12,4.8),dpi=180)
fig.subplots_adjust(top=.74,bottom=.20,left=.08,right=.97,wspace=.28)
fig.text(.08,.9,'THE V3.1 UPDATE',fontsize=20,weight='bold')
fig.text(.08,.83,'Three implementations, one matched benchmark',fontsize=11,color='#70665f')
for ax,field,title in zip(axes,('encode_seconds','decode_seconds'),('Encode','Decode')):
    nums=[summary['overall'][field]['mean_case_medians'][m] for m in ('original','published','candidate')]
    ax.bar(range(3),nums,color=('#dfa19b','#cf514a','#902e2d'),width=.55)
    ax.set_xticks(range(3),('V3.0','Published\nV3.1','Final\nV3.1'))
    ax.set_ylabel('Execution time (seconds)')
    ax.set_title(title,loc='left',weight='bold',pad=14)
    ax.set_ylim(0,max(nums)*1.25)
    ax.tick_params(axis='both',length=0,pad=8)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#e8e1dd')
    for i,v in enumerate(nums):ax.text(i,v+max(nums)*.04,f'{v:.6f}',ha='center',fontsize=10,weight='bold')
fig.text(.08,.055,'Final vs published V3.1: 10.28% lower encode time; decode effectively unchanged.',fontsize=10,color='#70665f')
fig.savefig(assets/'three-way-times.png')
plt.close(fig)
original=(HERE/'artifacts/public-main/README.md').read_text(encoding='utf-8')
section="""## V3.1

### Performance optimizations and a standalone Windows GUI, with no RD cost

V3.1 uses **41.03% less encoding time** and **53.59% less decoding time** than V3.0. The updated implementation also uses **10.28% less encoding time** than the previously published V3.1; its additional decode-time change is effectively neutral. All 100 tested image/preset cases produced identical compressed streams, reconstructed pixels and compressed sizes.

The combined V3.1 download includes a standalone Windows GUI with Encode, Decode and Registry tabs, alongside the Python library.

<img width="100%" alt="PBC V3.0 versus final V3.1 encoding times by preset; 41.03 percent lower overall execution time" src="docs/assets/v31/encode-time.png" />

<img width="100%" alt="PBC V3.0 versus final V3.1 decoding times by preset; 53.59 percent lower overall execution time" src="docs/assets/v31/decode-time.png" />

<img width="100%" alt="Mean case-median encode and decode times for original V3.0, published V3.1 and final V3.1" src="docs/assets/v31/three-way-times.png" />

Benchmarked on 25 images across four presets, using three measured repetitions per case. Results average the per-case median execution times. These are reductions in execution time, not throughput increases. [Benchmark details](docs/benchmarks/v31.md).
"""
prefix=original[:original.index('## V3.1')]
updated=prefix+section
review=HERE/'public_readme'
review.mkdir(exist_ok=True)
(review/'README.md').write_text(updated,encoding='utf-8',newline='\n')
patch=''.join(difflib.unified_diff(original.splitlines(keepends=True),updated.splitlines(keepends=True),fromfile='a/README.md',tofile='b/README.md'))
(HERE/'README-main.patch').write_text(patch,encoding='utf-8',newline='\n')
report=ROOT/'docs/benchmarks/v31.md'
report.parent.mkdir(exist_ok=True)
source=(HERE/'BENCHMARK.md').read_text(encoding='utf-8')
source=source.replace('Sources: provenance.json and prepare.py.','Sources are the authentic 3.0 release (2e59f1b), published V3.1 ZIP (SHA-256 edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede), and final Pass 28 source tree fd077ba70d6855e043721953d77a65ed43bfd5ea.')
source=source.replace('Topology is retained in provenance.json.','The machine has six SMT P-cores and eight E-cores.')
start=source.index('Detailed case medians,')
source=source[:start]+"""Raw repetition-level timings, configurations, source/input identities and hashes are retained locally. Raw measurement SHA-256: 98a7dbeba061b0fd6441171777c79be3bdcb6075a78b80b561aa438cf56ac1d8. Python 3.11.16, NumPy 2.4.6, Numba 0.68.0 and Pillow 11.3.0.

References: authentic V3.0 release commit 2e59f1b06980aa467da0cde94fa33c3f9a0d4d51; actual published V3.1 ZIP SHA-256 edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede; optimized source derived from final Pass 28 tree fd077ba70d6855e043721953d77a65ed43bfd5ea. No compression algorithm, model or PBC3 v0 stream change was made. Time reductions use 100 × (1 − new / old); these are not throughput increases. The 0.37% additional decode reduction versus published V3.1 is treated as neutral. Negative per-case reductions are regressions; three repetitions do not establish statistical significance.
"""
report.write_text(source,encoding='utf-8',newline='\n')
(OUT/'chart-values.json').write_text(json.dumps(values,indent=2)+'\n')
assert updated[:updated.index('## V3.1')]==prefix
print('README patch, public benchmark report, and three verified charts prepared')


review_assets=review/'docs/assets/v31'
review_assets.mkdir(parents=True,exist_ok=True)
for image in assets.glob('*.png'):
    shutil.copy2(image,review_assets/image.name)
(review/'docs/benchmarks').mkdir(parents=True,exist_ok=True)
shutil.copy2(report,review/'docs/benchmarks/v31.md')
