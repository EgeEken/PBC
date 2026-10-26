import json
import os
from pathlib import Path
from statistics import mean,median
import subprocess
import sys

from worker import ROOT,HERE,THREADS,request


def main():
    images=('cat.jpg','flower.jpg','monet.jpg','mario64.png','eiffel.png','googlewiki.png')
    presets=('quality','high_quality')
    sources=('integrated','init','iter')
    processes={}
    metadata={}
    rows=[]
    try:
        for source in sources:
            processes[source]=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--source',source],cwd=ROOT,
                env={**os.environ,**THREADS},stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            metadata[source]=request(processes[source],dict(op='metadata'))
        warmups=[request(processes[s],dict(image='googlewiki.png',preset=p)) for p in presets for s in sources]
        for trial in range(3):
            for i,image in enumerate(images):
                for j,preset in enumerate(presets):
                    offset=(trial+i+j)%3
                    for source in sources[offset:]+sources[:offset]:
                        row=request(processes[source],dict(image=image,preset=preset,trial=trial,trajectory=True))
                        if source!='integrated':
                            row['encode_seconds']=row['stats']['encode_scope_seconds']
                        rows.append(row)
                    print(trial,image,preset,flush=True)
        (HERE/'refinement_raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        compact=[]
        for image in images:
            for preset in presets:
                baseline=next(r for r in rows if (r['image'],r['preset'],r['source'])==(image,preset,'integrated'))
                for source in sources:
                    group=[r for r in rows if (r['image'],r['preset'],r['source'])==(image,preset,source)]
                    assert len({r['stream_sha256'] for r in group})==1
                    row={k:v for k,v in group[0].items() if k not in ('trajectory','trial')}
                    for metric in ('encode_seconds','decode_seconds'):
                        row[metric]=median(r[metric] for r in group)
                        row[metric+'_mean']=mean(r[metric] for r in group)
                    a,b=baseline['trajectory'],group[0]['trajectory']
                    row['first_geometry_change']=next((i for i,(x,y) in enumerate(zip(a,b)) if x[:7]!=y[:7]),None)
                    row['first_quantized_change']=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
                    row['patch_count_changed']=len(a)!=len(b)
                    compact.append(row)
        summary={}
        for source in ('init','iter'):
            pairs=[]
            for row in compact:
                if row['source']!=source:
                    continue
                base=next(r for r in compact if (r['image'],r['preset'],r['source'])==(row['image'],row['preset'],'integrated'))
                pairs.append({k:100*(row[k]/base[k]-1) for k in ('mse','bpp','encode_seconds','decode_seconds')})
            summary[source]=dict(mean_case_change={k:mean(p[k] for p in pairs) for k in pairs[0]},
                mse_wins=sum(p['mse']<0 for p in pairs),rd_wins=sum(p['mse']<0 and p['bpp']<=0 for p in pairs),
                cases=len(pairs),worst_mse_change=max(p['mse'] for p in pairs),
                accepted=sum(r['stats'].get('accepted',0) for r in compact if r['source']==source),
                attempted=sum(r['stats'].get('attempted',0) for r in compact if r['source']==source))
        (HERE/'refinement_screen.json').write_text(json.dumps(dict(config=dict(images=images,presets=presets,repeats=3,
            strength=.5,iterative_attempt_cap=3,iterative_min_grid_bits=8192,iterative_cell_greater_than=1),
            metadata=metadata,warmup_rows=len(warmups),rows=compact,summary=summary),indent=2)+'\n')
        print(json.dumps(summary,indent=2))
    finally:
        for process in processes.values():
            if process.poll() is None:
                request(process,dict(op='quit'))
                process.wait(timeout=20)


if __name__=='__main__':
    main()
