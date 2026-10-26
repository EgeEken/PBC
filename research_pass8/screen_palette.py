import hashlib
import json
import os
from pathlib import Path
from statistics import mean,median
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research_pass6.worker import THREADS,request


def main():
    gate=json.loads((HERE/'palette_feasibility.json').read_text())
    assert gate['summary']['98']['gate_pass']
    images=('cat.jpg','flower.jpg','monet.jpg','mario64.png','eiffel.png','googlewiki.png')
    modes=('control','robust')
    processes={}; metadata={}; rows=[]
    try:
        for mode in modes:
            processes[mode]=subprocess.Popen([sys.executable,str(HERE/'palette_worker.py'),'--mode',mode],cwd=ROOT,
                env={**os.environ,**THREADS},stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            metadata[mode]=request(processes[mode],dict(op='metadata'))
        warmups=[request(processes[m],dict(image='googlewiki.png',preset=p)) for p in ('quality','high_quality') for m in modes]
        with (HERE/'palette_screen_raw.jsonl').open('w') as raw:
            for trial in range(3):
                for i,image in enumerate(images):
                    for j,preset in enumerate(('quality','high_quality')):
                        for mode in (modes if (trial+i+j)%2==0 else modes[::-1]):
                            row=request(processes[mode],dict(image=image,preset=preset,trial=trial,trajectory=True))
                            row['mode']=mode
                            if mode=='robust':
                                row['encode_seconds']=row['stats']['encode_scope_seconds']
                            rows.append(row); raw.write(json.dumps(row)+'\n'); raw.flush()
                        print(trial,image,preset,flush=True)
        compact=[]
        for image in images:
            for preset in ('quality','high_quality'):
                baseline=next(r for r in rows if (r['image'],r['preset'],r['mode'])==(image,preset,'control'))
                for mode in modes:
                    group=[r for r in rows if (r['image'],r['preset'],r['mode'])==(image,preset,mode)]
                    assert len({r['stream_sha256'] for r in group})==len({r['pixels_sha256'] for r in group})==1
                    assert all(r['preview_equal'] for r in group)
                    row={k:v for k,v in group[0].items() if k not in ('trajectory','trial','source')}
                    for metric in ('encode_seconds','decode_seconds'):
                        row[metric]=median(r[metric] for r in group)
                        row[metric+'_samples']=[r[metric] for r in group]
                    if mode=='robust':
                        row['stats']['palette_seconds']=median(r['stats']['palette_seconds'] for r in group)
                    row['first_geometry_change']=next((i for i,(a,b) in enumerate(zip(baseline['trajectory'],group[0]['trajectory'])) if a[:7]!=b[:7]),None)
                    row['first_quantized_change']=next((i for i,(a,b) in enumerate(zip(baseline['trajectory'],group[0]['trajectory'])) if a!=b),None)
                    row['patch_count_changed']=row['patches']!=baseline['patches']
                    compact.append(row)
        provenance=dict(source_lf={p:hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in
            ('research_pass8/palette.py','research_pass8/palette_scope.py','research_pass8/palette_worker.py','research_pass8/screen_palette.py','research_pass6/worker.py','PBC3.py','pbc3_ops.py')},
            inputs={im:hashlib.sha256((ROOT/'hpt_data'/im).read_bytes()).hexdigest() for im in images},
            model_sha256=hashlib.sha256((ROOT/'patch_policy.npz').read_bytes()).hexdigest(),
            raw_sha256=hashlib.sha256((HERE/'palette_screen_raw.jsonl').read_bytes()).hexdigest())
        (HERE/'palette_screen.json').write_text(json.dumps(dict(percentile=98,repeats=3,rows=compact,
            metadata=metadata,warmups=warmups,provenance=provenance),indent=2)+'\n',newline='\n')
        print('Saved 72 measured calls',flush=True)
    finally:
        for p in processes.values():
            if p.poll() is None:
                request(p,dict(op='quit')); p.wait(timeout=20)


if __name__=='__main__':
    main()
