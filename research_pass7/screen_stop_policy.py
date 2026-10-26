import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research_pass6.worker import THREADS,request


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnose',action='store_true')
    args=parser.parse_args()
    images=('cat.jpg','flower.jpg','monet.jpg','mario64.png','eiffel.png','googlewiki.png')
    cases=[dict(image=im,preset=p) for im in images for p in ('quality','high_quality')]
    cases += [dict(image='cat.jpg',preset='compression'),dict(image='googlewiki.png',preset='balanced')]
    modes=('control','diagnose') if args.diagnose else ('control','S1','S2')
    repeats=1 if args.diagnose else 3
    processes={}
    rows=[]
    metadata={}
    stem='diagnosis' if args.diagnose else 'stop_screen'
    try:
        for mode in modes:
            processes[mode]=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--mode',mode],cwd=ROOT,
                env={**os.environ,**THREADS},stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            metadata[mode]=request(processes[mode],dict(op='metadata'))
        warmups=[request(processes[m],dict(image='googlewiki.png',preset=p)) for p in ('quality','high_quality') for m in modes]
        with (HERE/(stem+'_raw.jsonl')).open('w') as raw:
            for trial in range(repeats):
                for i,case in enumerate(cases):
                    offset=(trial+i)%len(modes)
                    for mode in modes[offset:]+modes[:offset]:
                        row=request(processes[mode],dict(**case,trial=trial,trajectory=True))
                        row['mode']=mode
                        if mode!='control':
                            row['encode_seconds']=row['stats']['encode_scope_seconds']
                        raw.write(json.dumps(row)+'\n'); raw.flush()
                        rows.append(row)
                    print(trial,case,flush=True)
        compact=[]
        for case in cases:
            base=next(r for r in rows if r['mode']=='control' and all(r[k]==v for k,v in case.items()))
            for mode in modes:
                group=[r for r in rows if r['mode']==mode and all(r[k]==v for k,v in case.items())]
                assert len({r['stream_sha256'] for r in group})==1
                row={k:v for k,v in group[0].items() if k not in ('trajectory','trial','source')}
                for metric in ('encode_seconds','decode_seconds'):
                    row[metric]=median(r[metric] for r in group)
                    row[metric+'_samples']=[r[metric] for r in group]
                row['first_trajectory_change']=next((i for i,(a,b) in enumerate(zip(base['trajectory'],group[0]['trajectory'])) if a!=b),None)
                row['patch_count_changed']=row['patches']!=base['patches']
                if args.diagnose:
                    assert row['stream_sha256']==base['stream_sha256'] and row['pixels_sha256']==base['pixels_sha256']
                assert row['preview_equal']
                compact.append(row)
        provenance=dict(source_lf={p:hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in
            ['PBC3.py','learned_filler.py','pbc3_ops.py','pbc3_heads.py','pbc3_stream.py','research_pass7/stop_policy.py','research_pass7/worker.py','research_pass7/screen_stop_policy.py','research_pass6/worker.py']},
            inputs={im:hashlib.sha256((ROOT/'hpt_data'/im).read_bytes()).hexdigest() for im in images},
            model_sha256=hashlib.sha256((ROOT/'patch_policy.npz').read_bytes()).hexdigest(),
            raw_sha256=hashlib.sha256((HERE/(stem+'_raw.jsonl')).read_bytes()).hexdigest())
        result=dict(config=dict(cases=cases,modes=modes,repeats=repeats,second_proposal_respects_stop=True),
            metadata=metadata,provenance=provenance,warmup_calls=len(warmups),rows=compact)
        (HERE/(stem+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print('Saved',stem,flush=True)
    finally:
        for p in processes.values():
            if p.poll() is None:
                request(p,dict(op='quit')); p.wait(timeout=20)


if __name__=='__main__':
    main()
