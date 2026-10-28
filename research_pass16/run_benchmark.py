import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research_pass15.run_integral_screen import send, stop, PRESETS, THREADS, IMAGES
from pbc3_types import PBC3Config
RUNTIME=('PBC3.py','pbc3_ops.py','pbc3_features.py','learned_filler.py','pbc3_heads.py','pbc3_types.py','pbc3_stream.py','pbc3_trace.py','patch_policy.npz')
BASELINE='c0d0339a509ceb4027574fc388cef18a643b14e5'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    source=HERE/'baseline'
    source.mkdir(exist_ok=True)
    for name in RUNTIME:
        expected=subprocess.check_output(['git','show',BASELINE+':'+name],cwd=ROOT)
        path=source/name
        if path.exists():
            assert path.read_bytes()==expected, 'baseline snapshot drift: '+name
        else:
            path.write_bytes(expected)
    if args.prepare_only:
        print('Immutable Pass 15 source snapshot verified'); return
    image_paths=sorted(p for folder in ('hpt_data','hpt_data_val') for p in (ROOT/folder).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')) if args.full else [ROOT/'hpt_data'/n for n in IMAGES]
    repeats=1 if args.full else 3
    output=HERE/('full.json' if args.full else 'screen.json')
    workers={}
    for name in ('baseline','candidate'):
        worker_source=source if name=='baseline' else ROOT
        env={**os.environ,**THREADS,'PBC_CPU':'12','PBC_IMAGES':json.dumps([str(p) for p in image_paths]),'PBC_SOURCE':str(worker_source),'PYTHONPATH':str(worker_source),'PBC_FEATURE_CANDIDATE':str(int(name=='candidate')),'PYTHONUNBUFFERED':'1'}
        workers[name]=subprocess.Popen([sys.executable,str(HERE/'worker.py')],cwd=worker_source,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        workers[name]._name=name
    rows=[]; diagnostics=[]
    equal_keys=('stream_base64','stream_sha256','ordered_body_sha256','decoded_rgb_sha256','patch_count','bytes','bpp','internal_mse','rgb_mse','size')
    try:
        for name,worker in workers.items():
            response=send(worker,{'op':'warm'})
            assert response.get('ready'),(name,response)
        # Diagnose before measuring; the full gate diagnoses every image/preset.
        for i in range(len(image_paths) if args.full else 1):
            for preset in PRESETS:
                pair={name:send(worker,{'op':'diagnose','image_index':i,'preset':preset}) for name,worker in workers.items()}
                for key in equal_keys+('feature_sha256','proposal_rng_sha256','policy_sha256','counts'):
                    assert pair['baseline'][key]==pair['candidate'][key],(image_paths[i].name,preset,key)
                for row in pair.values(): row.pop('stream_base64')
                diagnostics.append(pair)
            print('diagnosed',image_paths[i].name,flush=True)
        for rep in range(repeats):
            for i,path in enumerate(image_paths):
                for preset in PRESETS:
                    pair={}
                    order=('baseline','candidate') if (rep+i)%2==0 else ('candidate','baseline')
                    for name in order:
                        row=send(workers[name],{'op':'measure','image_index':i,'preset':preset})
                        pair[name]=row
                    for key in equal_keys:
                        assert pair['baseline'][key]==pair['candidate'][key],(path.name,preset,key)
                    for name,row in pair.items():
                        row.pop('stream_base64'); rows.append({**row,'variant':name,'rep':rep})
                print(f'repetition {rep+1}/{repeats}: {path.name}',flush=True)
    finally:
        for worker in workers.values(): stop(worker)
    summary={}; cases=[]
    for path in image_paths:
        for preset in PRESETS:
            groups={name:[r['encode_seconds'] for r in rows if r['image']==path.name and r['preset']==preset and r['variant']==name] for name in workers}
            a,b=[statistics.median(groups[n]) for n in workers]
            cases.append({'image':path.name,'preset':preset,'baseline':a,'candidate':b,'reduction_percent':100*(1-b/a),'cv_percent':{n:100*statistics.pstdev(v)/statistics.mean(v) for n,v in groups.items()}})
    for preset in (*PRESETS,'overall'):
        selected=[r for r in cases if preset=='overall' or r['preset']==preset]
        a,b=[statistics.mean(r[n] for r in selected) for n in workers]
        summary[preset]={'baseline':a,'candidate':b,'reduction_percent':100*(1-b/a),'case_wins':sum(r['candidate']<r['baseline'] for r in selected),'cases':len(selected)}
    import numpy,numba,PIL,psutil
    result={'baseline_commit':BASELINE,'baseline_id':'pbc31_integral_fused','candidate_id':'pbc31_feature_histogram','python':sys.version,'numpy':numpy.__version__,'numba':numba.__version__,'pillow':PIL.__version__,'platform':platform.platform(),'cpu':12,'thread_env':THREADS,'warmups':list(PRESETS),'repetitions':repeats,'source_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in RUNTIME},'candidate_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('feature_candidate.py','check_features.py')},'image_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in image_paths},'configs':{p:dataclasses.asdict(getattr(PBC3Config,p)()) for p in PRESETS},'all_streams_identical':True,'diagnostics':diagnostics,'rows':rows,'cases':cases,'summary':summary}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
