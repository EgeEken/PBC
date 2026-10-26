import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_pass6.worker import ROOT,HERE,THREADS,request

INPUTS=('cat.jpg','flower.jpg','googlewiki.png')
PRESETS=('compression','balanced','quality','high_quality')
VARIANTS=('baseline','all_mse','all')


def validate(spec):
    if not isinstance(spec,dict) or set(spec)-{'images','presets','variants','repeats'}:
        raise ValueError('Invalid research specification')
    for key,allowed,limit in (('images',INPUTS,3),('presets',PRESETS,4),('variants',VARIANTS,3)):
        value=spec.get(key)
        if not isinstance(value,list) or not 1<=len(value)<=limit or any(not isinstance(x,str) or x not in allowed for x in value) or len(set(value))!=len(value):
            raise ValueError('Invalid '+key)
    if type(spec.get('repeats')) is not int or not 1<=spec['repeats']<=3 or 'baseline' not in spec['variants']:
        raise ValueError('Baseline and 1..3 repeats required')
    return spec


def run(spec):
    manifest=json.loads((HERE/'runtime_manifest.json').read_text())
    for name,value in manifest['source_lf'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()==value,name
    for name,value in manifest['assets'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name
    processes={}
    metadata={}
    rows=[]
    def evaluate(image,preset,variant,trial):
        source='baseline' if variant=='baseline' else 'integrated'
        row=request(processes[source],dict(image=image,preset=preset,trial=trial,
            overrides=dict(compute_final_mse=variant!='all')))
        row['variant']=variant
        return row
    try:
        for source in ('baseline','integrated'):
            processes[source]=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--source',source],cwd=ROOT,
                env={**os.environ,**THREADS},stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            metadata[source]=request(processes[source],dict(op='metadata'))
        warmups=[evaluate('googlewiki.png',p,v,-1) for p in spec['presets'] for v in spec['variants']]
        for trial in range(spec['repeats']):
            for i,image in enumerate(spec['images']):
                for p,preset in enumerate(spec['presets']):
                    order=spec['variants']
                    offset=(trial+i+p)%len(order)
                    for variant in order[offset:]+order[:offset]:
                        rows.append(evaluate(image,preset,variant,trial))
        for row in rows:
            baseline=next(r for r in rows if (r['image'],r['preset'],r['trial'],r['variant'])==(row['image'],row['preset'],row['trial'],'baseline'))
            assert all(row[k]==baseline[k] for k in ('stream_sha256','pixels_sha256','trajectory_sha256','mse','bpp','patches'))
        return dict(metadata=dict(spec=spec,workers=metadata,manifest=manifest,warmups=warmups,
            timing='Complete server-side calls. Two isolated persistent source workers, same pinned CPU, sequential interleaving. Independent scoring/loading/IPC excluded.'),rows=rows,exact=True)
    finally:
        for process in processes.values():
            if process.poll() is None:
                request(process,dict(op='quit'))
                process.wait(timeout=20)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec-json',required=True)
    args=parser.parse_args()
    print(json.dumps(run(validate(json.loads(args.spec_json)))))
