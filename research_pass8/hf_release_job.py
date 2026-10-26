import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from worker import THREADS

INPUTS=('cat.jpg','cathedral.png','coast.jpg','ducklings.jpg','eiffel.png','field.jpg','flag.jpg','flower.jpg','flowers.jpg','googlewiki.png','mario64.png','monet.jpg','monkey.png','panorama.png','park.jpg','perlman.png','ratedistortion.png','seagull.jpg','stockpoint.jpg','sunset.jpg')
PRESETS=('compression','balanced','quality','high_quality')
VARIANTS=('release','development')


def validate(spec):
    if not isinstance(spec,dict) or set(spec)!={'images','presets','variants','repeats'}:
        raise ValueError('Expected fixed release-comparison fields')
    for key,allowed in (('images',INPUTS),('presets',PRESETS),('variants',VARIANTS)):
        value=spec[key]
        if not isinstance(value,list) or not 1<=len(value)<=len(allowed) or any(not isinstance(x,str) or x not in allowed for x in value) or len(set(value))!=len(value):
            raise ValueError('Invalid '+key)
    if set(spec['variants'])!=set(VARIANTS) or type(spec['repeats']) is not int or not 1<=spec['repeats']<=3:
        raise ValueError('Both sources and 1..3 repeats required')
    return spec


def request(process,job):
    process.stdin.write(json.dumps(job)+'\n'); process.stdin.flush()
    row=json.loads(process.stdout.readline())
    if 'error' in row:
        raise RuntimeError(row['error'])
    return row


def run(spec):
    manifest=json.loads((HERE/'runtime_manifest.json').read_text())
    for name,value in manifest['source_lf'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()==value,name
    for name,value in manifest['assets'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name
    processes={}; metadata={}; rows=[]
    def evaluate(source,image,preset,trial):
        return request(processes[source],dict(image=image,preset=preset,trial=trial))
    try:
        for source in VARIANTS:
            processes[source]=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--source',source],cwd=ROOT,
                env={**os.environ,**THREADS},stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            metadata[source]=request(processes[source],dict(op='metadata'))
        assert metadata['release']['cpu']==metadata['development']['cpu']
        warmups=[evaluate(s,'googlewiki.png',p,-1) for p in spec['presets'] for s in VARIANTS]
        for trial in range(spec['repeats']):
            for i,image in enumerate(spec['images']):
                for j,preset in enumerate(spec['presets']):
                    order=VARIANTS if (trial+i+j)%2==0 else VARIANTS[::-1]
                    for source in order:
                        rows.append(evaluate(source,image,preset,trial))
        for source in VARIANTS:
            for image in spec['images']:
                for preset in spec['presets']:
                    group=[r for r in rows if (r['source'],r['image'],r['preset'])==(source,image,preset)]
                    assert len({r['stream_sha256'] for r in group})==len({r['pixels_sha256'] for r in group})==1
        return dict(metadata=dict(spec=spec,workers=metadata,manifest=manifest,warmups=warmups,
            timing='Warmed full compress/decompress calls, separate persistent source processes, same pinned CPU, numerical threads one, sequential interleaving; loading/scoring/parsing/IPC excluded.'),rows=rows,deterministic=True)
    finally:
        for p in processes.values():
            if p.poll() is None:
                request(p,dict(op='quit')); p.wait(timeout=20)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec-json',required=True)
    args=parser.parse_args()
    print(json.dumps(run(validate(json.loads(args.spec_json)))))
