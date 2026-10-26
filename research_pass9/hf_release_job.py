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
VARIANTS=('release','dev_release_rng','dev_current_rng')
HOLDOUT=('15k_kills.png','ducks.jpg','old_man_standing.jpg','pigeon.JPG','pigeons.JPG')
EQUIVALENCE=('release','dev_release_rng')


def validate(spec):
    if not isinstance(spec,dict) or set(spec)!={'images','presets','variants','repeats'}:
        raise ValueError('Expected fixed release-comparison fields')
    for key,allowed in (('images',INPUTS+HOLDOUT),('presets',PRESETS),('variants',VARIANTS)):
        value=spec[key]
        if not isinstance(value,list) or not 1<=len(value)<=len(allowed) or any(not isinstance(x,str) or x not in allowed for x in value) or len(set(value))!=len(value):
            raise ValueError('Invalid '+key)
    if set(spec['variants']) not in (set(VARIANTS),set(EQUIVALENCE)) or type(spec['repeats']) is not int or not 1<=spec['repeats']<=3:
        raise ValueError('Release/RNG pair or three sources and 1..3 repeats required')
    if spec['repeats']>1 and (len(spec['images'])>8 or any(x in HOLDOUT for x in spec['images'])):
        raise ValueError('Repeated timing is limited to eight main images')
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
    variants=tuple(spec['variants'])
    def evaluate(source,image,preset,trial):
        return request(processes[source],dict(dataset='hpt_data_val' if image in HOLDOUT else 'hpt_data',image=image,preset=preset,trial=trial))
    try:
        for source in variants:
            processes[source]=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--source',source],cwd=ROOT,
                env={**os.environ,**THREADS},stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            metadata[source]=request(processes[source],dict(op='metadata'))
        assert len({m['cpu'] for m in metadata.values()})==1
        warmups=[evaluate(s,'googlewiki.png',p,-1) for p in spec['presets'] for s in variants]
        for trial in range(spec['repeats']):
            for i,image in enumerate(spec['images']):
                for j,preset in enumerate(spec['presets']):
                    offset=(trial+i+j)%len(variants)
                    order=variants[offset:]+variants[:offset]
                    for source in order:
                        rows.append(evaluate(source,image,preset,trial))
        for source in variants:
            for image in spec['images']:
                for preset in spec['presets']:
                    group=[r for r in rows if (r['source'],r['image'],r['preset'])==(source,image,preset)]
                    assert len({r['stream_sha256'] for r in group})==len({r['pixels_sha256'] for r in group})==1
        equivalent=[]
        for trial in range(spec['repeats']):
            for image in spec['images']:
                for preset in spec['presets']:
                    a,b=[next(r for r in rows if (r['source'],r['image'],r['preset'],r['trial'])==(s,image,preset,trial)) for s in EQUIVALENCE]
                    fields=('stream_sha256','pixels_sha256','trajectory_sha256','patches','mse','bpp','final_bytes')
                    differences=[k for k in fields if a[k]!=b[k]]
                    first=next((i for i,(x,y) in enumerate(zip(a['trajectory'],b['trajectory'])) if x!=y),None)
                    if differences:
                        raise AssertionError(dict(image=image,preset=preset,differences=differences,first_differing_patch=first))
                    equivalent.append(dict(image=image,preset=preset,trial=trial,equal_fields=list(fields)))
        return dict(equivalent=equivalent,metadata=dict(spec=spec,workers=metadata,manifest=manifest,warmups=warmups,
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
