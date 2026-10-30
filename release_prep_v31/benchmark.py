import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'
MODES=('original','published','candidate')
PRESETS=('compression','balanced','quality','high_quality')
provenance=json.loads((HERE/'provenance.json').read_text())
identity=hashlib.sha256((HERE/'provenance.json').read_bytes()+(HERE/'worker.py').read_bytes()).hexdigest()
path=OUT/'measurements.jsonl'
rows=[]
if path.exists():
    payload=path.read_bytes()
    offset=0
    for line in payload.splitlines(keepends=True):
        try:
            row=json.loads(line)
        except json.JSONDecodeError:
            assert offset+len(line)==len(payload), 'Damaged checkpoint before last row'
            (OUT/'interrupted-checkpoint-tail.bin').write_bytes(line)
            with path.open('r+b') as f:f.truncate(offset)
            break
        rows.append(row)
        offset+=len(line)
assert all(r['campaign']==identity and r['error'] is None for r in rows)
done={(r['image'],r['preset'],r['repeat'],r['variant']) for r in rows}
assert len(done)==len(rows)
workers={}
logs=[]
def send(mode,req):
    p=workers[mode]
    p.stdin.write(json.dumps(req)+'\n');p.stdin.flush()
    line=p.stdout.readline()
    if not line: raise RuntimeError(mode+' worker exited; inspect its stderr log')
    r=json.loads(line)
    if r.get('error'): raise RuntimeError(json.dumps(r))
    return r
try:
    for mode in MODES:
        log=open(OUT/(mode+'-stderr.log'),'a',encoding='utf-8');logs.append(log)
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        p=subprocess.Popen([sys.executable,'-I',str(HERE/'worker.py'),mode],cwd=OUT/'sources'/mode,env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8')
        workers[mode]=p
        ready=json.loads(p.stdout.readline())
        assert ready['ready']
        (OUT/(mode+'-worker.json')).write_text(json.dumps(ready,indent=2))
    with path.open('a',encoding='utf-8') as f:
        for index,image in enumerate(provenance['inputs']):
            for preset in PRESETS:
                if all((image,preset,r,m) in done for r in range(3) for m in MODES): continue
                req=dict(image=image,preset=preset)
                for mode in MODES:
                    warm=send(mode,dict(req,warmup=True))
                    with (OUT/'warmups.jsonl').open('a') as w: w.write(json.dumps(warm)+'\n')
                for repeat in range(3):
                    order=(index+PRESETS.index(preset)+repeat)%3
                    for mode in MODES[order:]+MODES[:order]:
                        key=(image,preset,repeat,mode)
                        if key in done: continue
                        row=send(mode,dict(req,repeat=repeat,save=repeat==0))
                        row['campaign']=identity;row['measured_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
                        f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
                        done.add(key)
                print(f'{len(done)}/900 {image} {preset}',flush=True)
finally:
    for p in workers.values():
        p.stdin.close()
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired: p.kill()
    for f in logs:f.close()
assert len(done)==900
print('COMPLETE: 900 complete encode/decode pairs',flush=True)
