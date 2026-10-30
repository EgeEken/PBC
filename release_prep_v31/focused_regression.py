import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'
frozen='--frozen' in sys.argv
target='frozen' if frozen else 'library'
if not frozen:
    shutil.copytree(HERE/'source',OUT/'sources/library',ignore=shutil.ignore_patterns('__pycache__','*.egg-info','build','dist'),dirs_exist_ok=True)
workers={}
logs=[]
def send(mode,req):
    p=workers[mode];p.stdin.write(json.dumps(req)+'\n');p.stdin.flush()
    r=json.loads(p.stdout.readline())
    assert not r.get('error'),r
    return r
rows=[]
try:
    for mode in ('candidate',target):
        log=open(OUT/(mode+'-focus.log'),'w');logs.append(log)
        env=os.environ.copy()
        if mode=='frozen':
            env['PATH']=r'C:\Windows\System32'
            env['NUMBA_CACHE_DIR']=str(OUT/'frozen-focus-cache')
            env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
            command=[str(OUT/'probe-dist/PBC-probe/PBC-probe.exe'),'--worker',
                     str(HERE.parent/'hpt_data/cat.jpg'),str(HERE.parent/'hpt_data/flower.jpg')]
        else:
            command=[sys.executable,'-I',str(HERE/'worker.py'),mode]
        p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        workers[mode]=p
        assert json.loads(p.stdout.readline())['ready']
    for image in ('hpt_data/cat.jpg','hpt_data/flower.jpg'):
        req=dict(image=image,preset='quality')
        for mode in workers:send(mode,req)
        for repeat in range(3):
            pair={}
            for mode in (('candidate',target) if repeat%2==0 else (target,'candidate')):
                pair[mode]=send(mode,dict(req,repeat=repeat))
            for key in ('stream_sha256','pixel_sha256','bytes','bpp','internal_mse'):
                assert pair['candidate'][key]==pair[target][key]
            rows.append(dict(image=image,repeat=repeat,**pair))
finally:
    for p in workers.values():p.stdin.close();p.wait(timeout=30)
    for f in logs:f.close()
summary={}
for image in ('hpt_data/cat.jpg','hpt_data/flower.jpg'):
    subset=[r for r in rows if r['image']==image]
    summary[image]={f:{m:st.median(r[m][f] for r in subset) for m in workers} for f in ('encode_seconds','decode_seconds')}
(OUT/('focused-frozen.json' if frozen else 'focused-regression.json')).write_text(json.dumps(dict(rows=rows,medians=summary,exact=True),indent=2)+'\n')
print(json.dumps(summary,indent=2))
