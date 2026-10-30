import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import psutil
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];PASS=ROOT/'research_pass28';dist=PASS/'artifacts/final'
psutil.Process().cpu_affinity([12])
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
work=Path(tempfile.mkdtemp(prefix='PBC28 memory ',dir=PASS/'artifacts'))
env=dict(os.environ,PATH=r'C:\Windows\System32',NUMBA_CACHE_DIR=str(work/'cache'))
for k in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):env.pop(k,None)
commands={'published':[sys.executable,str(PASS/'reference_cli.py')],'reference':[str(dist/'combined/onedir/pbc/pbc.exe')],'native':[str(dist/'combined/onedir/pbc/pbc.exe')],'decoder':[str(dist/'decoder/onedir/pbc-decode/pbc-decode.exe')]}
small=work/'warm.png';Image.new('RGB',(39,37),(80,130,220)).save(small)
for mode in ('published','reference','native'):
    args=['encode',str(small),str(work/(mode+'warm.pbc')),'--preset','quality']
    if mode!='published':args+=['--mse-backend',mode]
    subprocess.run(commands[mode]+args,env=env,cwd=work,check=True,stdout=subprocess.PIPE)
rows=[]
for image in ('cat.jpg','flower.jpg'):
    corpus=ROOT/'research_pass27/artifacts/corpus'/(image+'.quality.pbc')
    for repeat in range(2):
        modes=('published','reference','native','decoder')
        if repeat:modes=tuple(reversed(modes))
        streams=[]
        for mode in modes:
            target=work/(mode+image+str(repeat)+('.png' if mode=='decoder' else '.pbc'))
            args=[str(corpus),str(target)] if mode=='decoder' else ['encode',str(ROOT/'hpt_data'/image),str(target),'--preset','quality']
            if mode in ('reference','native'):args+=['--mse-backend',mode]
            process=subprocess.Popen(commands[mode]+args,env=env,cwd=work,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
            peak=0;rss=0;samples=0;family_count=0;observed=psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    family=[observed]+observed.children(recursive=True);information=[]
                    for child in family:
                        try:information.append(child.memory_info())
                        except psutil.Error:pass
                    family_count=max(family_count,len(information));peak=max(peak,sum(m.peak_wset for m in information));rss=max(rss,sum(m.rss for m in information));samples+=1
                except psutil.Error:pass
                time.sleep(.01)
            stdout,stderr=process.communicate();assert process.returncode==0,(mode,stderr)
            if mode!='decoder':
                digest=hashlib.sha256(target.read_bytes()).hexdigest();streams.append(digest);assert target.read_bytes()==corpus.read_bytes()
            else:
                expected=next(r for r in json.loads((ROOT/'research_pass27/decoder/corpus.json').read_text())['valid'] if r['name']==image+'.quality')
                assert hashlib.sha256(Image.open(target).tobytes()).hexdigest()==expected['pixel_sha256']
            rows.append(dict(image=image,mode=mode,repeat=repeat,peak_wset_observed=peak,max_sampled_process_tree_rss=rss,process_count_max=family_count,samples=samples,codec_result=json.loads(stdout)))
        assert len(set(streams))==1
(PASS/'artifacts/cli-memory.json').write_text(json.dumps(dict(rows=rows,sample_interval_seconds=.01,timings_discarded=True,scope='complete process tree including Windows venv launcher, imports and image I/O; primary observation is max concurrent sampled RSS'),indent=2)+'\n')
print('CLI_MEMORY_PASS',len(rows),'fresh complete commands')
