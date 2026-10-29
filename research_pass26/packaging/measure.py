import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
import tempfile
import time
import statistics
portable=Path(json.loads((PASS/'packaging/measurements.json').read_text())['portable_directory'])
commands={'python':[sys.executable,str(PASS/'packaging/cli.py')], 'onedir':[str(portable/'directory/PBC/pbc.exe')], 'onefile':[str(portable/'single/pbc.exe')]}
cache=Path(tempfile.mkdtemp(prefix='PBC26 startup caches '))
rows=[]
def run(mode,args):
    env=dict(os.environ,PATH=r'C:\Windows\System32',NUMBA_CACHE_DIR=str(cache/mode))
    t=time.perf_counter();p=subprocess.run(commands[mode]+list(map(str,args)),cwd=portable,env=env,capture_output=True,text=True,encoding='utf-8',check=True);elapsed=time.perf_counter()-t
    return dict(latency_seconds=elapsed,result=json.loads(p.stdout) if p.stdout.startswith('{') else p.stdout.strip())
for repeat in range(3):
    for mode in list(commands)[repeat:]+list(commands)[:repeat]:
        rows.append(dict(mode=mode,operation='version',repeat=repeat,**run(mode,['--version'])))
for mode in commands:
    output=cache/(mode+'.pbc')
    args=['encode',PASS/'artifacts/cli input.png',output,'--preset','quality','--force']
    rows.append(dict(mode=mode,operation='first_use_fresh_cache',**run(mode,args)))
    rows.append(dict(mode=mode,operation='second_process_same_cache',**run(mode,args)))
for repeat,name in enumerate(('googlewiki.png','cat.jpg')):
    order=list(commands)[repeat:]+list(commands)[:repeat]
    for mode in order:
        rows.append(dict(mode=mode,operation='warmed_codec',image=name,**run(mode,['benchmark',ROOT/'hpt_data'/name,'--preset','quality','--repeats','3'])))
        print(mode,name,flush=True)
save('packaging/performance.json',dict(memory_polling=False,cache_root=str(cache),rows=rows))
