import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
import tempfile
import time
root=Path(json.loads((PASS/'packaging/validation.json').read_text())['root'])
commands={'combined':[str(root/'combined/pbc/pbc.exe')],'decoder':[str(root/'decoder/pbc-decode/pbc-decode.exe')],'decoder_onefile':[str(root/'single/pbc-decode.exe')],'published_python':[sys.executable,str(PASS/'packaging/reference_cli.py')],'pass26_combined':[str(ROOT/'research_pass26/artifacts/onedir/pbc/pbc.exe')]}
(root/'cwd-other').mkdir(exist_ok=True)
cache=Path(tempfile.mkdtemp(prefix='PBC27 performance ',dir=PASS/'artifacts'));rows=[]
def run(mode,args,group,cwd=None):
    env=dict(os.environ,PATH=r'C:\Windows\System32',NUMBA_CACHE_DIR=str(cache/group),TEMP=str(root/'tmp'),TMP=str(root/'tmp'))
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):env.pop(key,None)
    t=time.perf_counter();p=subprocess.run(commands[mode]+list(map(str,args)),cwd=cwd or root/'cwd',env=env,capture_output=True,text=True,encoding='utf-8',check=True);elapsed=time.perf_counter()-t
    result=json.loads(p.stdout) if p.stdout.startswith('{') else p.stdout.strip()
    row=dict(mode=mode,operation=group,latency_seconds=elapsed,result=result);rows.append(row);return result
for repeat in range(3):
    order=['combined','decoder','decoder_onefile','pass26_combined']
    for mode in order[repeat:]+order[:repeat]: run(mode,['--version'],'version')
for repeat in range(3):
    for mode in ('published_python','pass26_combined','combined','decoder','decoder_onefile'):
        run(mode,['inspect',PASS/'artifacts/corpus/python_quality.pbc'],'inspect')
input=root/'input \u00f6 \u65e5\u672c.png'
for mode,backend in (('pass26_combined',None),('combined','reference'),('combined','native')):
    group='encode_'+mode+'_'+str(backend)
    args=['encode',input,cache/(group+'.pbc'),'--preset','quality','--force']
    if backend:args+=['--mse-backend',backend]
    for phase in ('fresh_cache','reused_cache','reused_cache_other_cwd'):
        result=run(mode,args,group,cwd=root/'cwd-other' if phase=='reused_cache_other_cwd' else None);rows[-1]['phase']=phase
for name in ('cat.jpg','googlewiki.png'):
    for backend in ('reference','native'):
        run('combined',['benchmark',ROOT/'hpt_data'/name,'--preset','quality','--repeats','3','--mse-backend',backend],'warm_encode_'+name+'_'+backend)
for name in ('cat.jpg.quality','googlewiki.png.quality','python_rgba','eiffel.png.high_quality'):
    path=PASS/'artifacts/corpus'/(name+'.pbc')
    for mode in ('published_python','combined','decoder','decoder_onefile'):
        result=run(mode,['benchdecode',path,'--repeats','3'],'warm_decode_'+name)
        expected=next(r for r in json.loads((PASS/'decoder/corpus.json').read_text())['valid'] if r['name']==name)
        assert result['pixel_sha256']==expected['pixel_sha256']
for mode in commands:
    target=cache/(mode+'.png');args=['decode',PASS/'artifacts/corpus/python_quality.pbc',target,'--force'] if mode in ('combined','pass26_combined','published_python') else [PASS/'artifacts/corpus/python_quality.pbc',target,'--force']
    for _ in range(3):run(mode,args,'decode_command')
save('packaging/performance.json',dict(rows=rows,memory_polling=False,cache_root=str(cache),one_core=12,threads=1))
print('performance measurements complete')
