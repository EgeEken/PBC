import base64,json,os,statistics,subprocess,sys,tempfile,time
from pathlib import Path
import psutil
psutil.Process().cpu_affinity([12])
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[n]='1'
ROOT=Path(__file__).resolve().parents[1];PASS=ROOT/'research_pass28';dist=PASS/'artifacts/dist'
commands={'published':[sys.executable,str(PASS/'reference_cli.py')],'combined':[str(dist/'combined/onedir/pbc/pbc.exe')],'decoder':[str(dist/'decoder/onedir/pbc-decode/pbc-decode.exe')],'onefile':[str(dist/'decoder/onefile/pbc-decode.exe')]}
work=Path(tempfile.mkdtemp(prefix='PBC28 latency ',dir=PASS/'artifacts'));(work/'cwd').mkdir();(work/'other-cwd').mkdir();(work/'temp').mkdir()
f=json.loads((PASS/'smoke/fixtures.json').read_text());rows=[]
for r in f['inputs']:(work/(r['name']+'.png')).write_bytes(base64.b64decode(r['png']))
for name in ('small','rgba'):
    row=next(r for r in f['inputs'] if r['name']==('RGB' if name=='small' else 'RGBA'))
    # Use saved streams only for decode timing; encoder inputs are fixed fixtures.
    stream=next(r for r in f['streams'] if r['name']==('python_quality' if name=='small' else 'python_rgba'))
    (work/(name+'.pbc')).write_bytes(base64.b64decode(stream['data']))
def run(mode,args,group,phase=None,other=False):
    env=dict(os.environ,PATH=r'C:\Windows\System32',NUMBA_CACHE_DIR=str(work/('cache-'+group)),TEMP=str(work/'temp'),TMP=str(work/'temp'))
    for n in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):env.pop(n,None)
    t=time.perf_counter();p=subprocess.run(commands[mode]+list(map(str,args)),cwd=work/('other-cwd' if other else 'cwd'),env=env,capture_output=True,text=True,encoding='utf-8',check=True);latency=time.perf_counter()-t
    result=json.loads(p.stdout) if p.stdout.startswith('{') else p.stdout.strip()
    rows.append(dict(mode=mode,operation=group,phase=phase,command_seconds=latency,result=result));return result
for repeat in range(3):
    modes=('combined','decoder','onefile')
    for mode in modes[repeat:]+modes[:repeat]:run(mode,['--version'],'startup')
for mode,backend in (('published',None),('combined','reference'),('combined','native')):
    group='first_'+mode+'_'+str(backend);args=['encode',work/'RGB.png',work/(group+'.pbc'),'--preset','quality','--force']
    if backend:args+=['--mse-backend',backend]
    for phase in ('empty_cache','reused_cache','other_cwd'):
        run(mode,args,group,phase,phase=='other_cwd')
files={'cat':ROOT/'research_pass27/artifacts/corpus/cat.jpg.quality.pbc','flower':ROOT/'research_pass27/artifacts/corpus/flower.jpg.quality.pbc','googlewiki':ROOT/'research_pass27/artifacts/corpus/googlewiki.png.quality.pbc','eiffel':ROOT/'research_pass27/artifacts/corpus/eiffel.png.high_quality.pbc','small':work/'small.pbc','rgba':work/'rgba.pbc'}
for index,(name,path) in enumerate(files.items()):
    modes=('published','combined','decoder','onefile')
    hashes=[]
    for mode in modes[index%4:]+modes[:index%4]:hashes.append(run(mode,['benchdecode',path,'--repeats','3'],'warm_decode_'+name)['pixel_sha256'])
    assert len(set(hashes))==1
for repeat in range(3):
    for name in ('small','rgba'):
        modes=('published','combined','decoder','onefile')
        for mode in modes[repeat:]+modes[:repeat]:
            args=[files[name],work/(mode+name+'.png'),'--force']
            if mode in ('published','combined'):args=['decode']+args
            run(mode,args,'full_decode_command_'+name)
(PASS/'artifacts/cli-performance.json').write_text(json.dumps(dict(rows=rows,one_core=12,threads=1,memory_polling=False,work=str(work)),indent=2)+'\n')
print('CLI measurements complete',len(rows))
