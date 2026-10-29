import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import shutil
import subprocess
import tempfile
import time

out=PASS/'artifacts/packaging_tests';out.mkdir(exist_ok=True)
portable=Path(tempfile.mkdtemp(prefix='PBC26 portability '))
shutil.copytree(PASS/'artifacts/onedir/pbc',portable/'directory/PBC')
(portable/'single').mkdir();shutil.copy2(PASS/'artifacts/onefile/pbc.exe',portable/'single/pbc.exe')
commands={'python':[sys.executable,str(PASS/'packaging/cli.py')],
          'onedir':[str(portable/'directory/PBC/pbc.exe')],
          'onefile':[str(portable/'single/pbc.exe')]}
input_image=load('eiffel.png').resize((96,95))
input_image.save(out/'spaces \u00e7\u0131\u011f RGB.png')
rgba=Image.fromarray(np.random.default_rng(26).integers(0,256,(91,127,4),dtype=np.uint8));rgba.save(out/'alpha \u00f6.png')
rows=[];checks=[]

def run(mode,args,expect=0):
    env=dict(os.environ,PATH=r'C:\Windows\System32',NUMBA_CACHE_DIR=str(portable/('cache_'+mode)))
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    t=time.perf_counter()
    p=subprocess.run(commands[mode]+list(map(str,args)),cwd=portable,env=env,capture_output=True,text=True,encoding='utf-8')
    stdout,stderr=p.stdout,p.stderr;elapsed=time.perf_counter()-t
    assert p.returncode==expect,(mode,args,p.returncode,stdout,stderr)
    if expect: assert 'Traceback' not in stderr
    return dict(latency_seconds=elapsed,stdout=stdout.strip(),stderr=stderr.strip(),exit_code=p.returncode)

for mode in commands:
    versions=[run(mode,['--version']) for _ in range(3)]
    rows.append(dict(mode=mode,operation='version',runs=versions))
    for preset in ('compression','balanced','quality','high_quality'):
        path=out/'spaces \u00e7\u0131\u011f RGB.png';target=out/(mode+'_'+preset+'.pbc')
        result=run(mode,['encode',path,target,'--preset',preset,'--force'])
        reference=PBC3.compress(input_image,config=getattr(PBC3Config,preset)())
        assert target.read_bytes()==reference.data
        rows.append(dict(mode=mode,operation='encode',fixture='small_rgb',preset=preset,**result))
        decoded=out/(mode+'_'+preset+'.png')
        result=run(mode,['decode',target,decoded,'--force'])
        assert Image.open(decoded).tobytes()==PBC3.decompress(reference.data).image.tobytes()
        checks.append(mode+'/'+preset+'/RGB')
    target=out/(mode+'_rgba.pbc')
    result=run(mode,['encode',out/'alpha \u00f6.png',target,'--preset','quality','--force'])
    reference=PBC3.compress(rgba,config=PBC3Config.quality())
    assert target.read_bytes()==reference.data
    decoded=out/(mode+'_rgba.png');run(mode,['decode',target,decoded,'--force'])
    assert Image.open(decoded).tobytes()==PBC3.decompress(reference.data).image.tobytes()
    rows.append(dict(mode=mode,operation='encode',fixture='small_rgba',**result))
    inspect=run(mode,['inspect',target]);assert json.loads(inspect['stdout'])['alpha']
    previous=target.read_bytes();run(mode,['encode',out/'alpha \u00f6.png',target],1);assert target.read_bytes()==previous
    run(mode,['encode',out/'missing.png',out/'missing.pbc'],1)
    for i,data in enumerate((b'',b'PBC3',b'PBC3\x01\x00',b'PBC3\x00\x07',previous[:15],previous+b'junk',b'PBC3\x00\x02x')):
        invalid=out/f'bad{i}.pbc';invalid.write_bytes(data)
        run(mode,['decode',invalid,out/'invalid.png','--force'],1)
    checks.append(mode+'/RGBA/inspect/no_overwrite/missing/7_invalid')
    for name in ('googlewiki.png','cat.jpg'):
        result=run(mode,['benchmark',ROOT/'hpt_data'/name,'--preset','quality','--repeats','3'])
        data=json.loads(result['stdout']);rows.append(dict(mode=mode,operation='benchmark',image=name,measurement=result,**data))
    print(mode,'passed',flush=True)
distributions={}
for mode in ('onedir','onefile'):
    folder=PASS/'artifacts'/mode
    files=[p for p in folder.rglob('*') if p.is_file()]
    exe=folder/'pbc/pbc.exe' if mode=='onedir' else folder/'pbc.exe'
    distributions[mode]=dict(executable=str(exe),exe_sha256=sha(exe.read_bytes()),exe_bytes=exe.stat().st_size,total_bytes=sum(p.stat().st_size for p in files),files=len(files))
save('packaging/measurements.json',dict(portable_directory=str(portable),path=r'C:\Windows\System32',checks=checks,rows=rows,distributions=distributions))
