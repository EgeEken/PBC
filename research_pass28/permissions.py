import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research_pass27'))
from common import *
PASS=ROOT/'research_pass28'
def save(name,data): (PASS/'artifacts/permission-validation.json').write_text(json.dumps(data,indent=2)+'\n',newline='\n')
import csv
import ctypes
import shutil
import subprocess
import tempfile

root=Path(tempfile.mkdtemp(prefix='portability \u00c7\u0131\u011f \u65e5\u672c ',dir=PASS/'artifacts'))
for kind,name in (('combined','pbc'),('decoder','pbc-decode')):
    archive=PASS/'artifacts/final'/('PBC-3.2.0-'+kind+'-Windows-x64.zip')
    shutil.unpack_archive(archive,root/kind)
(root/'single').mkdir(exist_ok=True)
shutil.copy2(PASS/'artifacts/final/decoder/onefile/pbc-decode.exe',root/'single/pbc-decode.exe')
for name in ('cwd','outputs','cache','tmp'): (root/name).mkdir(exist_ok=True)
commands={'combined':[str(root/'combined/pbc/pbc.exe')],'decoder':[str(root/'decoder/pbc-decode/pbc-decode.exe')],'decoder_onefile':[str(root/'single/pbc-decode.exe')]}
sid=next(csv.reader(subprocess.check_output(['whoami','/user','/fo','csv','/nh'],text=True).splitlines()))[1]
manifest=json.loads((ROOT/'research_pass27/decoder/corpus.json').read_text());expected={r['name']:r for r in manifest['valid']}
rows=[]
def run(kind,args,code=0,default_cache=False):
    env=dict(os.environ,PATH=r'C:\Windows\System32',NUMBA_CACHE_DIR=str(root/'cache'),TEMP=str(root/'tmp'),TMP=str(root/'tmp'))
    if default_cache:
        env.pop('NUMBA_CACHE_DIR',None);env['LOCALAPPDATA']=str(root/'user-local')
    for k in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):env.pop(k,None)
    p=subprocess.run(commands[kind]+list(map(str,args)),cwd=root/'cwd',env=env,capture_output=True,text=True,encoding='utf-8')
    assert p.returncode==code,(kind,args,p.returncode,p.stdout,p.stderr)
    assert 'Traceback' not in p.stderr
    if kind=='decoder_onefile': assert not list((root/'tmp').glob('_MEI*')),'temporary extraction leaked'
    return json.loads(p.stdout) if p.stdout.startswith('{') else p.stdout.strip()
acl=[]
try:
    for kind in ('combined','decoder'):
        app=Path(commands[kind][0]).parent
        assert app.resolve().is_relative_to((PASS/'artifacts').resolve())
        subprocess.run(['icacls',str(app),'/inheritance:r','/grant:r','*'+sid+':(OI)(CI)(RX,WDAC)','/q'],check=True,stdout=subprocess.PIPE)
        acl.append(app)
        try: (app/'write_probe').write_text('must fail')
        except PermissionError: pass
        else: raise AssertionError('application directory is writable')
        protected=app/'_internal/base_library.zip'
        try:
            with protected.open('r+b'): pass
        except PermissionError: pass
        else: raise AssertionError('existing application file is writable')
    image=load('eiffel.png').resize((96,95));input=root/'input \u00f6 \u65e5\u672c.png';image.save(input)
    for preset in ('compression','balanced','quality','high_quality'):
        target=root/'outputs'/(preset+'.pbc')
        result=run('combined',['encode',input,target,'--preset',preset])
        assert sha(target.read_bytes())==expected['python_'+preset]['stream_sha256']
        assert run('combined',['inspect',target])['patches']>0
        rows.append(dict(test='all_presets_readonly_native_default',**result))
        before=target.read_bytes();run('combined',['encode',input,target],1);assert before==target.read_bytes()
    run('combined',['encode',input,root/'outputs/default-cache.pbc','--preset','quality'],default_cache=True)
    assert list((root/'user-local/PBC/3.2.0/numba').rglob('*.nbc'))
    source=ROOT/'research_pass26/artifacts/packaging_tests/alpha \u00f6.png'
    target=root/'outputs/alpha.pbc';run('combined',['encode',source,target,'--preset','quality'])
    assert sha(target.read_bytes())==expected['python_rgba']['stream_sha256']
    for row in manifest['valid']:
        path=root/'outputs'/(row['name']+'.png')
        run('decoder',[row['file'],path])
        assert sha(Image.open(path).tobytes())==row['pixel_sha256']
    for kind in commands:
        for key in ('python_rgba','stored','warmup','historical30'):
            row=expected[key];path=root/'outputs'/(kind+'_'+key+'.png')
            args=['decode',row['file'],path] if kind=='combined' else [row['file'],path]
            run(kind,args);assert sha(Image.open(path).tobytes())==row['pixel_sha256']
            assert run(kind,['inspect',row['file']])['bytes']==row['bytes']
        for file in manifest['invalid'][:13]:
            args=['decode',file,root/'outputs/invalid.png','--force'] if kind=='combined' else [file,root/'outputs/invalid.png','--force']
            run(kind,args,1)
        run(kind,['inspect',root/'missing.pbc'],1)
        assert run(kind,['--version'])=='pbc 3.2.0; stream v0'
    assert list((root/'cache').rglob('*.nbc')),'persistent user cache was not populated'
    assert not list((root/'combined/pbc').rglob('*.nbc'))
    decoder_folder=root/'decoder/pbc-decode'
    assert not any('numba' in p.name.lower() or 'llvmlite' in p.name.lower() or p.suffix=='.npz' for p in decoder_folder.rglob('*') if 'licenses' not in p.parts)
finally:
    for app in acl: subprocess.run(['icacls',str(app),'/grant:r','*'+sid+':(OI)(CI)(F)','/inheritance:e','/t','/q'],check=True,stdout=subprocess.PIPE)
save('packaging/validation.json',dict(root=str(root),non_administrator=not bool(ctypes.windll.shell32.IsUserAnAdmin()),actual_acl_write_denied=True,no_python_or_compiler_on_path=True,empty_python_environment_variables=True,unicode_paths=True,all_presets_exact=True,rgba_exact=True,decoder_corpus_files=len(manifest['valid']),stored_warmup_historical_all_binaries=True,invalid_cases_per_binary=13,onefile_temporary_cleanup=True,user_cache_populated=True,decoder_no_numba_llvm_or_model=True,rows=rows))
print('distribution permission/identity checks passed')
