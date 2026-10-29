import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import ast
import dataclasses
import importlib.metadata
import platform
import subprocess
import zipfile
import winreg
from datetime import datetime, timezone

for p in PASS.rglob('*.py'):
    if 'artifacts' not in p.parts: ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
before=json.loads((PASS/'baseline.json').read_text())
stage=Path(r'C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC\PBC3_1_Release_Staging')
after={str(p.relative_to(stage)):sha(p.read_bytes()) for p in stage.rglob('*') if p.is_file() and p.suffix!='.pyc'}
assert before['staging_files']==after,'release staging changed'
assert subprocess.check_output(['git','rev-parse','v3.1'],text=True).strip()==before['release_tag']
changes=subprocess.check_output(['git','diff',before['research_before'],'--name-only','--','.',':!research_pass26'],text=True).strip()
assert not changes,changes
pack=json.loads((PASS/'packaging/measurements.json').read_text())
for form,record in pack['distributions'].items():
    exe=Path(record['executable']);assert sha(exe.read_bytes())==record['exe_sha256'] and exe.stat().st_size==record['exe_bytes']
    files=[p for p in (PASS/'artifacts'/form).rglob('*') if p.is_file()]
    assert sum(p.stat().st_size for p in files)==record['total_bytes'] and len(files)==record['files']
archive=PASS/'artifacts/PBC32Research-Windows-x64.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert sha(z.read('pbc/pbc.exe'))==pack['distributions']['onedir']['exe_sha256']
architecture=json.loads((PASS/'architecture/results.json').read_text())
for row in architecture:
    name,codec,q=row['image'],row['codec'],row['setting']
    file=(name+'.'+q+'.pbc') if codec=='PBC31' else (name+f'.d{q}.jxl') if codec=='JXL' else (name+'.'+codec+'.'+str(q)+'.'+codec.removeprefix('working_').lower())
    data=(PASS/'artifacts/architecture'/file).read_bytes()
    assert len(data)==row['bytes'] and sha(data)==row['sha256'],file
video=json.loads((PASS/'video/results.json').read_text())
for row in video:
    file=row['clip']+'.mp4' if row['mode']=='h264' else row['clip']+'.'+row['mode']+'.p26v'
    data=(PASS/'artifacts/video'/file).read_bytes()
    assert len(data)==row['bytes'] and sha(data)==row['sha256'],file
    assert sha((PASS/'artifacts/video'/(row['clip']+'.rgb')).read_bytes())==row['source_sha256']
key=winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0];winreg.CloseKey(key)
remote=subprocess.check_output(['git','ls-remote','github','refs/heads/main','refs/heads/rust','refs/tags/v3.1'],text=True)
hf=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True)
versions={n:importlib.metadata.version(n) for n in ('numpy','numba','llvmlite','Pillow','psutil','pyinstaller','pyinstaller-hooks-contrib')}
provenance=dict(verified_utc=datetime.now(timezone.utc).isoformat(),baseline_snapshot_utc=datetime.fromtimestamp((PASS/'baseline.json').stat().st_mtime,timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),cpu=cpu,affinity=psutil.Process().cpu_affinity(),numba_threads=numba.get_num_threads(),thread_environment={n:os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS')},versions=versions,
    gcc=subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],gcc_target=subprocess.check_output(['gcc','-dumpmachine'],text=True).strip(),ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
    research_before=before['research_before'],released_tag=before['release_tag'],released_zip_sha256=before['zip_sha256'],released_sources={r['file']:r['published_sha256'] for r in before['files']},model_path=str(SOURCE/'patch_policy.npz'),presets={n:dataclasses.asdict(getattr(PBC3Config,n)()) for n in ('compression','balanced','quality','high_quality')},
    inputs={n:sha((ROOT/'hpt_data'/n).read_bytes()) for n in ('cat.jpg','flower.jpg','googlewiki.png','monet.jpg','mario64.png','eiffel.png')},
    research_script_sha256={str(p.relative_to(PASS)):sha(p.read_bytes()) for p in PASS.rglob('*') if p.is_file() and 'artifacts' not in p.parts and '__pycache__' not in p.parts and p.suffix in ('.py','.c','.txt')},
    local_binaries={**pack['distributions'],'directory_zip':dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),'native_dll':dict(path=str(PASS/'artifacts/mse.dll'),bytes=(PASS/'artifacts/mse.dll').stat().st_size,sha256=sha((PASS/'artifacts/mse.dll').read_bytes()))},
    protected_ref_observations=remote,hf_ref=hf,main_entry=before['main_before'],main_external_change=dict(observed='d938d003f68063e8df607957d8929c5ba8e74344',parent=before['main_before'],message='Update README.md',writes_by_this_pass=False),staging_unchanged=True,architecture_files_verified=len(architecture),video_files_verified=len(video))
save('provenance.json',provenance)
print('verified:',len(architecture),'image files,',len(video),'videos, final binaries, source isolation, syntax and staging')
