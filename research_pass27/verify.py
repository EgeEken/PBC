import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import ast
import dataclasses
from datetime import datetime,timezone
import importlib.metadata
import platform
import subprocess
import zipfile

for p in PASS.rglob('*.py'):
    if 'artifacts' not in p.parts: ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
assert not subprocess.check_output(['git','diff','87fb89e184abce8065bc001dec0ea121cbe3cc72','--name-only','--','.',':!research_pass27'],text=True).strip()
previous=json.loads((ROOT/'research_pass26/provenance.json').read_text())
for key,item in previous['local_binaries'].items():
    p=Path(item.get('executable',item.get('path')))
    expected=item.get('exe_sha256',item.get('sha256'))
    assert sha(p.read_bytes())==expected,('Pass26 artifact changed',key)
original=json.loads((ROOT/'research_pass26/baseline.json').read_text())
stage=Path(r'C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC\PBC3_1_Release_Staging')
assert {str(p.relative_to(stage)):sha(p.read_bytes()) for p in stage.rglob('*') if p.is_file() and p.suffix!='.pyc'}==original['staging_files']
assert sha((PASS/'artifacts/published31.zip').read_bytes())=='edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede'
for row in original['files']:assert sha((PASS/'artifacts/published'/row['file']).read_bytes())==row['published_sha256']
public_ops=ast.parse((PUBLISHED/'pbc3_ops.py').read_text());pure_ops=ast.parse((CANDIDATE/'pbc3_decode_ops.py').read_text())
public={n.name:ast.dump(n,include_attributes=False) for n in public_ops.body if isinstance(n,ast.FunctionDef)}
for n in pure_ops.body:
    if isinstance(n,ast.FunctionDef):assert ast.dump(n,include_attributes=False)==public[n.name],n.name
artifacts={}
for kind,name in (('combined','pbc'),('decoder','pbc-decode')):
    folder=PASS/'artifacts'/kind/'onedir'/name;files=[p for p in folder.rglob('*') if p.is_file()]
    assert not any(p.suffix in ('.nbc','.nbi') for p in files)
    for doc in ('README.md','INSTALL.md','LICENSE'):assert (folder/doc).is_file()
    if kind=='decoder':assert not any('numba' in p.name.lower() or 'llvmlite' in p.name.lower() or p.suffix=='.npz' for p in files)
    exe=folder/(name+'.exe');archive=PASS/'artifacts'/('PBC32rc1-'+kind+'-Windows-x64.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert sha(z.read(name+'/'+name+'.exe'))==sha(exe.read_bytes())
    artifacts[kind]=dict(executable=str(exe),exe_bytes=exe.stat().st_size,exe_sha256=sha(exe.read_bytes()),files=len(files),extracted_bytes=sum(p.stat().st_size for p in files),zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=sha(archive.read_bytes()))
exe=PASS/'artifacts/decoder/onefile/pbc-decode.exe';artifacts['decoder_onefile']=dict(executable=str(exe),bytes=exe.stat().st_size,sha256=sha(exe.read_bytes()))
dll=CANDIDATE/'pbc3/mse.dll';artifacts['native_dll']=dict(path=str(dll),bytes=dll.stat().st_size,sha256=sha(dll.read_bytes()))
for p in (PASS/'artifacts/packages').iterdir():
    if p.is_file():artifacts[p.suffix]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
wheel=next((PASS/'artifacts/packages').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
    assert sha(z.read('pbc3/patch_policy.npz'))=='ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f'
    assert not any(n.endswith(('.dll','.nbc','.nbi')) for n in z.namelist())
save('artifacts.json',artifacts)
versions={n:importlib.metadata.version(n) for n in ('numpy','numba','llvmlite','Pillow','psutil','pyinstaller','pyinstaller-hooks-contrib','build','setuptools','wheel')}
manifest=dict(verified_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),cpu='Intel Core i9-12900H',affinity=psutil.Process().cpu_affinity(),threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')},versions=versions,
    original_head='87fb89e184abce8065bc001dec0ea121cbe3cc72',published_asset_sha256=original['zip_sha256'],released_sources={r['file']:r['published_sha256'] for r in original['files']},
    input_hashes={n:sha((ROOT/'hpt_data'/n).read_bytes()) for n in ('cat.jpg','flower.jpg','seagull.jpg','googlewiki.png','monet.jpg','eiffel.png')},
    research_source_sha256={str(p.relative_to(PASS)):sha(p.read_bytes()) for p in PASS.rglob('*') if p.is_file() and 'artifacts' not in p.parts and '__pycache__' not in p.parts and p.suffix in ('.py','.c','.txt','.toml','.in')},
    generated_candidate_sha256={str(p.relative_to(CANDIDATE)):sha(p.read_bytes()) for p in CANDIDATE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'build' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts) and p.suffix in ('.py','.npz','.dll','.toml')},
    build_environment_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),
    decoder_environment_freeze=subprocess.check_output(['uv','pip','freeze','--python',str(PASS/'artifacts/decoder-env/Scripts/python.exe')],text=True).splitlines(),
    protected_refs=subprocess.check_output(['git','ls-remote','github','refs/heads/main','refs/heads/rust','refs/tags/v3.1'],text=True),hf_ref=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True),pass26_preserved=True,release_staging_preserved=True,artifacts=artifacts)
save('provenance.json',manifest)
print('verified frozen baseline, preserved Pass26/staging, shared operation bodies, package contents and artifact hashes')
