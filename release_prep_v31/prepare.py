import ctypes as ct
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'artifacts'
OUT.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=ROOT/'research_pass28/artifacts/final-source'
manifest=json.loads((ROOT/'research_pass28/SOURCE_TREE.json').read_text())
for name,digest in manifest['files'].items():
    assert sha(frozen/name)==digest,name
archive=ROOT/'research_pass26/artifacts/released.zip'
assert archive.stat().st_size==130098
assert sha(archive)=='edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede'
sourcezip=ROOT/'research_pass28/artifacts/final/PBC-3.2.0-release-source.zip'
assert sha(sourcezip)=='34d338686bc3bb62e65f044dacfff0cc155d16e1cdda63a7bfa0f0584021fd53'
sources=OUT/'sources'
sources.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(sources/'published')
for variant,original in [('original',ROOT/'research_pass1/release'),('candidate',frozen)]:
    shutil.copytree(original,sources/variant,ignore=shutil.ignore_patterns('__pycache__','*.egg-info','*.nbc','*.nbi'),dirs_exist_ok=True)
dll=frozen/'pbc3/mse.dll'
assert sha(dll)=='0def06d35fb5d4f97949ba2bc1ddb55651afeb2d007971a7c626c951cce928f3'
assets=json.loads((ROOT/'research/assets.json').read_text())['inputs']
for name,info in assets.items():
    assert sha(ROOT/name)==info['sha256'],name
for p in (sources/'original').glob('*.py'):
    data=subprocess.check_output(['git','show','2e59f1b06980aa467da0cde94fa33c3f9a0d4d51:'+p.name],cwd=ROOT)
    assert data.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n'),p.name
topology=[]
size=ct.c_ulong()
fn=ct.windll.kernel32.GetLogicalProcessorInformationEx
fn(0,None,ct.byref(size))
buf=ct.create_string_buffer(size.value)
assert fn(0,buf,ct.byref(size))
raw=buf.raw
offset=0
while offset<size.value:
    relation=int.from_bytes(raw[offset:offset+4],'little')
    length=int.from_bytes(raw[offset+4:offset+8],'little')
    if relation==0:
        count=int.from_bytes(raw[offset+30:offset+32],'little')
        masks=[int.from_bytes(raw[offset+32+16*i:offset+40+16*i],'little') for i in range(count)]
        topology.append(dict(flags=raw[offset+8],efficiency_class=raw[offset+9],masks=masks,
                             logical_cpus=[i for i in range(64) if masks[0]>>i&1]))
    offset+=length
selected=next(c for c in topology if 10 in c['logical_cpus'])
assert selected['flags']&1,selected
provenance=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    original_commit='2e59f1b06980aa467da0cde94fa33c3f9a0d4d51',
    source_tree=manifest['git_tree_sha1'],published_zip_sha256=sha(archive),candidate_zip_sha256=sha(sourcezip),
    native_dll_sha256=sha(dll),python=sys.version,executable=sys.executable,
    versions={n:md.version(n) for n in ('numpy','numba','Pillow','llvmlite','pyinstaller','psutil','setuptools')},
    cpu_topology=topology,selected_cpu=10,inputs=assets,
    sources={v:{str(p.relative_to(sources/v)).replace('\\','/'):sha(p) for p in (sources/v).rglob('*')
                if p.is_file() and '__pycache__' not in p.parts} for v in ('original','published','candidate')})
(HERE/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(dict(versions=provenance['versions'],selected_core=selected,inputs=len(assets)),indent=2))




if (HERE/'source/pbc3').exists():
    for name in ('patch_policy.npz','mse.dll'):
        shutil.copy2(frozen/'pbc3'/name,HERE/'source/pbc3'/name)

import base64
fixtures=json.loads((ROOT/'research_pass28/smoke/fixtures.json').read_text())
corpus=json.loads((ROOT/'research_pass27/decoder/corpus.json').read_text())
if (OUT/'probe-candidate/validation.json').exists():
    fixtures['expected']=json.loads((OUT/'probe-candidate/validation.json').read_text())['rows']
    fixtures['valid']=[dict(row,data=base64.b64encode(Path(row['file']).read_bytes()).decode()) for row in corpus['valid']]
    fixtures['invalid']=[base64.b64encode(Path(path).read_bytes()).decode() for path in corpus['invalid']]
    (OUT/'frozen-fixtures.json').write_text(json.dumps(fixtures))
