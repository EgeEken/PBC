import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
PASS=ROOT/'research_pass28'
subprocess.run([sys.executable,str(ROOT/'research_pass27/packaging/prepare.py')],check=True)
source=PASS/'artifacts/release-source';source.mkdir(parents=True,exist_ok=True)
for p in (ROOT/'research_pass27/artifacts/candidate').iterdir():
    if p.is_file() and p.suffix in ('.py','.toml','.md','.in') or p.name=='LICENSE':
        shutil.copy2(p,source/p.name)
(source/'pbc3').mkdir(exist_ok=True)
for name in ('__init__.py','mse.c','patch_policy.npz'):
    shutil.copy2(ROOT/'research_pass27/artifacts/candidate/pbc3'/name,source/'pbc3'/name)
shutil.copytree(PASS/'licenses',source/'licenses',dirs_exist_ok=True)
shutil.copy2(PASS/'THIRD_PARTY_NOTICES.md',source/'THIRD_PARTY_NOTICES.md')
installed=[]
for name in ('numpy','Pillow','numba','llvmlite','pyinstaller'):
    dist=metadata.distribution(name)
    for entry in dist.files:
        if any(t in str(entry).lower() for t in ('license','copying','notice')):
            p=Path(dist.locate_file(entry))
            if not p.is_file():continue
            target=source/'licenses'/name/str(entry)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
            installed.append(dict(package=name,version=dist.version,path=str(target.relative_to(source)).replace('\\','/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
python_license=Path(sys.base_prefix)/'LICENSE.txt'
shutil.copy2(python_license,source/'licenses/PYTHON_LICENSE.txt')
installed.append(dict(package='CPython',version=sys.version.split()[0],path='licenses/PYTHON_LICENSE.txt',sha256=hashlib.sha256(python_license.read_bytes()).hexdigest()))
(source/'licenses/installed-sources.json').write_text(json.dumps(installed,indent=2)+'\n',newline='\n')
(source/'tools/hooks').mkdir(parents=True,exist_ok=True)
shutil.copy2(ROOT/'research_pass26/packaging/hooks/hook-numba.py',source/'tools/hooks/hook-numba.py')
for name in ('requirements-build.txt','requirements-decoder-build.txt'):
    shutil.copy2(ROOT/'research_pass27'/name,source/'tools'/name)
shutil.copy2(PASS/'build_windows.py',source/'tools/build_windows.py')
shutil.copy2(PASS/'BUILD.md',source/'BUILD.md')
license_files=['LICENSE','THIRD_PARTY_NOTICES.md']+[str(p.relative_to(source)).replace('\\','/') for p in (source/'licenses').rglob('*') if p.is_file()]
p=source/'pyproject.toml';p.write_text(p.read_text().replace('license-files = ["LICENSE"]','license-files = '+json.dumps(license_files)).replace('["patch_policy.npz", "mse.c"]','["patch_policy.npz", "mse.c", "IMPLEMENTATION.json"]'),newline='\n')
p=source/'MANIFEST.in';p.write_text(p.read_text()+'\ninclude THIRD_PARTY_NOTICES.md BUILD.md IMPLEMENTATION.json\nrecursive-include licenses *\nrecursive-include tools *.py *.txt\n',newline='\n')
files={str(p.relative_to(source)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.suffix in ('.py','.npz','.c') and 'tools' not in p.parts}
(source/'IMPLEMENTATION.json').write_text(json.dumps(dict(version='3.2.0rc1',baseline='published 3.1 asset edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede',codec_source_commit='1b207e1c67fee4c911c57c9f7d0d4c0845e82952',files=files),indent=2)+'\n',newline='\n')
shutil.copy2(source/'IMPLEMENTATION.json',source/'pbc3/IMPLEMENTATION.json')
print(source)
