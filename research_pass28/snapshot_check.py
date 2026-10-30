import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];PASS=ROOT/'research_pass28';dist=PASS/'artifacts/final'
manifest=json.loads((PASS/'PBC32_DISTRIBUTION_MANIFEST.json').read_text())
entry=next(r for r in manifest['artifacts'] if r['kind']=='release_source')
archive=Path(entry['path']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==entry['sha256']
root=Path(tempfile.mkdtemp(prefix='PBC28 exact source ',dir=PASS/'artifacts'))
with zipfile.ZipFile(archive) as z:z.extractall(root)
source=root/'PBC-3.2.0-source'
for name,digest in json.loads((PASS/'SOURCE_TREE.json').read_text())['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
subprocess.run(['gcc','-O3','-std=c11','-Wall','-Wextra','-Werror','-shared','-static-libgcc','pbc3/mse.c','-o','pbc3/mse.dll'],cwd=source,check=True)
output=root/'rebuilt'
for kind in ('combined','decoder'):
    subprocess.run([sys.executable,str(source/'tools/build_windows.py'),'--kind',kind,'--decoder-python',str(ROOT/'research_pass27/artifacts/decoder-env/Scripts/python.exe'),'--output',str(output)],check=True,cwd=root)
subprocess.run([sys.executable,'-m','build','--wheel','--no-isolation','--outdir',str(output),str(source)],check=True,cwd=root)
with zipfile.ZipFile(next(dist.glob('*.whl'))) as a,zipfile.ZipFile(next(output.glob('*.whl'))) as b:
    assert set(a.namelist())==set(b.namelist())
    assert all(a.read(n)==b.read(n) for n in a.namelist())
fixtures=json.loads((PASS/'smoke/fixtures.json').read_text());checks=[]
import os
env=dict(os.environ,PATH=r'C:\Windows\System32',NUMBA_CACHE_DIR=str(root/'cache'))
for k in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):env.pop(k,None)
combined=output/'combined/onedir/pbc/pbc.exe';decoder=output/'decoder/onedir/pbc-decode/pbc-decode.exe'
for r in fixtures['inputs']:
    input=root/(r['name']+'.png');input.write_bytes(base64.b64decode(r['png']))
    for preset,expected in r['expected'].items():
        encoded=root/(r['name']+preset+'.pbc');decoded=encoded.with_suffix('.png')
        subprocess.run([str(combined),'encode',str(input),str(encoded),'--preset',preset],env=env,cwd=root,check=True)
        assert hashlib.sha256(encoded.read_bytes()).hexdigest()==expected['stream_sha256']
        subprocess.run([str(decoder),str(encoded),str(decoded)],env=env,cwd=root,check=True)
        image=Image.open(decoded);assert list(image.size)==expected['size'] and hashlib.sha256(image.tobytes()).hexdigest()==expected['pixel_sha256']
        checks.append(r['name']+'/'+preset)
result=dict(archive_sha256=entry['sha256'],root=str(root),source_files_exact=True,rebuilt_combined_decoder=True,wheel_payloads_identical=True,reference_comparisons=checks,python_compiler_off_runtime_path=True,uses_prior_research_source=False,pe_bit_identity_promised=False)
(PASS/'artifacts/snapshot-check.json').write_text(json.dumps(result,indent=2)+'\n')
print('SNAPSHOT_PASS',len(checks),'reference comparisons')
