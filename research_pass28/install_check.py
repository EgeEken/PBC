import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
PASS=ROOT/'research_pass28'
artifacts=PASS/'artifacts'
expected=json.loads((artifacts/'runtime-check.json').read_text())['rows']['published']
code=r"""
import base64,hashlib,importlib.metadata,io,json,sys
from pathlib import Path
from PIL import Image
from PBC3 import PBC3
import PBC3 as module
from pbc3 import PBC3 as Alias,PBC3Config
assert Alias is PBC3 and Path(module.__file__).is_relative_to(Path(sys.prefix))
assert importlib.metadata.version('pbc-codec')=='3.2.0'
fixtures=json.loads(Path(sys.argv[1]).read_text())
expected=json.loads(Path(sys.argv[2]).read_text())['rows']['published']
implementation=json.loads((Path(module.__file__).parent/'pbc3/IMPLEMENTATION.json').read_text())
for name,digest in implementation['files'].items():
    if name=='pbc3/mse.c' or name.endswith(('.py','.npz')):
        assert hashlib.sha256((Path(module.__file__).parent/name).read_bytes()).hexdigest()==digest,name
rows=[]
for row in fixtures['inputs']:
    im=Image.open(io.BytesIO(base64.b64decode(row['png'])))
    for preset in ('compression','balanced','quality','high_quality'):
        cfg=getattr(PBC3Config,preset)();model=Path(cfg.learned_filler_model_path)
        assert model.is_file() and model.is_relative_to(Path(sys.prefix))
        result=PBC3.compress(im,config=cfg);decoded=PBC3.decompress(result.data).image
        rows.append(dict(name=row['name'],preset=preset,stream=hashlib.sha256(result.data).hexdigest(),pixels=hashlib.sha256(decoded.tobytes()).hexdigest(),mse=result.mse))
assert rows==expected
print(json.dumps(dict(version='3.2.0',module=module.__file__,model=str(model),comparisons=len(rows),exact=True)))
"""
rows=[]
for kind,pattern in (('wheel','*.whl'),('sdist','*.tar.gz')):
    package=next((artifacts/'final').glob(pattern))
    venv=artifacts/('install-'+kind)
    if not venv.exists():subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
    python=venv/'Scripts/python.exe'
    subprocess.run([str(python),'-m','pip','install','--force-reinstall',str(package)],check=True)
    subprocess.run([str(python),'-m','pip','check'],check=True)
    cwd=tempfile.mkdtemp(prefix='PBC28 install ')
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    result=json.loads(subprocess.check_output([str(python),'-c',code,str(PASS/'smoke/fixtures.json'),str(artifacts/'runtime-check.json')],cwd=cwd,env=env,text=True))
    rows.append(dict(kind=kind,package=str(package),**result))
(artifacts/'install-check.json').write_text(json.dumps(rows,indent=2)+'\n')
print('INSTALL_PASS',len(rows),'fresh unrelated-CWD installs')
