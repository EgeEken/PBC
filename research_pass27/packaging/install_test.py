import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
import tempfile
import zipfile
out=PASS/'artifacts/packages';out.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-m','build','--no-isolation','--outdir',str(out),str(CANDIDATE)],check=True)
wheel=next(out.glob('*.whl'));sdist=next(out.glob('*.tar.gz'))
with zipfile.ZipFile(wheel) as z:
    names=z.namelist()
    assert 'pbc3/patch_policy.npz' in names and all(not n.endswith('.dll') for n in names)
    assert any(n.endswith('/licenses/LICENSE') for n in names)
    assert sha(z.read('pbc3/patch_policy.npz'))=='ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f'
rows=[]
for kind in ('source','wheel'):
    env=PASS/'artifacts'/('install-'+kind)
    if not env.exists(): subprocess.run(['uv','venv','--no-project',str(env),'--python',sys.executable],check=True)
    python=env/'Scripts/python.exe'
    subprocess.run(['uv','pip','install','--python',str(python),'pip==24.0'],check=True)
    item=str(CANDIDATE) if kind=='source' else str(wheel)
    subprocess.run([str(python),'-m','pip','install','--force-reinstall',item],check=True,cwd=tempfile.gettempdir())
    subprocess.run([str(python),'-m','pip','check'],check=True)
    cwd=Path(tempfile.mkdtemp(prefix='PBC27 unrelated install '))
    code="from PBC3 import PBC3;from pbc3 import PBC3 as P, PBC3Config;from pathlib import Path;from PIL import Image;import hashlib,json;assert P is PBC3;cfg=PBC3Config.quality();p=Path(cfg.learned_filler_model_path);assert p.is_file();im=Image.new('RGB',(39,37),(80,130,220));a=PBC3.compress(im,config=cfg);assert PBC3.decompress(a.data).image.size==im.size;print(json.dumps(dict(model=str(p),model_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),stream_sha256=hashlib.sha256(a.data).hexdigest())))"
    data=json.loads(subprocess.check_output([str(python),'-c',code],text=True,cwd=cwd))
    rows.append(dict(kind=kind,environment=str(env),cwd=str(cwd),**data))
assert rows[0]['stream_sha256']==rows[1]['stream_sha256']
save('packaging/installations.json',dict(rows=rows,wheel=dict(path=str(wheel),bytes=wheel.stat().st_size,sha256=sha(wheel.read_bytes())),sdist=dict(path=str(sdist),bytes=sdist.stat().st_size,sha256=sha(sdist.read_bytes()))))
print('source/wheel install tests passed')
