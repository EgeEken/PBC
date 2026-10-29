import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
sys.path.insert(0,str(PUBLISHED))
from PBC3 import PBC3
from pbc3_types import PBC3Config
image=load('eiffel.png').resize((43,41));path=PASS/'artifacts/install-probe.png';image.save(path)
expected={n:sha(PBC3.compress(image,config=getattr(PBC3Config,n)()).data) for n in ('compression','balanced','quality','high_quality')}
rows=[]
for kind in ('source','wheel'):
    python=PASS/'artifacts'/('install-'+kind)/'Scripts/python.exe'
    code="import sys,json,hashlib;from pathlib import Path;from PIL import Image;from PBC3 import PBC3;from pbc3 import PBC3 as P,PBC3Config;assert P is PBC3;assert Path(sys.modules['PBC3'].__file__).resolve().is_relative_to(Path(sys.prefix)/'Lib/site-packages');image=Image.open(sys.argv[1]);rows={};"+"\nfor preset in ('compression','balanced','quality','high_quality'):\n cfg=getattr(PBC3Config,preset)();model=Path(cfg.learned_filler_model_path);assert model.is_relative_to(Path(sys.prefix)/'Lib/site-packages');assert hashlib.sha256(model.read_bytes()).hexdigest()=='ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f';r=PBC3.compress(image,config=cfg);assert PBC3.decompress(r.data).image.size==image.size;rows[preset]=hashlib.sha256(r.data).hexdigest()\nprint(json.dumps(rows))"
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    actual=json.loads(subprocess.check_output([str(python),'-c',code,str(path)],cwd=PASS/'artifacts',env=env,text=True));assert actual==expected
    rows.append(dict(installation=kind,streams=actual,actual_site_packages_import=True,installed_model_sha256_verified=True))
save('packaging/install_validation.json',rows);print('installed all-presets/reference checks passed')
