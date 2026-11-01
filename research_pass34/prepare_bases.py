from __future__ import annotations
import hashlib,json,os,platform,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import numba
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PBC3 import PBC3
from pbc3_types import PBC3Config
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
OUT=HERE/'outputs'

def sha(data): return hashlib.sha256(data).hexdigest()

def main():
    selection=json.loads((HERE/'dataset_selection.json').read_text())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    model=ROOT/'patch_policy.npz'
    cfg=PBC3Config.quality(learned_filler_model_path=str(model))
    numba.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    rows=[]
    for chosen in selection['images']:
        source=ROOT/'hpt_data'/chosen['filename']
        raw=source.read_bytes()
        if sha(raw)!=chosen['source_sha256']:
            raise ValueError(f'input hash changed: {source.name}')
        image=ImageOps.exif_transpose(Image.open(source)).convert('RGB')
        if image.size!=(chosen['width'],chosen['height']): raise ValueError(f'dimensions changed: {source.name}')
        started=time.perf_counter()
        encoded=PBC3.compress(image,config=cfg,rng_mode='release')
        elapsed=time.perf_counter()-started
        stream=OUT/'bases'/f'{source.name}.pbc3'; stream.parent.mkdir(exist_ok=True)
        stream.write_bytes(encoded.data)
        rows.append({'image':source.name,'source_sha256':sha(raw),'width':image.width,'height':image.height,
            'base_stream_path':str(stream.relative_to(HERE)),'base_stream_sha256':sha(encoded.data),'base_bytes':len(encoded.data),
            'encode_seconds':elapsed,'rng_mode':'release','config':asdict(cfg)})
        print('encoded base',source.name,len(encoded.data),f'{elapsed:.2f}s',flush=True)
    source_files=('PBC3.py','pbc3_types.py','pbc3_ops.py','pbc3_heads.py','pbc3_features.py','pbc3_stream.py','pbc3_trace.py','learned_filler.py','patch_policy.npz')
    provenance={'selection_head':selection['selection_head'],'base_creation_head':head,
        'baseline_id':'pbc31_release_rng_baseline','baseline_manifest_path':'research_pass10/baseline_manifest.json',
        'baseline_manifest_sha256':sha((ROOT/'research_pass10/baseline_manifest.json').read_bytes()),
        'pinned_release_rng_revision':'8006f30debfc4a57289df06df21636aeb52baffd',
        'implementation':'root PBC3.compress(..., rng_mode="release") at base_creation_head; code hashes below; pbc3_features.py and pbc3_ops.py include later semantics-preserving runtime kernels relative to pinned revision',
        'model_sha256':sha(model.read_bytes()),'source_code_sha256':{name:sha((ROOT/name).read_bytes()) for name in source_files if name!='patch_policy.npz'},
        'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'numba':numba.__version__,'pillow':Image.__version__ if hasattr(Image,'__version__') else __import__('PIL').__version__,'threads':1},
        'quality_config':asdict(cfg),'images':rows}
    (OUT/'base_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    run=subprocess.run([sys.executable,'-B',str(HERE/'decode_bases_fresh.py')],cwd=ROOT,check=True,capture_output=True,text=True)
    print(run.stdout,end='')
    verification=json.loads((OUT/'base_decode_verification.json').read_text())
    if len(verification['images'])!=4 or not all(x['fresh_process_verified'] for x in verification['images']):
        raise AssertionError('base fresh-process decode verification failed')
    print('all four bases independently decoded in fresh process',flush=True)
if __name__=='__main__': main()
