import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import dataclasses
import hashlib
import json
import time
import traceback
import numba
import numpy as np
import psutil
from PIL import Image,ImageOps
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
variant=sys.argv[1]
SOURCE=(HERE/'artifacts/sources'/variant).resolve()
sys.path.insert(0,str(SOURCE))
os.chdir(SOURCE)
os.environ['NUMBA_CACHE_DIR']=str(HERE/'artifacts/cache'/variant)
numba.set_num_threads(1)
psutil.Process().cpu_affinity([10])
from PBC3 import PBC3
from pbc3_types import PBC3Config,BitReader
images={}
for name in json.loads((ROOT/'research/assets.json').read_text())['inputs']:
    with Image.open(ROOT/name) as opened:
        images[name]=ImageOps.exif_transpose(opened).convert('RGB')
for name,module in tuple(sys.modules.items()):
    if name=='PBC3' or name.startswith('pbc3') or name=='learned_filler':
        origin=getattr(module,'__file__',None)
        if origin: assert Path(origin).resolve().is_relative_to(SOURCE),(name,origin)
def sha(data): return hashlib.sha256(data).hexdigest()
print(json.dumps(dict(ready=True,source=str(SOURCE),affinity=psutil.Process().cpu_affinity(),
    origins={n:str(m.__file__) for n,m in sys.modules.items() if (n=='PBC3' or n.startswith('pbc3') or n=='learned_filler') and getattr(m,'__file__',None)},
    native_dll=variant=='candidate')),flush=True)
for line in sys.stdin:
    req=json.loads(line)
    try:
        image=images[req['image']]
        cfg=getattr(PBC3Config,req['preset'])()
        assert cfg.compute_final_mse and cfg.random_seed==2003
        kw=dict(mse_backend='native') if variant=='candidate' else {}
        start=time.perf_counter()
        result=PBC3.compress(image,config=cfg,**kw)
        enc=time.perf_counter()-start
        start=time.perf_counter()
        decoded=PBC3.decompress(result.data).image
        dec=time.perf_counter()-start
        scored=decoded.convert('RGB')
        if scored.size!=image.size: scored=scored.resize(image.size,Image.Resampling.BICUBIC)
        a=np.asarray(image,dtype=np.float64);b=np.asarray(scored,dtype=np.float64)
        rgb_mse=float(np.mean((a-b)**2))
        body=PBC3._open_body(result.data)[1]
        patches=int(PBC3._read_header(BitReader(body))[10])
        if req.get('save'):
            path=HERE/'artifacts/streams'/variant/(Path(req['image']).name+'.'+req['preset']+'.pbc')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(result.data)
        row=dict(variant=variant,image=req['image'],preset=req['preset'],repeat=req.get('repeat'),
            encode_seconds=enc,decode_seconds=dec,bytes=len(result.data),
            bpp=len(result.data)*8/(image.width*image.height),stream_sha256=sha(result.data),
            pixel_sha256=sha(decoded.tobytes()),mode=decoded.mode,size=decoded.size,source_size=image.size,
            rgb_mse=rgb_mse,internal_mse=result.mse,patch_count=patches,config=dataclasses.asdict(cfg),error=None)
    except Exception:
        row=dict(variant=variant,**req,error=traceback.format_exc())
    print(json.dumps(row),flush=True)
