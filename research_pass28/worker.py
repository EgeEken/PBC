import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research_pass27'))
from common import *
CANDIDATE=ROOT/'research_pass28/artifacts/release-source'
PUBLISHED=ROOT/'research_pass27/artifacts/published'
import dataclasses
import time
mode=sys.argv[1]
sys.path.insert(0,str(PUBLISHED if mode=='published' else CANDIDATE))
from PBC3 import PBC3
from pbc3_types import PBC3Config
for line in sys.stdin:
    req=json.loads(line);name=req['image']
    if name in ('small','rgba'):
        image=Image.fromarray(np.random.default_rng(2702).integers(0,256,(61,79,4 if name=='rgba' else 3),dtype=np.uint8))
    else: image=load(name)
    cfg=getattr(PBC3Config,req['preset'])()
    kwargs={} if mode=='published' else dict(mse_backend='native' if mode=='native' else 'reference')
    t=time.perf_counter();r=PBC3.compress(image,config=cfg,**kwargs);enc=time.perf_counter()-t
    t=time.perf_counter();decoded=PBC3.decompress(r.data).image;dec=time.perf_counter()-t
    print(json.dumps(dict(encode_seconds=enc,decode_seconds=dec,bytes=len(r.data),bpp=len(r.data)*8/(image.width*image.height),stream_sha256=sha(r.data),pixel_sha256=sha(decoded.tobytes()),internal_mse=r.mse,rgb_mse=mse(image.convert('RGB'),decoded.convert('RGB')),config=dataclasses.asdict(cfg))),flush=True)
