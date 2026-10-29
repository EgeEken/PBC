import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
sys.path.insert(0,str(CANDIDATE))
from pbc3_native import select_metric
import pbc3_ops as ops
from PBC3 import PBC3
from pbc3_types import PBC3Config
from concurrent.futures import ThreadPoolExecutor
native=select_metric('native',ops.final_mse)
rng=np.random.default_rng(2703);count=0
for dtype in (np.uint8,np.int16,np.int32,np.float32,np.float64):
    for shape in ((1,1),(19,17,3),(31,29,4)):
        a=rng.integers(0,256,shape).astype(dtype);b=rng.integers(0,256,shape).astype(dtype)
        if dtype!=np.uint8: a-=127;b-=127
        for layout in ('C','F','reverse','strided'):
            aa,bb=(np.asfortranarray(a),np.asfortranarray(b)) if layout=='F' else (a[::-1],b[::-1]) if layout=='reverse' else (a[:,::2],b[:,::2]) if layout=='strided' else (a,b)
            assert native(aa,bb)==ops.final_mse(aa,bb);count+=1
for extreme in (-32768,32767):
    a=np.full((11,13,3),extreme,np.int16);assert native(a,np.zeros_like(a))==ops.final_mse(a,np.zeros_like(a))
a=rng.integers(0,256,(43,47,4),dtype=np.uint8);b=np.flip(a,0)
with ThreadPoolExecutor(4) as pool: assert list(pool.map(lambda _:native(a,b),range(32)))==[ops.final_mse(a,b)]*32
im=Image.fromarray(a);cfg=PBC3Config.quality()
expected=PBC3.compress(im,config=cfg,mse_backend='reference')
with ThreadPoolExecutor(4) as pool:
    results=list(pool.map(lambda backend:PBC3.compress(im,config=cfg,mse_backend=backend),('reference','native','reference','native')))
assert all(r.data==expected.data and r.mse==expected.mse for r in results)
assert select_metric('reference',ops.final_mse) is ops.final_mse
save('native/validation.json',dict(direct_layout_type_cases=count,signed_extremes=2,concurrent_metric=32,concurrent_mixed_backend_encodes=4,all_exact=True,no_global_patch=True))
print('native integration correctness passed')
