import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from kernel import mse_u8,install
import pbc3_ops as ops
from concurrent.futures import ThreadPoolExecutor
rng=np.random.default_rng(2601)
rows=[]
for shape in ((1,1),(1,1,3),(1,31,4),(27,1,3),(9,17,3),(32,47,4),(128,129,3)):
    a=rng.integers(0,256,shape,dtype=np.uint8);b=rng.integers(0,256,shape,dtype=np.uint8)
    for kind in ('C','F','reverse','stride'):
        aa,bb=(np.asfortranarray(a),np.asfortranarray(b)) if kind=='F' else (a[::-1],b[::-1]) if kind=='reverse' else (a[:,::2],b[:,::2]) if kind=='stride' else (a,b)
        expected=ops.image_mse(aa,bb)
        assert mse_u8(aa,bb)==expected
        rows.append(dict(shape=aa.shape,layout=kind,mse=expected))
a=np.zeros((11,7,4),np.uint8);b=np.full_like(a,255)
assert mse_u8(a,b)==65025 and mse_u8(a,a)==0
for _ in range(100):
    shape=(int(rng.integers(1,65)),int(rng.integers(1,65)),int(rng.integers(1,5)))
    a=rng.integers(0,256,shape,dtype=np.uint8);b=rng.integers(0,256,shape,dtype=np.uint8)
    assert mse_u8(a,b)==ops.image_mse(a,b)
with ThreadPoolExecutor(4) as pool:
    assert list(pool.map(lambda _:mse_u8(a,b),range(32)))==[ops.image_mse(a,b)]*32
invalid=[(np.zeros((0,2),np.uint8),np.zeros((0,2),np.uint8)),(np.zeros((2,0,3),np.uint8),np.zeros((2,0,3),np.uint8)),(np.zeros((2,3),np.uint8),np.zeros((3,2),np.uint8)),(np.zeros((2,3,5),np.uint8),np.zeros((2,3,5),np.uint8)),(np.array([[-32768,32767]],np.int16),np.zeros((1,2),np.int16)),(np.array([[-1]],np.int32),np.zeros((1,1),np.int32))]
for a,b in invalid:
    try: mse_u8(a,b)
    except (ValueError,TypeError): pass
    else: raise AssertionError('accepted invalid input')
original=install()
for dtype in (np.int16,np.int32,np.float32):
    a=np.array([[-32768,32767,-1,0]],dtype=dtype);b=np.zeros_like(a)
    assert ops.final_mse(a,b)==original(a,b)
ops.final_mse=original
save('native/correctness.json',dict(layout_fixtures=rows,randomized=100,extremes=2,concurrent_calls=32,rejected=len(invalid),fallback_signed_and_float_cases=3))
print('native correctness passed')
