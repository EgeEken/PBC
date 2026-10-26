import hashlib, json, os, sys
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
import numpy as np
import numba
from PIL import Image, ImageOps
numba.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PBC3 import PBC3
from pbc3_types import BitReader, PBC3Config
sha=lambda b:hashlib.sha256(b).hexdigest()
cases=[('cat.jpg','quality'),('cat.jpg','high_quality'),('flower.jpg','quality'),('monet.jpg','quality')]
out=[]
for image,preset in cases:
 p=ROOT/'hpt_data'/image
 with Image.open(p) as im: src=ImageOps.exif_transpose(im).convert('RGB')
 cfg=getattr(PBC3Config,preset)()
 prep=PBC3.prepare(src,cfg)
 enc=PBC3.compress(src,config=cfg,rng_mode='release')
 method,body=PBC3._open_body(enc.data); br=BitReader(body); header=PBC3._read_header(br)
 _,ow,oh,w,h,color,channels,channel_bits,bias,alpha,count,*_=header
 trajectory=[]
 for i in range(count):
  c,x,y,pw,ph,cell,values,bits=PBC3._read_patch(br,channel_bits,bias)
  trajectory.append(dict(index=i,channel=int(c),x=int(x),y=int(y),w=int(pw),h=int(ph),cell=int(cell),grid_w=int(values.shape[1]),grid_h=int(values.shape[0]),bits=int(bits),values_sha256=sha(np.asarray(values,dtype=np.int16).tobytes())))
 out.append(dict(image=image,preset=preset,input_pixels_sha256=sha(np.asarray(src).tobytes()),prepared_array_sha256=sha(prep['arr'].tobytes()),prepared_target_sha256=sha(prep['target'].tobytes()),stream_sha256=sha(enc.data),final_bytes=len(enc.data),patch_count=count,trajectory=trajectory))
print(json.dumps(dict(python=sys.version.split()[0],numpy=np.__version__,numba=numba.__version__,pillow=Image.__version__,cases=out),separators=(',',':')))
