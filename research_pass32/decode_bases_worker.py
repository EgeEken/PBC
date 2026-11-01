from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

def sha(data): return hashlib.sha256(data).hexdigest()

def main():
    request=json.loads(Path(sys.argv[1]).read_text()); sys.path.insert(0,request['runtime'])
    from PBC3 import PBC3
    rows=[]
    for item in request['images']:
        source=np.asarray(ImageOps.exif_transpose(Image.open(item['source_path'])).convert('RGB'),np.uint8)
        image=PBC3.decompress(Path(item['stream']).read_bytes()).image.convert('RGB')
        base=np.asarray(image,np.uint8)
        if base.shape!=tuple(item['source_shape']) or source.shape!=base.shape:
            raise ValueError('decoded base dimensions mismatch')
        np.save(item['base_npy'],base,allow_pickle=False)
        diff=source.astype(np.int32)-base.astype(np.int32)
        rows.append({'image':item['image'],'shape':list(base.shape),'decoded_base_sha256':sha(base.tobytes()),
                     'mse':float(np.square(diff,dtype=np.int64).sum()/(base.shape[0]*base.shape[1]*3))})
    print(json.dumps({'images':rows}))
if __name__=='__main__': main()