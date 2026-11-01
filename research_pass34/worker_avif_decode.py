from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import psutil
manifest=json.loads(Path(sys.argv[1]).read_text());source=np.asarray(Image.open(manifest['source']).convert('RGB'),np.uint8);rows=[];proc=psutil.Process()
for item in manifest['files']:
 t=time.perf_counter()
 with Image.open(item['path']) as im: rec=np.asarray(im.convert('RGB'),np.uint8)
 if rec.shape!=source.shape:raise ValueError('fresh AVIF dimensions changed')
 mse=float(np.square(source.astype(np.int32)-rec.astype(np.int32),dtype=np.int64).mean())
 rows.append({'quality':item['quality'],'mse':mse,'decoded_sha256':hashlib.sha256(rec.tobytes()).hexdigest(),'decode_seconds':time.perf_counter()-t,'peak_working_set_bytes':proc.memory_info().peak_wset})
print(json.dumps({'rows':rows}))
