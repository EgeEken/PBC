from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import numba
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PBC3 import PBC3
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));OUT=HERE/'outputs'
numba.set_num_threads(1)
def sha(data): return hashlib.sha256(data).hexdigest()
rows=[]
manifest=json.loads((OUT/'base_manifest.json').read_text())
for row in manifest['images']:
    stream=(HERE/row['base_stream_path']).read_bytes()
    decoded=PBC3.decompress(stream).image.convert('RGB')
    array=np.asarray(decoded,dtype=np.uint8)
    if array.shape!=(row['height'],row['width'],3): raise ValueError('decoded base dimensions mismatch')
    np.save(OUT/'bases'/f"{row['image']}.base.npy",array,allow_pickle=False)
    rows.append({'image':row['image'],'pixels_sha256':sha(np.ascontiguousarray(array).tobytes()),'fresh_process_verified':True})
(HERE/'outputs'/'base_decode_verification.json').write_text(json.dumps({'images':rows},indent=2)+'\n')
print(json.dumps(rows))
