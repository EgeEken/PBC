import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys
source=Path(sys.argv[1]).resolve() if sys.argv[1]!='installed' else None
if source: sys.path.insert(0,str(source))
import base64
import dataclasses
import hashlib
import io
import json
import time
import numpy as np
from PIL import Image
from PBC3 import PBC3
from pbc3_types import PBC3Config
import pbc3_native
import pbc3_decoder
import pbc3
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True,exist_ok=True)
fixtures=json.loads((ROOT/'research_pass28/smoke/fixtures.json').read_text())
def sha(data):return hashlib.sha256(data).hexdigest()
rows=[]
inputs=[]
for row in fixtures['inputs']:
    image=Image.open(io.BytesIO(base64.b64decode(row['png']))).copy()
    inputs.append((row['name'],image))
for name,size in [('one_pixel',(1,7)),('wide',(317,5))]:
    inputs.append((name,Image.fromarray(np.random.default_rng(3100).integers(0,256,(size[1],size[0],3),dtype=np.uint8))))
for name,image in inputs:
    image.save(OUT/(name+'.png'))
    for preset in ('compression','balanced','quality','high_quality'):
        config=getattr(PBC3Config,preset)()
        a=PBC3.compress(image,config=config)
        b=PBC3.compress(image,config=getattr(PBC3Config,preset)())
        assert a.data==b.data
        decoded=PBC3.decompress(a.data).image
        assert decoded.mode==image.mode and decoded.size==image.size
        assert decoded.tobytes()==pbc3_decoder.decode(a.data).tobytes()
        assert a.image.tobytes()==decoded.tobytes()
        (OUT/(name+'.'+preset+'.pbc')).write_bytes(a.data)
        decoded.save(OUT/(name+'.'+preset+'.png'))
        rows.append(dict(name=name,preset=preset,stream=sha(a.data),pixels=sha(decoded.tobytes()),size=decoded.size,mode=decoded.mode,mse=a.mse))
image=inputs[0][1]
config=PBC3Config.quality(patch_count=7,search_depth=15,proposal_depth=8,exact_depth=3,use_lzma=False,color_space='RGB',downsample_rate=1.0)
result=PBC3.compress(image,config=config)
rows.append(dict(name='manual',stream=sha(result.data),pixels=sha(PBC3.decompress(result.data).image.tobytes()),mse=result.mse))
known=[]
for row in fixtures['streams']:
    data=base64.b64decode(row['data'])
    decoded=PBC3.decompress(data).image
    assert sha(decoded.tobytes())==row['pixel_sha256']
    known.append(row['name'])
corpus=json.loads((ROOT/'research_pass27/decoder/corpus.json').read_text())
for row in corpus['valid']:
    data=Path(row['file']).read_bytes()
    decoded=PBC3.decompress(data).image
    assert sha(decoded.tobytes())==row['pixel_sha256']
    assert list(decoded.size)==row['size'] and decoded.mode==row['mode']
    for limit,expected in row['partial'].items():
        assert sha(pbc3_decoder.decode(data,int(limit)).tobytes())==expected
for name in corpus['invalid']:
    try:pbc3_decoder.decode(Path(name).read_bytes())
    except (ValueError,EOFError,IndexError,KeyError,OverflowError):pass
    else:raise AssertionError('Accepted invalid input '+name)
large=OUT/'oversized.pbc'
with large.open('wb') as f:f.truncate(pbc3_decoder.MAX_FILE+1)
try:pbc3_decoder.read_file(large)
except ValueError:pass
else:raise AssertionError('Accepted oversized file')
large.unlink()
assert isinstance(pbc3_native.select_metric('auto',lambda a,b:None),pbc3_native.NativeMSE)
assert pbc3.__version__ in ('3.1.0','3.2.0')
origin=Path(sys.modules['PBC3'].__file__).resolve()
if source:assert origin.is_relative_to(source)
else:assert origin.is_relative_to(Path(sys.prefix))
result=dict(rows=rows,known_streams=known,corpus_valid=len(corpus['valid']),
    partial_checks=sum(len(r['partial']) for r in corpus['valid']),invalid_rejected=len(corpus['invalid'])+1,
    module_origin=str(origin),model=PBC3Config().learned_filler_model_path,
    native_dll=str(pbc3_native.DLL),version=pbc3.__version__)
(OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
