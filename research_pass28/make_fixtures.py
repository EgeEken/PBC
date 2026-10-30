import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_pass27/artifacts/published'))
from PBC3 import PBC3
from pbc3_types import PBC3Config
sha=lambda b:hashlib.sha256(b).hexdigest()
fixtures=dict(format=1,producer='actual published 3.1; edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede',inputs=[],streams=[],invalid=[])
for mode in ('RGB','RGBA'):
    array=np.random.default_rng(2801).integers(0,256,(37,39,len(mode)),dtype=np.uint8)
    image=Image.fromarray(array);png=io.BytesIO();image.save(png,format='PNG')
    row=dict(name=mode,mode=mode,size=list(image.size),png=base64.b64encode(png.getvalue()).decode(),expected={})
    for preset in ('compression','balanced','quality','high_quality'):
        result=PBC3.compress(image,config=getattr(PBC3Config,preset)())
        decoded=PBC3.decompress(result.data).image
        row['expected'][preset]=dict(stream_sha256=sha(result.data),pixel_sha256=sha(decoded.tobytes()),mode=decoded.mode,size=list(decoded.size))
    fixtures['inputs'].append(row)
corpus=json.loads((ROOT/'research_pass27/decoder/corpus.json').read_text())
for row in corpus['valid']:
    if row['name'] in ('historical30','python_quality','python_rgba','stored','warmup','boundary_0_RGB','boundary_1_YCbCr','signed_wrap_-1'):
        data=Path(row['file']).read_bytes()
        fixtures['streams'].append({k:row[k] for k in ('name','mode','size','stream_sha256','pixel_sha256','partial','origin') if k in row}|dict(data=base64.b64encode(data).decode()))
assert any(r['name']=='historical30' for r in fixtures['streams'])
for path in corpus['invalid'][:13]:
    path=Path(path);fixtures['invalid'].append(dict(name=path.name,data=base64.b64encode(path.read_bytes()).decode()))
(ROOT/'research_pass28/smoke/fixtures.json').write_text(json.dumps(fixtures,indent=2)+'\n',encoding='utf-8',newline='\n')
print('fixtures',len(fixtures['streams']),len(fixtures['invalid']))
