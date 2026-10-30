import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[n]='1'
ROOT=Path(__file__).resolve().parents[1]
PASS=ROOT/'research_pass28'
fixtures=json.loads((PASS/'smoke/fixtures.json').read_text())
if len(sys.argv)>1:
    mode=sys.argv[1]
    source=ROOT/'research_pass27/artifacts/published' if mode=='published' else PASS/'artifacts/release-source'
    sys.path.insert(0,str(source))
    import numpy as np
    from PIL import Image
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    sys.path.append(str(PASS/'artifacts/release-source'))
    from pbc3_decoder import decode
    sha=lambda data:hashlib.sha256(data).hexdigest()
    if mode=='published':
        # The lightweight reader is used only as the independent comparison.
        sys.path.insert(0,str(PASS/'artifacts/release-source'))
    rows=[]
    for row in fixtures['inputs']:
        image=Image.open(io.BytesIO(base64.b64decode(row['png'])))
        for preset in ('compression','balanced','quality','high_quality'):
            kwargs={} if mode=='published' else dict(mse_backend=mode)
            result=PBC3.compress(image,config=getattr(PBC3Config,preset)(),**kwargs)
            assert PBC3.compress(image,config=getattr(PBC3Config,preset)(),**kwargs).data==result.data
            reconstructed=PBC3.decompress(result.data).image
            assert reconstructed.size==image.size and reconstructed.mode==image.mode
            assert decode(result.data).tobytes()==reconstructed.tobytes()
            rows.append(dict(name=row['name'],preset=preset,stream=sha(result.data),pixels=sha(reconstructed.tobytes()),mse=result.mse))
    for row in fixtures['streams']:
        data=base64.b64decode(row['data']);out=PBC3.decompress(data).image
        assert sha(out.tobytes())==row['pixel_sha256'] and list(out.size)==row['size']
        for limit,expected in row['partial'].items():assert sha(decode(data,int(limit)).tobytes())==expected
    print(json.dumps(rows))
else:
    result={}
    for mode in ('published','reference','native'):
        result[mode]=json.loads(subprocess.check_output([sys.executable,__file__,mode],text=True))
    assert result['published']==result['reference']==result['native']
    (PASS/'artifacts/runtime-check.json').write_text(json.dumps(dict(exact=True,comparisons=8,known_streams=len(fixtures['streams']),partial_checks=2*len(fixtures['streams']),rows=result),indent=2)+'\n')
    print('RUNTIME_PASS',8,'three-way exact comparisons')
