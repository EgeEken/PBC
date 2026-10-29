import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np
source=Path(__file__).resolve().parents[1]/'artifacts/published'
sys.path.insert(0,str(source))
blocked=types.ModuleType('numba')
def njit(*args,**kwargs):
    def decorate(fn):
        def forbidden(*a,**k): raise AssertionError('decoder invoked JIT kernel: '+fn.__name__)
        return forbidden
    return decorate
blocked.njit=njit;blocked.uint32=np.uint32;blocked.uint64=np.uint64
sys.modules['numba']=blocked
from PBC3 import PBC3
root=source.parents[2]
rows=[]
folder=root/'research_pass26/artifacts/packaging_tests'
for path in sorted(folder.glob('python_*.pbc'))+[folder/'stored.pbc',folder/'warmup.pbc']:
    decoded=PBC3.decompress(path.read_bytes()).image
    from PIL import Image
    expected=folder/((path.stem if path.stem.startswith('python_') else 'python_'+path.stem)+'.png')
    assert decoded.tobytes()==Image.open(expected).tobytes()
    rows.append(dict(file=path.name,sha256=hashlib.sha256(decoded.tobytes()).hexdigest()))
(root/'research_pass26/packaging/decoder_dependencies.json').write_text(json.dumps(dict(all_jit_calls_forbidden=True,rows=rows),indent=2)+'\n')
print('AI-free decoder completed without JIT:',len(rows))
