import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[n]='1'
import ctypes
ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(),1<<12)
import sys
from pathlib import Path
import json,hashlib,time
PASS=Path(__file__).resolve().parents[1]
mode=sys.argv[1]
sys.path.insert(0,str(PASS.parent/'research_pass26/artifacts/published' if mode=='published' else PASS/'artifacts/candidate'))
if mode=='published':
    from PBC3 import PBC3
    def decode(data):return PBC3.decompress(data).image
else:
    from pbc3_decoder import decode
for line in sys.stdin:
    data=Path(json.loads(line)['file']).read_bytes()
    t=time.perf_counter();image=decode(data);elapsed=time.perf_counter()-t
    print(json.dumps(dict(seconds=elapsed,pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())),flush=True)
