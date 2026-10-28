import collections
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.update({k: '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS')})
ROOT = Path(__file__).resolve().parents[1]
SOURCE=ROOT/'research_pass16/baseline'
sys.path.insert(0, str(SOURCE))
import numpy as np
import numba
import PIL
import psutil
from PIL import Image, ImageOps
from PBC3 import PBC3
from pbc3_types import PBC3Config
from pbc3_trace import TimingTrace
import pbc3_features as features

psutil.Process().cpu_affinity([12])
numba.set_num_threads(1)
rows = []
line_cost = collections.defaultdict(float)
line_count = collections.Counter()
last = {}
def trace_line(frame, event, arg):
    if frame.f_code is not features.extract_features.__code__:
        return None
    now = time.perf_counter()
    key = id(frame)
    if key in last:
        lineno, start = last[key]
        line_cost[lineno] += now - start
        line_count[lineno] += 1
    if event == 'return':
        last.pop(key, None)
    else:
        last[key] = (frame.f_lineno, now)
    return trace_line

for name in ('cat.jpg','eiffel.png','monet.jpg','seagull.jpg'):
    path = ROOT / 'hpt_data' / name
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    for preset in ('quality','high_quality'):
        config = getattr(PBC3Config,preset)()
        PBC3.compress(image, config=config)
        times=[]
        for _ in range(3):
            start=time.perf_counter(); PBC3.compress(image,config=config); times.append(time.perf_counter()-start)
        trace=TimingTrace('encode')
        PBC3.compress(image,config=config,trace=trace)
        report=trace.report()
        sys.settrace(trace_line)
        try: PBC3.compress(image,config=config)
        finally: sys.settrace(None)
        rows.append({'image':name,'preset':preset,'full_seconds':times,'trace':report['summary'], 'config':dataclasses.asdict(config)})
        print(name,preset,statistics.median(times),flush=True)
model=np.load(ROOT/'patch_policy.npz')
result={'baseline_commit':'c0d0339a509ceb4027574fc388cef18a643b14e5','python':sys.version,'numpy':np.__version__,'numba':numba.__version__,'pillow':PIL.__version__,'cpu':12,'model':{k:model[k].tolist() for k in ('cheap','feature_names','hidden')},'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/n for n in ('PBC3.py','pbc3_ops.py','pbc3_features.py','learned_filler.py','pbc3_heads.py','pbc3_types.py','pbc3_stream.py','patch_policy.npz')]},'images':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'hpt_data').iterdir() if p.is_file()},'rows':rows,'feature_lines':[{'line':n,'source':inspect.getsource(features).splitlines()[n-1].strip(),'seconds':v,'count':line_count[n]} for n,v in sorted(line_cost.items())]}
(ROOT/'research_pass16/profile.json').write_text(json.dumps(result,indent=2)+'\n')
print('Line costs',json.dumps(sorted(result['feature_lines'],key=lambda r:r['seconds'],reverse=True)[:14],indent=2))
