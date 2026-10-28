import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from check_features import candidate_function, pbc3_features
from feature_candidate import integer_order_stats
import psutil
psutil.Process().cpu_affinity([12])
rng=np.random.default_rng(1776)
checks=0
for size in (1,2,3,4,7,19,20,21,99,100,101,799,160000):
    for offset in (-2000,-3,0,3,2000,2**40):
        flat=rng.integers(-400,401,size=size).astype(np.float64)+offset
        actual,unique=integer_order_stats(flat,int(flat.min()),int(flat.max()))
        expected=np.percentile(flat,[5,25,50,75,95])
        assert actual.tobytes()==expected.tobytes(),(size,offset,actual,expected)
        assert unique==np.unique(np.rint(flat/2)).size/max(1,size)
        checks+=1
candidate=candidate_function()
for dtype in (np.int32,np.int64,np.float64):
    target=np.array([[[0,65535,0],[1000000,2000000,0]]],dtype=dtype)
    canvas=-target
    args=(target,canvas,(1,0,0,2,1),0,0.8,2,1,50,3)
    assert candidate(*args).tobytes()==pbc3_features.extract_features(*args).tobytes()
bench={}
for size in (16,64,200,400):
    t=rng.integers(0,256,size=(size,size,3),dtype=np.int32)
    cv=rng.integers(-100,356,size=t.shape,dtype=np.int32)
    args=(t,cv,(0,0,0,size,size),7,0.8,size,size,50,3)
    for fn in (candidate,pbc3_features.extract_features): fn(*args)
    samples={name:[] for name in ('baseline','candidate')}
    for rep in range(15):
        order=(('baseline',pbc3_features.extract_features),('candidate',candidate)) if rep%2==0 else (('candidate',candidate),('baseline',pbc3_features.extract_features))
        for name,fn in order:
            start=time.perf_counter(); fn(*args); samples[name].append(time.perf_counter()-start)
    bench[str(size)]={name:statistics.median(times) for name,times in samples.items()}
result={'order_stat_cases':checks,'wide_range_cases':3,'microbench_seconds':bench,'histogram_max_bytes':65536*8,'source_checks':{}}
for name in ('PBC3.py','pbc3_ops.py','pbc3_features.py','learned_filler.py','pbc3_heads.py','pbc3_types.py','pbc3_stream.py','pbc3_trace.py','patch_policy.npz'):
    expected=subprocess.check_output(['git','show','c0d0339a509ceb4027574fc388cef18a643b14e5:'+name],cwd=ROOT)
    actual=(ROOT/'research_pass16/baseline'/name).read_bytes()
    assert actual.replace(b'\r\n',b'\n')==expected.replace(b'\r\n',b'\n'),name
    result['source_checks'][name]={'file_sha256':hashlib.sha256(actual).hexdigest(),'git_blob_content_sha256':hashlib.sha256(expected).hexdigest()}
(Path(__file__).parent/'micro.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
