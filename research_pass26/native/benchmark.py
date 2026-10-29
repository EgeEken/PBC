import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
import statistics
import time
from kernel import mse_u8
import pbc3_ops as ops
micro=[]
for name in ('googlewiki.png','cat.jpg'):
    image=load(name);a=np.asarray(image);b=np.roll(a,1,axis=0)
    ops.image_mse(a,b);mse_u8(a,b)
    for kind in ('array','PIL'):
        aa,bb=(image,Image.fromarray(b)) if kind=='PIL' else (a,b)
        timings={'reference':[],'native':[]}
        for rep in range(5):
            for mode in (('reference','native') if rep%2==0 else ('native','reference')):
                fn=ops.final_mse if mode=='reference' else mse_u8
                t=time.perf_counter();value=fn(aa,bb);elapsed=time.perf_counter()-t
                assert value==ops.image_mse(a,b)
                timings[mode].append(elapsed)
        micro.append(dict(image=name,input=kind,seconds=timings))
workers={m:subprocess.Popen([sys.executable,str(PASS/'native/worker.py'),m],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) for m in ('reference','native')}
cases=[(n,p) for n in ('cat.jpg','googlewiki.png','eiffel.png') for p in ('quality','high_quality')]+[('rgba','quality')]
rows=[]
try:
    for image,preset in cases:
        request=json.dumps(dict(image=image,preset=preset))+'\n'
        for mode,w in workers.items():
            w.stdin.write(request);w.stdin.flush();json.loads(w.stdout.readline())
        for rep in range(3):
            pair={}
            for mode in (('reference','native') if rep%2==0 else ('native','reference')):
                w=workers[mode];w.stdin.write(request);w.stdin.flush();pair[mode]=json.loads(w.stdout.readline())
            for key in ('stream_sha256','pixel_sha256','bytes','internal_mse','rgb_mse'):
                assert pair['reference'][key]==pair['native'][key],(image,preset,key)
            rows.append(dict(image=image,preset=preset,repeat=rep,**pair))
        print(image,preset,flush=True)
finally:
    for w in workers.values():
        w.stdin.close();w.wait(timeout=30)
save('native/benchmark.json',dict(micro=micro,rows=rows))
