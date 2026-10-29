import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
import statistics as st
modes=('published','candidate_same_environment','candidate_without_numba')
workers={mode:subprocess.Popen([str(PASS/'artifacts/decoder-env/Scripts/python.exe') if mode=='candidate_without_numba' else sys.executable,str(PASS/'decoder/worker.py'),'published' if mode=='published' else 'candidate'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=dict(os.environ,PYTHONHASHSEED='2003')) for mode in modes}
rows=[]
try:
    for name in ('cat.jpg.quality','googlewiki.png.quality','eiffel.png.high_quality','python_rgba'):
        request=json.dumps(dict(file=str(PASS/'artifacts/corpus'/(name+'.pbc'))))+'\n'
        for w in workers.values():w.stdin.write(request);w.stdin.flush();json.loads(w.stdout.readline())
        for repeat in range(3):
            pair={}
            for mode in modes[repeat:]+modes[:repeat]:
                w=workers[mode];w.stdin.write(request);w.stdin.flush();pair[mode]=json.loads(w.stdout.readline())
            assert len(set(v['pixel_sha256'] for v in pair.values()))==1
            rows.append(dict(image=name,repeat=repeat,**pair))
finally:
    for w in workers.values():w.stdin.close();w.wait(timeout=30)
save('decoder/benchmark.json',rows)
for name in sorted(set(r['image'] for r in rows)):
    subset=[r for r in rows if r['image']==name]
    print(name,{m:st.median(r[m]['seconds'] for r in subset) for m in modes})
