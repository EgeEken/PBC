import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass17.analyze_residuals import setup,provenance,load,sha,np,PBC3,PBC3Config,Image,psutil,blocks
from research_pass2.stream_tools import parse,serialize
import pbc3_ops as ops

HERE=Path(__file__).resolve().parent
ARTIFACTS=HERE/'artifacts'
IMAGES=('googlewiki.png','monet.jpg','cat.jpg','flower.jpg')
STEPS=(64,128,256)


def write(name,value):
    (HERE/name).write_text(json.dumps(value,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')


def meta():
    result=provenance();result.pop('timing',None)
    result.update(start_commit='4b41f1794d005241444e825c2b6ce125ed135e7f',source_sha256_pass25={p.name:sha(p.read_bytes()) for p in HERE.glob('*.py')},plan_sha256=sha((HERE/'PLAN.md').read_bytes()),sparse_source_sha256=sha((ROOT/'research_pass21/zero_runs.py').read_bytes()),quantization_steps=STEPS)
    return result


def mse(a,b):
    error=np.subtract(a,b,dtype=np.float32)
    return float(np.mean(error**2,dtype=np.float64))


def saved(name,preset):
    row=next(r for r in json.loads((ROOT/'research_pass17/baseline.json').read_text())['rows'] if r['image']==name and r['preset']==preset)
    data=(ROOT/'research_pass17'/row['file']).read_bytes()
    assert len(data)==row['bytes'] and sha(data)==row['stream_sha256']
    return data,row
