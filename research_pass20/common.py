import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_pass17'))
sys.path.insert(0,str(ROOT))
from research_pass17.analyze_residuals import PBC3,PBC3Config,Image,np,load,setup,provenance,sha,blocks,psutil
import hybrid as h17

HERE=Path(__file__).resolve().parent
ARTIFACTS=HERE/'artifacts'
INITIAL=('googlewiki.png','mario64.png','cat.jpg')
EXTRA=('monet.jpg','eiffel.png','flower.jpg')
START='0a2f1e80d7fd5aaed5acf683eea03aeb1a7b947e'


def write_json(name,value):
    (HERE/name).write_text(json.dumps(value,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')


def meta():
    result=provenance()
    result.update(research_start_commit=START,research_source_sha256={p.name:sha(p.read_bytes()) for p in HERE.glob('*.py')},plan_sha256=sha((HERE/'PLAN.md').read_bytes()))
    return result


def frozen_base(name):
    data=(ROOT/'research_pass17/artifacts'/(name+'.dct_q128.p17')).read_bytes()
    fields=h17.HEADER.unpack_from(data)
    base=data[h17.HEADER.size:h17.HEADER.size+fields[3]]
    prior=json.loads((ROOT/'research_pass17/extra_controls.json').read_text())['rows']
    expected=next(r for r in prior if r['image']==name and r['preset']=='quality_p10_base')
    assert sha(base)==expected['stream_sha256']
    return base


def mse(image,output):
    difference=np.asarray(image,dtype=np.float32)-np.asarray(output,dtype=np.float32)
    return float(np.mean(difference**2,dtype=np.float64))
