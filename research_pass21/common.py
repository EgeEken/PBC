import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_pass17'))
sys.path.insert(0,str(ROOT))
from research_pass17.analyze_residuals import PBC3,PBC3Config,Image,np,load,setup,provenance,sha,psutil
import hybrid as h17

HERE=Path(__file__).resolve().parent
ARTIFACTS=HERE/'artifacts'
INITIAL=(('googlewiki.png',128),('mario64.png',128),('cat.jpg',128))
EXTRA=(('monet.jpg',64),('eiffel.png',32),('flower.jpg',128))
START='53410c1bc4bef78c77e31bffb36c9a466c923f0d'


def write_json(name,value):
    (HERE/name).write_text(json.dumps(value,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')


def meta():
    result=provenance()
    result.update(research_start_commit=START,research_source_sha256={p.name:sha(p.read_bytes()) for p in HERE.glob('*.py')},plan_sha256=sha((HERE/'PLAN.md').read_bytes()))
    return result


def frozen(name,step):
    data=(ROOT/'research_pass17/artifacts'/(name+f'.dct_q{step}.p17')).read_bytes()
    fields=h17.HEADER.unpack_from(data)
    base=data[h17.HEADER.size:h17.HEADER.size+fields[3]]
    prior=json.loads((ROOT/'research_pass17/hybrid_results.json').read_text())['rows']
    expected=next(r for r in prior if r['image']==name and r['step']==step)
    assert sha(data)==expected['stream_sha256']
    return data,base,expected


def coefficients(name,step):
    data,base,expected=frozen(name,step)
    fields=h17.HEADER.unpack_from(data)
    cache=ROOT/'research_pass20/artifacts'/(name+'.q128.npy')
    if step==128 and (name,step) in INITIAL:
        q=np.load(cache)
        recovered=h17.unpack_coefficients(data[h17.HEADER.size+len(base):],fields[2],fields[1])
        assert np.array_equal(q,recovered)
    else:q=h17.unpack_coefficients(data[h17.HEADER.size+len(base):],fields[2],fields[1])
    return q
