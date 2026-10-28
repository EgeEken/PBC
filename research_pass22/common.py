import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass17.analyze_residuals import setup,provenance,sha,np,PBC3,load,psutil
from research_pass21 import sparse_hybrid as reference,zero_runs

HERE=Path(__file__).resolve().parent
CASES=('googlewiki.png','eiffel.png','cat.jpg')


def write(name,value):
    (HERE/name).write_text(json.dumps(value,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')


def meta():
    result=provenance()
    result.update(start_commit='a0824ff1a319f6cadcfbc16d146f666ec9f91832',source_sha256_pass22={p.name:sha(p.read_bytes()) for p in HERE.glob('*.py')},plan_sha256=sha((HERE/'PLAN.md').read_bytes()))
    for name in ('research_pass21/sparse_hybrid.py','research_pass21/zero_runs.py','research_pass17/hybrid.py'):
        result['source_sha256'][name]=sha((ROOT/name).read_bytes())
    return result


def saved(name,ordinary=False):
    folder=ROOT/('research_pass17' if ordinary else 'research_pass21')
    rows=json.loads((folder/('baseline.json' if ordinary else 'files.json')).read_text())['rows']
    row=next(r for r in rows if r['image']==name and (not ordinary or r['preset']=='high_quality'))
    data=(folder/row['file']).read_bytes()
    assert sha(data)==row['stream_sha256'] and len(data)==row['bytes']
    return data,row
