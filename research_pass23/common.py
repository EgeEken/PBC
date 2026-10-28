import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass17.analyze_residuals import setup,provenance,load,sha,np,PBC3,PBC3Config,Image,psutil

HERE=Path(__file__).resolve().parent
ARTIFACTS=HERE/'artifacts'
IMAGES=('googlewiki.png','mario64.png','cat.jpg','flower.jpg')
SETTINGS={'ordinary':(0.,0,0.),'conservative':(.5,1,8.),'stronger':(1.,2,16.)}


def write(name,value):
    (HERE/name).write_text(json.dumps(value,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')


def meta():
    result=provenance();result.pop('timing',None)
    result.update(start_commit='0ca503759f6fcc507489dd999a5fe4c848ae2eb7',source_sha256_pass23={p.name:sha(p.read_bytes()) for p in HERE.glob('*.py')},plan_sha256=sha((HERE/'PLAN.md').read_bytes()),settings=SETTINGS)
    return result
