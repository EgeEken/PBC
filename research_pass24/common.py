import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass17.analyze_residuals import setup,provenance,load,sha,np,PBC3,psutil
from research_pass2.stream_tools import parse,serialize
import pbc3_ops as ops

HERE=Path(__file__).resolve().parent
ARTIFACTS=HERE/'artifacts'
IMAGES=('googlewiki.png','mario64.png','cat.jpg','flower.jpg')
PRESETS=('quality','high_quality')


def write(name,value):
    (HERE/name).write_text(json.dumps(value,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')


def meta():
    result=provenance();result.pop('timing',None)
    result.update(start_commit='5d1b3a1f0db036dfef0dd9d51c6898802f7d48d8',source_sha256_pass24={p.name:sha(p.read_bytes()) for p in HERE.glob('*.py')},plan_sha256=sha((HERE/'PLAN.md').read_bytes()),parser_sha256=sha((ROOT/'research_pass2/stream_tools.py').read_bytes()))
    return result


def saved(name,preset):
    row=next(r for r in json.loads((ROOT/'research_pass17/baseline.json').read_text())['rows'] if r['image']==name and r['preset']==preset)
    data=(ROOT/'research_pass17'/row['file']).read_bytes()
    assert sha(data)==row['stream_sha256'] and len(data)==row['bytes']
    return data,row


def rgb_mse(source,output):
    error=np.subtract(source,output,dtype=np.float32)
    return float(np.mean(error**2,dtype=np.float64))


def descriptions(header,patches,accounting):
    seen=np.zeros((header[4],header[3],header[6]),dtype=np.bool_)
    rows=[]
    for index in range(len(patches)-1,-1,-1):
        patch=patches[index];c,x,y,w,h=(patch[k] for k in ('channel','x','y','w','h'))
        overlap=float(np.mean(seen[y:y+h,x:x+w,c]));seen[y:y+h,x:x+w,c]=True
        row=dict(accounting['patches'][index],x=x,y=y,grid_size=list(patch['indices'].shape[::-1]),effective_bit_depth=patch['bitcount'],later_same_component_overlap=overlap,iterative_position=index-header[6],iterative_fraction=(index-header[6])/max(1,len(patches)-header[6]-1))
        rows.append(row)
    return rows[::-1]


def select(rows,initial):
    iterative=rows[initial:];n=len(iterative)
    if not n:return []
    chosen=set(initial+max(0,min(n-1,k)) for k in (0,1,round((n-1)*.25),round((n-1)*.5),round((n-1)*.75),n-2,n-1))
    chosen.update(r['patch'] for r in sorted(iterative,key=lambda r:(-r['total_bits'],r['patch']))[:2])
    chosen.update(r['patch'] for r in sorted(iterative,key=lambda r:(-r['later_same_component_overlap'],r['patch']))[:2])
    for row in iterative:
        if len(chosen)>=min(10,n):break
        chosen.add(row['patch'])
    return sorted(chosen)


def evaluate(header,patches,source):
    start=time.perf_counter();data=serialize(header,patches);packed=time.perf_counter()
    output=np.asarray(PBC3.decompress(data).image.convert('RGB'));decoded=time.perf_counter()
    mse=rgb_mse(source,output);scored=time.perf_counter()
    return data,output,dict(bytes=len(data),rgb_mse=mse,repack_seconds=packed-start,decode_seconds=decoded-packed,score_seconds=scored-decoded,evaluation_seconds=scored-start)
