import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
PASS=ROOT/'research_pass27'
assert importlib.util.find_spec('numba') is None
assert importlib.util.find_spec('llvmlite') is None
sys.path.insert(0,str(ROOT/'research_pass28/artifacts/release-source'))
from pbc3_decoder import decode,inspect,read_file,MAX_FILE
import numpy as np
manifest=json.loads((PASS/'decoder/corpus.json').read_text())
rows=[]
for row in manifest['valid']:
    data=read_file(row['file']);image=decode(data)
    assert hashlib.sha256(image.tobytes()).hexdigest()==row['pixel_sha256'],row['name']
    assert list(image.size)==row['size'] and image.mode==row['mode']
    assert decode(data).tobytes()==image.tobytes()
    for limit,expected in row['partial'].items(): assert hashlib.sha256(decode(data,int(limit)).tobytes()).hexdigest()==expected
    rows.append(row['name'])
for path in manifest['invalid']:
    try: decode(read_file(path))
    except (ValueError,EOFError,IndexError,KeyError,OverflowError): pass
    else: raise AssertionError('accepted invalid input: '+path)
large=PASS/'artifacts/oversized.pbc'
with large.open('wb') as f: f.truncate(MAX_FILE+1)
try: read_file(large)
except ValueError: pass
else: raise AssertionError('oversized file accepted')
small=Path(manifest['valid'][0]['file']).read_bytes()
with ThreadPoolExecutor(4) as pool: outputs=list(pool.map(lambda _:decode(small).tobytes(),range(16)))
assert all(p==outputs[0] for p in outputs)
for name in ('numba','llvmlite','PBC3','pbc3_ops','pbc3_heads','pbc3_features','learned_filler'): assert name not in sys.modules,name
result=dict(valid=rows,invalid_rejected=len(manifest['invalid'])+1,partial_checks=len(rows)*2,concurrent_calls=16,numba_unavailable=True,llvmlite_unavailable=True,encoder_modules_loaded=False)
(ROOT/'research_pass28/artifacts/compatibility.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(result)
