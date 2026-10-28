import ast
import hashlib
import json
from pathlib import Path

from check_features import candidate_source

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
screen=json.loads((HERE/'screen.json').read_text())
full=json.loads((HERE/'full.json').read_text())
edges=json.loads((HERE/'edges.json').read_text())
assert screen['all_streams_identical'] and screen['summary']['overall']['reduction_percent']>=3
assert all(r['reduction_percent']>0 for r in screen['summary'].values())
assert full['all_streams_identical'] and len(full['diagnostics'])==len(full['cases'])==100
assert edges['all_edge_streams_pixels_metrics_identical']
path=ROOT/'pbc3_features.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()==screen['source_sha256']['pbc3_features.py']
source=path.read_text()
module=ast.parse(source)
function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='extract_features')
original='\n'.join(source.splitlines()[function.lineno-1:function.end_lineno])
optimized=candidate_source().rstrip().replace('integer_order_stats(', '_integer_order_stats(')
helper=(HERE/'feature_candidate.py').read_text().replace('import numpy as np\n','').replace('from numba import njit\n','').strip().replace('integer_order_stats(', '_integer_order_stats(')
source=source.replace('import numpy as np','import numpy as np\nfrom numba import njit',1)
source=source.replace(original,helper+'\n\n\n'+optimized)
path.write_text(source,encoding='utf-8',newline='\n')
print('Integrated validated histogram candidate:',hashlib.sha256(path.read_bytes()).hexdigest())
