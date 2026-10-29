import os
import sys
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
PASS=Path(__file__).resolve().parent
ROOT=PASS.parent
SOURCE=PASS/'artifacts/published'
sys.path.insert(0,str(SOURCE))
import hashlib
import json
import numpy as np
import numba
import psutil
from PIL import Image, ImageOps
from PBC3 import PBC3
from pbc3_types import PBC3Config
for row in json.loads((PASS/'baseline.json').read_text())['files']:
    assert hashlib.sha256((SOURCE/row['file']).read_bytes()).hexdigest()==row['published_sha256'],row['file']
assert Path(sys.modules['PBC3'].__file__).resolve().parent==SOURCE
psutil.Process().cpu_affinity([12])
numba.set_num_threads(1)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def load(name):
    return ImageOps.exif_transpose(Image.open(ROOT/'hpt_data'/name)).convert('RGB')

def mse(a,b):
    return float(np.mean((np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))**2))

def save(name,value):
    (PASS/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
