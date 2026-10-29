import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
PASS=ROOT/'research_pass27'
CANDIDATE=PASS/'artifacts/candidate'
PUBLISHED=ROOT/'research_pass26/artifacts/published'
import hashlib
import json
import numpy as np
import psutil
from PIL import Image,ImageOps
psutil.Process().cpu_affinity([12])

def sha(data): return hashlib.sha256(data).hexdigest()
def save(name,data): (PASS/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
def load(name): return ImageOps.exif_transpose(Image.open(ROOT/'hpt_data'/name)).convert('RGB')
def mse(a,b): return float(np.mean((np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))**2))
