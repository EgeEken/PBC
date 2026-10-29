import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('command',choices=('decode','inspect','benchdecode'));p.add_argument('input',type=Path);p.add_argument('output',nargs='?',type=Path);p.add_argument('--force',action='store_true');p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
sys.path.insert(0,str(ROOT/'research_pass26/artifacts/published'))
t=time.perf_counter()
from PBC3 import PBC3
imports=time.perf_counter()-t
data=a.input.read_bytes()
if a.command=='inspect':
    sys.path.insert(0,str(ROOT/'research_pass26/packaging'))
    from cli import checked_body
    result=checked_body(data)
elif a.command=='decode':
    t=time.perf_counter();image=PBC3.decompress(data).image;elapsed=time.perf_counter()-t
    image.save(a.output);result=dict(codec_seconds=elapsed)
else:
    PBC3.decompress(data);times=[]
    for _ in range(a.repeats):
        t=time.perf_counter();image=PBC3.decompress(data).image;times.append(time.perf_counter()-t)
    result=dict(seconds=times,pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
print(json.dumps(dict(import_seconds=imports,**result)))
