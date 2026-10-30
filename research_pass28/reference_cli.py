import argparse,hashlib,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command',choices=('encode','benchmark','benchdecode','decode'));p.add_argument('input',type=Path);p.add_argument('output',nargs='?',type=Path);p.add_argument('--preset',default='quality');p.add_argument('--repeats',type=int,default=3);p.add_argument('--force',action='store_true');a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_pass27/artifacts/published'))
t=time.perf_counter()
from PBC3 import PBC3
from pbc3_types import PBC3Config
from PIL import Image,ImageOps
imports=time.perf_counter()-t
if a.command in ('encode','benchmark'):
    image=ImageOps.exif_transpose(Image.open(a.input)).copy();cfg=getattr(PBC3Config,a.preset)()
    t=time.perf_counter();r=PBC3.compress(image,config=cfg);first=time.perf_counter()-t
    if a.command=='encode':
        with a.output.open('wb' if a.force else 'xb') as f:f.write(r.data)
        result=dict(codec_seconds=first,bytes=len(r.data))
    else:
        rows=[]
        for _ in range(a.repeats):
            t=time.perf_counter();n=PBC3.compress(image,config=cfg);seconds=time.perf_counter()-t;assert n.data==r.data;rows.append(seconds)
        result=dict(first_encode_seconds=first,seconds=rows,stream_sha256=hashlib.sha256(r.data).hexdigest())
else:
    data=a.input.read_bytes()
    if a.command=='benchdecode':
        PBC3.decompress(data);times=[]
        for _ in range(a.repeats):
            t=time.perf_counter();im=PBC3.decompress(data).image;times.append(time.perf_counter()-t)
        result=dict(seconds=times,pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest())
    else:
        t=time.perf_counter();im=PBC3.decompress(data).image;seconds=time.perf_counter()-t;im.save(a.output);result=dict(codec_seconds=seconds)
print(json.dumps(dict(import_seconds=imports,**result)))
