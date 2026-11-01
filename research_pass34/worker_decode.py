from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import psutil
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'research_pass33'))
sys.path.insert(1,str(HERE.parent))
import pbc33_codec as p33
import pbc34_codec as p34

def sha(data): return hashlib.sha256(data).hexdigest()
def p33_account(data):
    import math,lzma
    _,_,w,h,_,_,_,q,_,n,rawlen,xz=p33.HEADER.unpack_from(data)
    dec=lzma.LZMADecompressor(format=lzma.FORMAT_XZ,memlimit=64*1024*1024);raw=dec.decompress(data[p33.HEADER.size:],max_length=rawlen+1)
    ml=math.ceil(n*3/8);modes=p33.unpack_modes(raw[:ml],n);params=raw[ml:];cursor=dc=detail=0
    for mode in modes:
        if mode==1:dc+=3;cursor+=3
        elif mode==4:detail+=48;cursor+=48
    if cursor!=len(params):raise ValueError('control accounting mismatch')
    return {'header_bytes':p33.HEADER.size,'mode_bytes':ml,'dc_bytes':dc,'detail_grid_bytes':detail,'raw_parameter_bytes':len(params),'raw_payload_bytes':len(raw),'xz_payload_bytes':xz,'xz_vs_raw_delta_bytes':xz-len(raw),'cell_mode_counts':{str(int(m)):int(np.count_nonzero(modes==m)) for m in (0,1,4)}}

rows=[];proc=psutil.Process();manifest=json.loads(Path(sys.argv[1]).read_text())
for item in manifest['files']:
    base=np.load(item['base_npy'],allow_pickle=False);data=Path(item['stream']).read_bytes();profile={}
    if item['type']=='p34':decoded,modes,params=p34.decode(data,base,profile);rec=p34.accounting(data)
    else:decoded,modes,params=p33.decode(data,base,profile);rec=p33_account(data)
    digest=sha(np.ascontiguousarray(decoded).tobytes())
    if digest!=item['expected_pixels_sha256']:raise ValueError('pixel hash mismatch: '+item['key'])
    modes_hash=sha(modes.tobytes());params_hash=sha(params)
    if modes_hash!=item['expected_modes_sha256'] or params_hash!=item['expected_params_sha256']:raise ValueError('mode/parameter hash mismatch: '+item['key'])
    out=HERE/'outputs'/'reconstructions'/Path(item['stream']).parent.name;out.mkdir(parents=True,exist_ok=True)
    np.save(out/(item['key'].replace('|','_')+'.npy'),decoded,allow_pickle=False)
    rec.update({'key':item['key'],'pixels_sha256':digest,'modes_sha256':modes_hash,'params_sha256':params_hash,'profile':profile,
        'peak_working_set_bytes':proc.memory_info().peak_wset,'fresh_process_verified':True})
    rows.append(rec)
print(json.dumps({'rows':rows}))
