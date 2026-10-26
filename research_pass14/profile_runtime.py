import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

THREADS={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
os.environ.update(THREADS)
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import numba
from PIL import Image, ImageOps
numba.set_num_threads(1)
try:
    import psutil
    allowed=psutil.Process().cpu_affinity()
    cpu=min(allowed)
    psutil.Process().cpu_affinity([cpu])
except Exception:
    cpu=None
from PBC3 import PBC3
from pbc3_types import BitReader, PBC3Config

IMAGES=('cat.jpg','googlewiki.png','monet.jpg','mario64.png','eiffel.png','seagull.jpg')
PRESETS=('quality','high_quality')
SAMPLES=3
sha=lambda b:hashlib.sha256(b).hexdigest()

def load(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')

def timing(values):
    return dict(samples=len(values),mean_seconds=statistics.mean(values),median_seconds=statistics.median(values),min_seconds=min(values),max_seconds=max(values),cv_percent=100*statistics.pstdev(values)/statistics.mean(values))

def main():
    rows=[]
    traces=[]
    for name in IMAGES:
        source=load(ROOT/'hpt_data'/name)
        for preset in PRESETS:
            cfg=getattr(PBC3Config,preset)()
            warm=PBC3.compress(source,config=cfg)
            PBC3.decompress(warm.data)
            profiled=PBC3.compress(source,config=cfg,trace=True)
            decoded=PBC3.decompress(profiled.data,trace=True)
            assert profiled.data==warm.data
            assert np.array_equal(np.asarray(decoded.image.convert('RGB')),np.asarray(PBC3.decompress(warm.data).image.convert('RGB')))
            encode=[]; decode=[]
            for _ in range(SAMPLES):
                start=time.perf_counter(); result=PBC3.compress(source,config=cfg); encode.append(time.perf_counter()-start)
                start=time.perf_counter(); PBC3.decompress(result.data); decode.append(time.perf_counter()-start)
            enc_trace=profiled.timings
            dec_trace=decoded.timings
            body=PBC3._open_body(warm.data)[1]
            br=BitReader(body)
            patches=PBC3._read_header(br)[10]
            row=dict(image=name,preset=preset,input_sha256=sha((ROOT/'hpt_data'/name).read_bytes()),stream_sha256=sha(warm.data),file_bytes=len(warm.data),patches=patches,
                encode=timing(encode),decode=timing(decode),encode_trace={k:v for k,v in enc_trace['summary'].items()},decode_trace={k:v for k,v in dec_trace['summary'].items()},
                encode_total_traced_seconds=enc_trace['total_ns']/1e9,decode_total_traced_seconds=dec_trace['total_ns']/1e9,
                encode_unattributed_seconds=enc_trace['unattributed_ns']/1e9,decode_unattributed_seconds=dec_trace['unattributed_ns']/1e9,
                encode_counters=enc_trace['counters'],decode_counters=dec_trace['counters'])
            rows.append(row)
            traces.append(dict(image=name,preset=preset,encode_events=enc_trace['events'],decode_events=dec_trace['events']))
            print(f"{name} {preset}: enc={row['encode']['median_seconds']:.4f}s dec={row['decode']['median_seconds']:.4f}s",flush=True)
    output=dict(schema='pbc-pass14-profile-v1',commit='8006f30debfc4a57289df06df21636aeb52baffd',baseline_id='pbc31_release_rng_baseline',python=sys.version,platform=sys.platform,numpy=np.__version__,numba=numba.__version__,pillow=Image.__version__,cpu_affinity=cpu,thread_env=THREADS,repeats=SAMPLES,warmups=1,rows=rows,traces=traces)
    (OUT/'profile_results.json').write_text(json.dumps(output,indent=2)+chr(10),encoding='utf-8')
if __name__=='__main__': main()
