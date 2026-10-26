import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
THREADS = {k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}


def request(process,job):
    process.stdin.write(json.dumps(job)+'\n')
    process.stdin.flush()
    row=json.loads(process.stdout.readline())
    if 'error' in row:
        raise RuntimeError(row['error'])
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',choices=('baseline','integrated','init','iter'),required=True)
    args=parser.parse_args()
    os.environ.update(THREADS)
    sys.path.insert(0,str(ROOT))
    if args.source=='baseline':
        sys.path.insert(0,str(HERE/'baseline'))
    import numpy as np
    from PIL import Image,ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config,BitReader
    import pbc3_ops as ops
    if hasattr(os,'sched_getaffinity'):
        cpu=min(os.sched_getaffinity(0))
        os.sched_setaffinity(0,{cpu})
    else:
        import psutil
        cpu=psutil.Process().cpu_affinity()[0]
        psutil.Process().cpu_affinity([cpu])
    sha=lambda data:hashlib.sha256(data).hexdigest()
    def encode(job):
        if 'synthetic' in job:
            spec=job['synthetic']
            a=np.random.default_rng(spec.get('seed',2003)).integers(0,256,tuple(spec['shape']),dtype=np.uint8)
            if spec.get('opaque'):
                a[:,:,-1]=255
            image=Image.fromarray(a)
            name='synthetic'
        else:
            name=job['image']
            with Image.open(ROOT/job.get('dataset','hpt_data')/name) as opened:
                image=ImageOps.exif_transpose(opened).convert(job.get('mode','RGB'))
        cfg=getattr(PBC3Config,job.get('preset','balanced'))(learned_filler_model_path=str(ROOT/'patch_policy.npz'),**job.get('overrides',{}))
        stats={}
        if args.source in ('init','iter'):
            from refinement import enabled
            scope=enabled(args.source,stats)
        else:
            from contextlib import nullcontext
            scope=nullcontext()
        with scope:
            started=time.perf_counter()
            result=PBC3.compress(image,config=cfg)
            encode_seconds=time.perf_counter()-started
        started=time.perf_counter()
        decoded=PBC3.decompress(result.data)
        decode_seconds=time.perf_counter()-started
        pixels=np.asarray(decoded.image)
        difference=np.subtract(np.asarray(image.convert('RGB')),np.asarray(decoded.image.convert('RGB')),dtype=np.float32)
        _,body=PBC3._open_body(result.data)
        reader=BitReader(body)
        header=PBC3._read_header(reader)
        patches=[]
        for _ in range(header[10]):
            c,x,y,w,h,cell,values,bits=PBC3._read_patch(reader,header[7],header[8])
            patches.append([c,x,y,w,h,cell,bits,sha(values.tobytes())])
        row=dict(image=name,preset=job.get('preset','balanced'),source=args.source,trial=job.get('trial',-1),
            mode=image.mode,width=image.width,height=image.height,mse=float(np.mean(difference**2,dtype=np.float64)),
            internal_mse=result.mse,final_bytes=len(result.data),raw_bytes=len(body)+6,bpp=len(result.data)*8/(image.width*image.height),
            patches=header[10],stream_sha256=sha(result.data),pixels_sha256=sha(pixels.tobytes()),
            trajectory_sha256=sha(json.dumps(patches).encode()),encode_seconds=encode_seconds,decode_seconds=decode_seconds,
            preview_equal=bool(np.array_equal(np.asarray(result.image),pixels)),stats=stats)
        if job.get('payload'):
            row['payload']=base64.b64encode(result.data).decode()
        if job.get('trajectory'):
            row['trajectory']=patches
        return row
    for line in sys.stdin:
        job=json.loads(line)
        try:
            op=job.get('op','encode')
            if op=='quit':
                print('{}',flush=True)
                break
            if op=='metadata':
                import importlib.metadata
                result=dict(source=args.source,pid=os.getpid(),cpu=cpu,threads=THREADS,python=sys.version,
                    versions={n:importlib.metadata.version(n) for n in ('numpy','numba','Pillow')},
                    modules={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.copy().items()
                        if getattr(m,'__file__',None) and (n=='PBC3' or n.startswith('pbc3_') or n=='learned_filler')})
            elif op=='decode':
                decoded=PBC3.decompress(base64.b64decode(job['payload']))
                result=dict(pixels_sha256=sha(np.asarray(decoded.image).tobytes()))
            elif op=='decode_outcome':
                try:
                    PBC3.decompress(base64.b64decode(job['payload']))
                    result=dict(accepted=True)
                except Exception as error:
                    result=dict(accepted=False,exception=type(error).__name__)
            elif op=='concurrent':
                sequential=[encode(x) for x in job['cases']]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    overlapping=list(pool.map(encode,job['cases']*2))
                assert all(r['stream_sha256']==sequential[i%len(sequential)]['stream_sha256'] and r['pixels_sha256']==sequential[i%len(sequential)]['pixels_sha256'] for i,r in enumerate(overlapping))
                result=dict(cases=len(overlapping),exact=True)
            else:
                result=encode(job)
            print(json.dumps(result),flush=True)
        except Exception:
            import traceback
            print(json.dumps(dict(error=traceback.format_exc())),flush=True)


if __name__=='__main__':
    main()
