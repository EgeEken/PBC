import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
THREADS={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',choices=('release','dev_release_rng','dev_current_rng'),required=True)
    source=parser.parse_args().source
    os.environ.update(THREADS)
    sys.path.insert(0,str(ROOT))
    if source=='release':
        sys.path.insert(0,str(ROOT/'research_pass1/release'))
    import numpy as np
    from PIL import Image,ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config,BitReader
    if hasattr(os,'sched_getaffinity'):
        cpu=min(os.sched_getaffinity(0))
        os.sched_setaffinity(0,{cpu})
    else:
        import psutil
        cpu=psutil.Process().cpu_affinity()[0]
        psutil.Process().cpu_affinity([cpu])
    model=ROOT/'patch_policy.npz'
    sha=lambda data:hashlib.sha256(data).hexdigest()
    presets=('compression','balanced','quality','high_quality')
    for line in sys.stdin:
        job=json.loads(line)
        try:
            if job.get('op')=='quit':
                print('{}',flush=True); break
            if job.get('op')=='metadata':
                result=dict(source=source,cpu=cpu,threads=THREADS,python=sys.version,
                    versions={n:importlib.metadata.version(n) for n in ('numpy','numba','Pillow')},
                    presets={p:vars(getattr(PBC3Config,p)()) for p in presets},
                    rng='numpy.random.default_rng / '+type(np.random.default_rng(2003).bit_generator).__name__ if source in ('release','dev_release_rng') else 'pbc3_ops.PBC3Rng / PCG32, modulo integers',
                    model_sha256=sha(model.read_bytes()),
                    modules={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.copy().items()
                        if getattr(m,'__file__',None) and (n=='PBC3' or n=='learned_filler' or n.startswith('pbc3_'))})
            else:
                path=ROOT/job.get('dataset','hpt_data')/job['image']
                with Image.open(path) as opened:
                    image=ImageOps.exif_transpose(opened).convert('RGB')
                config=getattr(PBC3Config,job['preset'])(learned_filler_model_path=str(model))
                assert config.compute_final_mse
                started=time.perf_counter()
                encoded=PBC3.compress(image,config=config,**({'rng_mode':'release' if source=='dev_release_rng' else 'current'} if source!='release' else {}))
                encode_seconds=time.perf_counter()-started
                started=time.perf_counter()
                decoded=PBC3.decompress(encoded.data)
                decode_seconds=time.perf_counter()-started
                assert decoded.image.size==image.size
                pixels=np.asarray(decoded.image.convert('RGB'))
                error=np.subtract(np.asarray(image),pixels,dtype=np.float32)
                _,body=PBC3._open_body(encoded.data)
                reader=BitReader(body)
                header=PBC3._read_header(reader)
                trajectory=[]
                for _ in range(header[10]):
                    c,x,y,w,h,cell,values,bits=PBC3._read_patch(reader,header[7],header[8])
                    trajectory.append([c,x,y,w,h,cell,bits,sha(values.tobytes())])
                result=dict(source=source,dataset=job.get('dataset','hpt_data'),image=job['image'],preset=job['preset'],trial=job.get('trial',-1),
                    width=image.width,height=image.height,working_width=header[3],working_height=header[4],
                    input_sha256=sha(path.read_bytes()),mse=float(np.mean(error**2,dtype=np.float64)),internal_mse=encoded.mse,
                    final_bytes=len(encoded.data),bpp=8*len(encoded.data)/(image.width*image.height),
                    patches=header[10],initialization_patches=header[6],iterative_patches=header[10]-header[6],
                    stream_sha256=sha(encoded.data),pixels_sha256=sha(pixels.tobytes()),
                    trajectory_sha256=sha(json.dumps(trajectory).encode()),trajectory=trajectory,
                    encode_seconds=encode_seconds,decode_seconds=decode_seconds,
                    preview_equal=bool(np.array_equal(np.asarray(encoded.image.convert('RGB')),pixels)))
                assert result['preview_equal'] and np.isclose(result['mse'],result['internal_mse'],rtol=1e-6)
            print(json.dumps(result),flush=True)
        except Exception:
            import traceback
            print(json.dumps(dict(error=traceback.format_exc())),flush=True)


if __name__=='__main__':
    main()
