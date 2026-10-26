import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
THREADS = {k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
MODES = ('baseline','grid_skip','search','all')
PRESETS = ('compression','balanced','quality','high_quality')
INPUTS = ('googlewiki.png','flower.jpg')


def validate(spec):
    if not isinstance(spec,dict):
        raise ValueError('research must be an object')
    if set(spec)-{'images','presets','variants','repeats'}:
        raise ValueError('Unknown research fields')
    for key,allowed,limit in (('images',INPUTS,2),('presets',PRESETS,2),('variants',MODES,4)):
        value=spec.get(key)
        if not isinstance(value,list) or not 1<=len(value)<=limit or len(set(value))!=len(value) or any(v not in allowed for v in value):
            raise ValueError('Invalid '+key)
    if type(spec.get('repeats')) is not int or not 1<=spec['repeats']<=3:
        raise ValueError('repeats must be 1..3')
    if 'baseline' not in spec['variants']:
        raise ValueError('baseline is required')
    return spec


def digest(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def enabled(mode):
    from PBC3 import PBC3
    import pbc3_stream as stream
    import hf_candidate_grid as grid
    from hf_candidate_search import SearchReuse
    descriptor=PBC3.__dict__['_read_grid']
    old_read,old_write=stream.read_grid,stream.write_grid
    grid.GRID_IO_MODE='numpy'
    try:
        if mode in ('grid_skip','all'):
            stream.read_grid,stream.write_grid=grid.read_grid,grid.write_grid
            PBC3._read_grid=staticmethod(grid.read_grid)
        yield
    finally:
        stream.read_grid,stream.write_grid=old_read,old_write
        PBC3._read_grid=descriptor


def run(spec):
    os.environ.update(THREADS)
    sys.path.insert(0,str(ROOT))
    import numpy as np
    from PIL import Image, ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config,BitReader
    from hf_candidate_search import SearchReuse
    if hasattr(os,'sched_getaffinity'):
        affinity=sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0,{affinity[0]})
    else:
        import psutil
        affinity=psutil.Process().cpu_affinity()
        psutil.Process().cpu_affinity([affinity[0]])
    manifest=json.loads((HERE/'hf_runtime_manifest.json').read_text())
    assert all(digest((ROOT/p).read_text(encoding='utf-8').encode())==value for p,value in manifest['source_lf'].items())
    assert digest((ROOT/'patch_policy.npz').read_bytes())==manifest['model_sha256']
    assert all(digest((ROOT/'hpt_data'/name).read_bytes())==value for name,value in manifest['inputs'].items())
    model=ROOT/'patch_policy.npz'
    metadata=dict(label=os.environ.get('PBC_RESEARCH_HOST_LABEL','UNSPECIFIED_HOST'),spec=spec,
        python=sys.version,platform=platform.platform(),cpu=platform.processor(),affinity=[affinity[0]],
        versions={n:importlib.metadata.version(n) for n in ('numpy','numba','Pillow')},
        threads=THREADS,manifest=manifest,warmup='one complete encode/decode per mode/preset; input googlewiki; all measured modes share this worker',
        module_paths={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.copy().items() if getattr(m,'__file__',None) and (n=='PBC3' or n.startswith('pbc3_') or n=='learned_filler')})
    def evaluate(name,preset,mode,trial):
        with Image.open(ROOT/'hpt_data'/name) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
        cfg=getattr(PBC3Config,preset)(learned_filler_model_path=str(model),compute_final_mse=mode not in ('grid_skip','all'))
        with enabled(mode):
            started=time.perf_counter()
            if mode in ('search','all'):
                state=SearchReuse(incremental=True)
                with state.enabled():
                    result=PBC3.compress(image,config=cfg)
            else:
                result=PBC3.compress(image,config=cfg)
            encode=time.perf_counter()-started
            started=time.perf_counter()
            decoded=PBC3.decompress(result.data)
            decode=time.perf_counter()-started
        assert decoded.image.size==image.size
        pixels=np.asarray(decoded.image.convert('RGB'))
        error=np.subtract(np.asarray(image),pixels,dtype=np.float32)
        _,body=PBC3._open_body(result.data)
        header=PBC3._read_header(BitReader(body))
        return dict(image=name,preset=preset,variant=mode,trial=trial,
            config=vars(cfg),width=image.width,height=image.height,working_width=header[3],working_height=header[4],
            initialization_patches=header[6],iterative_patches=header[10]-header[6],
            final_bytes=len(result.data),raw_bytes=len(body)+6,bpp=8*len(result.data)/(image.width*image.height),
            mse=float(np.mean(error**2,dtype=np.float64)),internal_mse=result.mse,
            encode_seconds=encode,decode_seconds=decode,stream_sha256=digest(result.data),pixels_sha256=digest(pixels.tobytes()),
            input_sha256=digest((ROOT/'hpt_data'/name).read_bytes()))
    metadata['warmups']=[evaluate('googlewiki.png',p,v,-1) for p in spec['presets'] for v in spec['variants']]
    rows=[]
    for trial in range(spec['repeats']):
        for i,name in enumerate(spec['images']):
            for p,preset in enumerate(spec['presets']):
                order=spec['variants']
                offset=(trial+i+p)%len(order)
                for mode in order[offset:]+order[:offset]:
                    rows.append(evaluate(name,preset,mode,trial))
    for row in rows:
        baseline=next(r for r in rows if (r['image'],r['preset'],r['trial'],r['variant'])==(row['image'],row['preset'],row['trial'],'baseline'))
        assert all(row[k]==baseline[k] for k in ('stream_sha256','pixels_sha256','mse','bpp','initialization_patches','iterative_patches'))
    return dict(metadata=metadata,rows=rows,exact=True,measurement='Full compress/decompress wall time, including search state setup/cleanup; excludes loading, independent metrics, IPC and HTTP. No DB writes.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec-json',required=True)
    args=parser.parse_args()
    print(json.dumps(run(validate(json.loads(args.spec_json)))))
