import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
THREADS={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS')}
os.environ.update(THREADS)


def worker(variant):
    source=HERE/'baseline' if variant=='baseline' else ROOT
    sys.path.insert(0,str(source))
    sys.path.insert(0,str(HERE))
    import numpy as np
    import psutil
    from PIL import Image
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    if variant=='candidate' and '--integrated' not in sys.argv:
        import learned_filler
        from check_features import candidate_function
        learned_filler.extract_features=candidate_function()
    process=psutil.Process(); process.cpu_affinity([12])
    jobs=[]
    rng=np.random.default_rng(1592)
    for h,w in ((32,40),(97,113)):
        for channels in (3,4):
            array=rng.integers(0,256,size=(h,w,channels),dtype=np.uint8)
            image=Image.fromarray(array)
            for preset in ('compression','balanced','quality','high_quality'):
                jobs.append((f'{w}x{h}/{channels}/{preset}',image,preset))
    def encode(job):
        name,image,preset=job
        config=getattr(PBC3Config,preset)()
        config.learned_filler_model_path=str(ROOT/'patch_policy.npz')
        encoded=PBC3.compress(image,config)
        decoded=PBC3.decompress(encoded.data).image
        return {'name':name,'stream':base64.b64encode(encoded.data).decode(),'pixels_sha256':hashlib.sha256(decoded.tobytes()).hexdigest(),'mode':decoded.mode,'size':decoded.size,'internal_mse':encoded.mse}
    serial=[encode(job) for job in jobs]
    warm_memory=process.memory_info()._asdict()
    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel=list(executor.map(encode,jobs))
    assert serial==parallel,'concurrent encoding differs'
    result={'variant':variant,'cases':serial,'concurrency_identical':True,'warm_memory':warm_memory,'after_memory':process.memory_info()._asdict()}
    print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--worker',choices=('baseline','candidate')); parser.add_argument('--integrated',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.worker); return
    results={}
    for name in ('baseline','candidate'):
        results[name]=json.loads(subprocess.check_output([sys.executable,str(__file__),'--worker',name]+(['--integrated'] if args.integrated else []),env={**os.environ,**THREADS},cwd=ROOT,text=True))
    assert results['baseline']['cases']==results['candidate']['cases']
    results['all_edge_streams_pixels_metrics_identical']=True
    (HERE/('integrated_edges.json' if args.integrated else 'edges.json')).write_text(json.dumps(results,indent=2)+'\n')
    print('16 RGB/RGBA cases match, serial and two-thread concurrent, in both workers')

if __name__=='__main__': main()
