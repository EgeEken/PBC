import argparse
import gc
import json
import subprocess
import sys
from common import *
from research_pass22 import decoder,fixtures


def worker(name,label):
    setup()
    warm=fixtures.small()
    reference.decode(warm);decoder.decode(warm)
    data,record=saved(name)
    gc.collect();before=psutil.Process().memory_info()
    output=(reference.decode if label=='reference' else decoder.decode)(data)
    after=psutil.Process().memory_info()
    result=dict(image=name,variant=label,baseline_rss=before.rss,baseline_peak=before.peak_wset,final_rss=after.rss,peak_rss=after.peak_wset,peak_above_baseline_rss=max(0,after.peak_wset-before.rss),peak_increase=max(0,after.peak_wset-before.peak_wset))
    assert sha(output.tobytes())==record['pixels_sha256']
    print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs=2);args=parser.parse_args()
    if args.worker:worker(*args.worker);return
    setup();rows=[]
    for name in CASES:
        for rep in range(2):
            for label in ('reference','optimized'):
                result=json.loads(subprocess.check_output([sys.executable,__file__,'--worker',name,label],cwd=ROOT,text=True));result['rep']=rep;rows.append(result)
        print('memory',name,flush=True)
    write('memory.json',dict(provenance=meta(),rows=rows,note='Two fresh processes per image/variant. Identical tiny decoder-only fixture warms reference/optimized/Numba/SciPy; target decode once. Windows peak_wset includes interpreter/import/warmup, with pre-call RSS and pre-call peak recorded. Peak captured before hash/scoring. No baseline-output arrays or profiles coexist. Native allocator retention included; not a theoretical minimum.'))


if __name__=='__main__':main()
