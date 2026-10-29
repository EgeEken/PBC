import argparse
import gc
import subprocess
import sys
from common import *
from research_pass25 import base,hybrid
from research_pass21 import sparse_hybrid
from research_pass22.decoder import decode as sparse_decode
from research_pass22.fixtures import small


def worker(name,variant):
    setup();tiny=small();PBC3.decompress(sparse_hybrid.split(tiny)[1]);sparse_decode(tiny)
    prep=PBC3.prepare(Image.new('RGB',(13,9),(100,130,170)),PBC3Config.quality())
    warm=hybrid.frame(base.encode(prep,128),[])[0];hybrid.decode(warm)
    if variant=='p21s':
        row=next(r for r in json.loads((ROOT/'research_pass21/files.json').read_text())['rows'] if r['image']==name)
        data=(ROOT/'research_pass21'/row['file']).read_bytes();decoder=sparse_decode
    else:
        row=next(r for r in json.loads((HERE/'screen.json').read_text())['rows'] if r['image']==name and r['preset']=='high_quality' and r['variant']==variant)
        data=(HERE/row['file']).read_bytes();decoder=(lambda data:PBC3.decompress(data).image) if variant=='ordinary' else hybrid.decode
    assert sha(data)==row['stream_sha256']
    gc.collect();before=psutil.Process().memory_info();output=decoder(data);after=psutil.Process().memory_info()
    pixels=output.tobytes()
    assert sha(pixels)==row['pixels_sha256']
    print(json.dumps(dict(image=name,variant=variant,baseline_rss=before.rss,baseline_peak=before.peak_wset,peak_rss=after.peak_wset,final_rss=after.rss,peak_above_baseline=max(0,after.peak_wset-before.rss),rgb_mse=row['rgb_mse'],bytes=row['bytes'])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs=2);args=parser.parse_args()
    if args.worker:worker(*args.worker);return
    setup();rows=[]
    for name in IMAGES:
        for variant in ('ordinary','hybrid'):
            rows.append(json.loads(subprocess.check_output([sys.executable,__file__,'--worker',name,variant],cwd=ROOT,text=True)))
        print('memory',name,flush=True)
    rows.append(json.loads(subprocess.check_output([sys.executable,__file__,'--worker','cat.jpg','p21s'],cwd=ROOT,text=True)))
    write('memory.json',dict(provenance=meta(),rows=rows,note='One fresh process per image/variant; same tiny ordinary/working-DCT/P21S warmups; one actual saved HQ decode. Windows peak_wset includes imports/Numba/native allocator; pre-call RSS/peak recorded. Peak read before output hashing. No source scoring or large alternate output retained. P21S Cat is a distinct much-better-quality research point, for memory scale only, not matched-RD performance evidence.'))


if __name__=='__main__':main()
