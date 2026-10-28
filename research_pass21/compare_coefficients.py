import importlib
import json
import lzma
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass21.common import *
from research_pass21 import zero_runs


def main():
    setup();ARTIFACTS.mkdir(exist_ok=True)
    alternative='--alternative' in sys.argv
    if alternative:
        sparse=importlib.import_module('research_pass21.occupancy')
        previous=json.loads((HERE/'coefficients.json').read_text())
        rows=previous['rows'];meta_before=previous['provenance']
        variants=('occupancy',)
    else:
        sparse=zero_runs;rows=[];meta_before=meta();variants=('dense_extreme','dense_normal','zero_runs')
    warm=np.zeros((2,3,3,8,8),dtype=np.int16);warm[0,0,0,0,0]=-32768;warm[-1,-1,2,7,7]=32767
    warmed=sparse.pack(warm);assert np.array_equal(sparse.unpack(warmed,warm.size),warm.transpose(2,3,4,0,1).ravel())
    for name,step in INITIAL:
        q=coefficients(name,step)
        expected=q.transpose(2,3,4,0,1).reshape(-1)
        def evaluate(variant,rep):
            start=time.perf_counter()
            if variant.startswith('dense'):raw=q.transpose(2,3,4,0,1).astype('<i2',copy=False).tobytes()
            else:raw=sparse.pack(q)
            packing=time.perf_counter()-start
            preset=lzma.PRESET_EXTREME if variant=='dense_extreme' else 0
            start=time.perf_counter();payload=lzma.compress(raw,format=lzma.FORMAT_XZ,preset=preset);compression=time.perf_counter()-start
            start=time.perf_counter()
            unpacked=lzma.decompress(payload,format=lzma.FORMAT_XZ) if variant.startswith('dense') else sparse.decompress(payload,q.size)
            decompression=time.perf_counter()-start
            start=time.perf_counter()
            recovered=np.frombuffer(unpacked,dtype='<i2') if variant.startswith('dense') else sparse.unpack(unpacked,q.size)
            unpacking=time.perf_counter()-start
            assert np.array_equal(recovered,expected)
            path=ARTIFACTS/(name+'.'+variant+'.xz')
            if rep==0:path.write_bytes(payload)
            elif rep>0:assert sha(payload)==sha(path.read_bytes())
            return dict(image=name,step=step,variant=variant,rep=rep,coefficient_count=q.size,nonzero_count=int(np.count_nonzero(q)),raw_bytes=len(raw),payload_bytes=len(payload),packing_seconds=packing,compression_seconds=compression,total_encode_seconds=packing+compression,decompression_seconds=decompression,unpacking_seconds=unpacking,total_decode_seconds=decompression+unpacking,payload_sha256=sha(payload),raw_sha256=sha(raw),process_peak_working_set=psutil.Process().memory_info().peak_wset)
        for variant in variants:evaluate(variant,-1)
        for rep in range(3):
            order=variants[rep%len(variants):]+variants[:rep%len(variants)]
            for variant in order:rows.append(evaluate(variant,rep))
        print(name,[(variant,[(k,statistics.median(r[k] for r in rows if r['image']==name and r['variant']==variant)) for k in ('raw_bytes','payload_bytes','total_encode_seconds','total_decode_seconds')]) for variant in variants],flush=True)
    summary=[]
    for name,step in INITIAL:
        for variant in dict.fromkeys(r['variant'] for r in rows):
            chosen=[r for r in rows if r['image']==name and r['variant']==variant]
            if not chosen:continue
            summary.append(dict(image=name,variant=variant,**{k:statistics.median(r[k] for r in chosen) for k in ('coefficient_count','nonzero_count','raw_bytes','payload_bytes','packing_seconds','compression_seconds','total_encode_seconds','decompression_seconds','unpacking_seconds','total_decode_seconds')}))
    write_json('coefficients.json',dict(provenance=meta_before,alternative_provenance=meta() if alternative else None,rows=rows,summary=summary,repetitions=3,timing='Three warmed rotated coefficient-stage repetitions. Packing includes direct order traversal/counting/header/output copies. Decompression includes bounded parsing for sparse formats; unpack is separately timed. Process peak working set is cumulative.'))


if __name__=='__main__':main()
