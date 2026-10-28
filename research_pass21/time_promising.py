import json
import lzma
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass21.common import *
from research_pass21 import sparse_hybrid as codec


def main():
    setup();rows=[]
    measured=json.loads((HERE/'files.json').read_text())['rows']
    ordinary=json.loads((ROOT/'research_pass17/baseline.json').read_text())['rows']
    choices=(('googlewiki.png',128),('eiffel.png',32))
    variants=('ordinary','sparse','dense_extreme','dense_normal')
    def pack_normal(q):return lzma.compress(q.transpose(2,3,4,0,1).astype('<i2',copy=False).tobytes(),format=lzma.FORMAT_XZ,preset=0)
    for name,step in choices:
        image=load(name);prior,base,expected=frozen(name,step)
        record=next(r for r in measured if r['image']==name)
        ordinary_expected=next(r for r in ordinary if r['image']==name and r['preset']=='high_quality')
        def evaluate(variant,rep):
            start=time.perf_counter()
            if variant=='ordinary':data=PBC3.compress(image,config=PBC3Config.high_quality()).data
            elif variant=='sparse':data=codec.encode(image,step)
            elif variant=='dense_extreme':data=h17.encode(image,step)
            else:
                original=h17.pack_coefficients;h17.pack_coefficients=pack_normal
                try:data=h17.encode(image,step)
                finally:h17.pack_coefficients=original
            enc=time.perf_counter()-start
            start=time.perf_counter()
            decoded=PBC3.decompress(data) if variant=='ordinary' else codec.decode(data) if variant=='sparse' else h17.decode(data)
            dec=time.perf_counter()-start
            pixels=np.asarray(decoded.image) if variant=='ordinary' else decoded
            digest=sha(data)
            expected_file=ordinary_expected['stream_sha256'] if variant=='ordinary' else record['stream_sha256'] if variant=='sparse' else expected['stream_sha256'] if variant=='dense_extreme' else record['dense_normal_sha256']
            expected_pixels=ordinary_expected['pixels_sha256'] if variant=='ordinary' else expected['pixels_sha256']
            assert digest==expected_file and sha(pixels.tobytes())==expected_pixels
            return dict(image=name,step=step,variant=variant,rep=rep,bytes=len(data),encode_seconds=enc,decode_seconds=dec,stream_sha256=digest,process_peak_working_set=psutil.Process().memory_info().peak_wset)
        for variant in variants:evaluate(variant,-1)
        for rep in range(3):
            order=variants[rep:]+variants[:rep]
            for variant in order:rows.append(evaluate(variant,rep))
        print('timed',name,flush=True)
    summary=[]
    for name,step in choices:
        for variant in variants:
            selected=[r for r in rows if r['image']==name and r['variant']==variant]
            summary.append(dict(image=name,variant=variant,encode_seconds=statistics.median(r['encode_seconds'] for r in selected),decode_seconds=statistics.median(r['decode_seconds'] for r in selected),encode_cv_percent=100*statistics.pstdev(r['encode_seconds'] for r in selected)/statistics.mean(r['encode_seconds'] for r in selected),decode_cv_percent=100*statistics.pstdev(r['decode_seconds'] for r in selected)/statistics.mean(r['decode_seconds'] for r in selected)))
    write_json('timings.json',dict(provenance=meta(),repetitions=3,rows=rows,summary=summary,timing='Three warmed rotated full-call repetitions per promising image, source-isolated research methods in one serial pinned process. Ordinary is HQ at a different measured RD point; dense/sparse use exactly matching p10/quantizers. Setup/hook installation, internal base MSE, all transforms and entropy included; loading/hash/I/O/external scoring excluded.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
