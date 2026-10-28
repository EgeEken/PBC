import statistics
import time
from common import *
from research_pass22 import decoder


def main():
    setup();rows=[];inputs={}
    variants=('ordinary_hq','reference','optimized')
    for name in CASES:
        data,record=saved(name);ordinary,ordinary_record=saved(name,True)
        inputs[name]={'hybrid':{k:record[k] for k in ('stream_sha256','pixels_sha256','step','bytes','rgb_mse')},'ordinary_hq':ordinary_record}
        def evaluate(label,rep):
            stream=ordinary if label=='ordinary_hq' else data
            start=time.perf_counter()
            output=PBC3.decompress(stream) if label=='ordinary_hq' else reference.decode(stream) if label=='reference' else decoder.decode(stream)
            elapsed=time.perf_counter()-start
            pixels=np.asarray(output.image) if label=='ordinary_hq' else output
            assert sha(pixels.tobytes())==(ordinary_record if label=='ordinary_hq' else record)['pixels_sha256']
            return dict(image=name,variant=label,rep=rep,seconds=elapsed)
        for label in variants:evaluate(label,-1)
        for rep in range(3):
            order=variants[rep:]+variants[:rep]
            for label in order:rows.append(evaluate(label,rep))
        print('complete',name,flush=True)
    summary=[]
    for name in CASES:
        for label in variants:
            times=[r['seconds'] for r in rows if r['image']==name and r['variant']==label]
            summary.append(dict(image=name,variant=label,median=statistics.median(times),mean=statistics.mean(times),minimum=min(times),maximum=max(times),cv_percent=100*statistics.pstdev(times)/statistics.mean(times)))
    write('timings.json',dict(provenance=meta(),inputs=inputs,rows=rows,summary=summary,repetitions=3,warmup='One complete call per variant on the actual saved image stream before three rotated rounds.',timing='Complete saved-file decoding including parser/base/recovery/transforms/output; excludes IO/hash/scoring/logging. Ordinary HQ at its existing different RD point. All saved stream and output hashes checked.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
