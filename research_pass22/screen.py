import statistics
import time
from common import *
from research_pass22 import decoder


def main():
    setup();rows=[]
    variants={'reference':reference.decode,'sparse_only':lambda d:decoder.decode(d,reuse=False),'reuse_only':lambda d:decoder.decode(d,sparse=False),'sparse_reuse':decoder.decode}
    for name in CASES:
        data,record=saved(name)
        for label,fn in variants.items():
            output=fn(data)
            assert sha(output.tobytes())==record['pixels_sha256'],(name,label,np.count_nonzero(output!=reference.decode(data)))
        for rep in range(3):
            names=list(variants);names=names[rep:]+names[:rep]
            for label in names:
                start=time.perf_counter();output=variants[label](data);elapsed=time.perf_counter()-start
                assert sha(output.tobytes())==record['pixels_sha256']
                rows.append(dict(image=name,variant=label,rep=rep,seconds=elapsed))
        print(name,{label:statistics.median(r['seconds'] for r in rows if r['image']==name and r['variant']==label) for label in variants},flush=True)
    write('screen.json',dict(provenance=meta(),rows=rows,repetitions=3,warmup='Each complete variant once per actual saved file.',timing='Full decode, rotated variants; hashes/I/O/scoring outside. Diagnostic ablations of one sparse method and one buffer reuse method.'))


if __name__=='__main__':main()
