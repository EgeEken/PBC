import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_residuals import setup,load,provenance
from hybrid import encode,decode
from PBC3 import PBC3
from pbc3_types import PBC3Config


def main():
    setup()
    baseline=json.loads((HERE/'baseline.json').read_text())
    hybrids=json.loads((HERE/'hybrid_results.json').read_text())
    rows=[]
    choices=(('googlewiki.png',128,'high_quality'),('eiffel.png',64,'quality'))
    for name,step,preset in choices:
        image=load(name)
        cfg=getattr(PBC3Config,preset)()
        def evaluate(variant):
            start=time.perf_counter()
            data=encode(image,step) if variant=='hybrid' else PBC3.compress(image,config=cfg).data
            enc=time.perf_counter()-start
            start=time.perf_counter()
            output=decode(data) if variant=='hybrid' else PBC3.decompress(data).image
            dec=time.perf_counter()-start
            expected=next(r for r in (hybrids['rows'] if variant=='hybrid' else baseline['rows']) if r['image']==name and (r['step']==step if variant=='hybrid' else r['preset']==preset))
            digest=hashlib.sha256(data).hexdigest(); assert digest==expected['stream_sha256']
            return {'image':name,'variant':variant,'step':step if variant=='hybrid' else None,'ordinary_preset':preset,'encode_seconds':enc,'decode_seconds':dec,'stream_sha256':digest}
        evaluate('ordinary'); evaluate('hybrid')
        for rep in range(3):
            order=('ordinary','hybrid') if rep%2==0 else ('hybrid','ordinary')
            for variant in order: rows.append({**evaluate(variant),'rep':rep})
        print('timed',name,flush=True)
    summary={}
    for name,step,preset in choices:
        summary[name]={}
        for variant in ('ordinary','hybrid'):
            selected=[r for r in rows if r['image']==name and r['variant']==variant]
            summary[name][variant]={k:statistics.median(r[k] for r in selected) for k in ('encode_seconds','decode_seconds')}
            summary[name][variant]['encode_cv_percent']=100*statistics.pstdev(r['encode_seconds'] for r in selected)/statistics.mean(r['encode_seconds'] for r in selected)
    (HERE/'timings.json').write_text(json.dumps({'provenance':provenance(),'repetitions':3,'rows':rows,'summary':summary,'frozen_hybrid_sha256':hashlib.sha256((HERE/'hybrid.py').read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
