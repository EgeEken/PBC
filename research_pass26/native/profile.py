import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import time
from pbc3_trace import TimingTrace
rows=[]
for name in ('cat.jpg','googlewiki.png','eiffel.png'):
    image=load(name)
    for preset in ('quality','high_quality'):
        cfg=getattr(PBC3Config,preset)()
        PBC3.compress(image,config=cfg)
        row=dict(image=name,preset=preset)
        for operation in ('encode','decode'):
            trace=TimingTrace(operation)
            if operation=='encode': r=PBC3.compress(image,config=cfg,trace=trace)
            else: PBC3.decompress(r.data,trace=trace)
            report=trace.report()
            row[operation]=dict(total_ns=report['total_ns'],summary=report['summary'])
        row['stream_sha256']=sha(r.data)
        rows.append(row)
        print(name,preset,flush=True)
save('native/profile.json',rows)
