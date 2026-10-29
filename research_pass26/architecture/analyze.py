import json
import math
from pathlib import Path
PASS=Path(__file__).resolve().parents[1]
rows=json.loads((PASS/'architecture/results.json').read_text())
summary=[]
for target in [r for r in rows if r['codec']=='PBC31']:
    item=dict(image=target['image'],preset=target['setting'],pbc_bytes=target['bytes'],pbc_bpp=target['bpp'],pbc_mse=target['mse'],codecs={})
    for codec in ('JPEG','WEBP','AVIF','JXL','working_JPEG','working_WEBP'):
        points=sorted([r for r in rows if r['image']==target['image'] and r['codec']==codec],key=lambda r:r['bytes'])
        close=min(points,key=lambda r:abs(math.log(r['bytes']/target['bytes'])))
        near=[r for r in points if abs(r['bytes']/target['bytes']-1)<=.05]
        dominate=[r for r in points if r['bytes']<=target['bytes'] and r['mse']<=target['mse']]
        bracket=[]
        for a,b in zip(points,points[1:]):
            if a['bytes']<=target['bytes']<=b['bytes'] and a['bytes']<b['bytes']:
                estimate=a['mse']+(b['mse']-a['mse'])*(target['bytes']-a['bytes'])/(b['bytes']-a['bytes'])
                bracket.append(dict(lower_setting=a['setting'],upper_setting=b['setting'],lower_bytes=a['bytes'],upper_bytes=b['bytes'],estimated_mse=estimate))
        item['codecs'][codec]=dict(closest=close,near=near,dominating=dominate,brackets=bracket)
    summary.append(item)
(PASS/'architecture/comparisons.json').write_text(json.dumps(summary,indent=2)+'\n')
