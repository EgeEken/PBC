import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_residuals import ARTIFACTS,IMAGES,load,setup,measure,provenance,sha
from pbc3_types import PBC3Config
import numpy as np
from hybrid import HEADER
from PBC3 import PBC3


def main():
    setup(); rows=[]
    targeted='--targeted' in sys.argv
    budgets={'monet.jpg':8,'googlewiki.png':13,'mario64.png':13,'eiffel.png':5} if targeted else dict.fromkeys(IMAGES,10)
    for name,budget in budgets.items():
        image=load(name)
        config=PBC3Config.high_quality(patch_count=budget)
        data,output,row=measure(image,config)
        path=ARTIFACTS/(name+f'.high_quality_p{budget}.pbc'); path.write_bytes(data)
        row.update({'image':name,'preset':f'high_quality_p{budget}','file':str(path.relative_to(HERE))})
        rows.append(row)
        if targeted:
            print(name,budget,row['bytes'],round(row['rgb_mse'],3),flush=True); continue
        hybrid=(ARTIFACTS/(name+'.dct_q128.p17')).read_bytes()
        header=HEADER.unpack_from(hybrid); size=header[3]
        base=hybrid[HEADER.size:HEADER.size+size]
        pixels=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32)
        error=np.asarray(image,dtype=np.float32)-pixels
        rows.append({'image':name,'preset':'quality_p10_base','bytes':len(base),'bpp':8*len(base)/(image.width*image.height),'rgb_mse':float(np.mean(error**2,dtype=np.float64)),'stream_sha256':sha(base),'pixels_sha256':sha(pixels.astype(np.uint8).tobytes()),'encode_seconds':None,'decode_seconds':None,'file':'Embedded unchanged Quality p10 base from the saved full hybrid file; not another encode/timing.'})
        print(name,row['bytes'],round(row['rgb_mse'],3),flush=True)
    (HERE/('targeted_controls.json' if targeted else 'extra_controls.json')).write_text(json.dumps({'provenance':provenance(),'rows':rows},indent=2)+'\n')

if __name__=='__main__': main()
