import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_residuals import ARTIFACTS,IMAGES,load,setup,sha,provenance
import numpy as np
from PIL import Image
from hybrid import encode_tiled,decode


def main():
    setup(); meta=provenance()
    baseline=json.loads((HERE/'baseline.json').read_text())
    warm=Image.new('RGB',(40,32),(100,140,170)); decode(encode_tiled(warm,64,100000))
    rows=[]
    for name in IMAGES:
        image=load(name); original=np.asarray(image,dtype=np.float32)
        for preset in ('quality','high_quality'):
            control=next(r for r in baseline['rows'] if r['image']==name and r['preset']==preset)
            start=time.perf_counter(); data=encode_tiled(image,64,control['bytes']); enc=time.perf_counter()-start
            path=ARTIFACTS/(name+f'.tiles64_{preset}.p17'); path.write_bytes(data)
            stored=path.read_bytes()
            start=time.perf_counter(); output=decode(stored); dec=time.perf_counter()-start
            checked,parts,delta,base=decode(stored,details=True)
            assert np.array_equal(output,checked) and len(data)<=control['bytes']
            error=original-output.astype(np.float32)
            row={'image':name,'architecture':'adaptive_tiles','step':64,'target_preset':preset,'target_bytes':control['bytes'],'base_patch_count':10,'bytes':len(data),'bpp':8*len(data)/(image.width*image.height),'rgb_mse':float(np.mean(error**2,dtype=np.float64)),'encode_seconds':enc,'decode_seconds':dec,'residual_reconstruction_mse':float(np.mean((original-base-delta)**2,dtype=np.float64)),'enhancement_rate_fraction':parts['enhancement_bytes']/len(data),'file_consistent':data==stored,'stream_sha256':sha(data),'pixels_sha256':sha(output.tobytes()),'file':str(path.relative_to(HERE)),'rate_gap_percent':100*(len(data)/control['bytes']-1),'mse_change_percent':100*(np.mean(error**2,dtype=np.float64)/control['rgb_mse']-1),**parts}
            assert row['base_bytes']+row['enhancement_bytes']+row['container_bytes']==row['bytes']
            rows.append(row)
            print(name,preset,row['selected_tiles'],row['bytes'],round(row['rgb_mse'],3),round(row['mse_change_percent'],2),flush=True)
    (HERE/'tiles_results.json').write_text(json.dumps({'provenance':meta,'rows':rows},indent=2)+'\n')

if __name__=='__main__': main()
