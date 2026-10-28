import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_residuals import ARTIFACTS, IMAGES, load, setup, sha, provenance
import numpy as np
from PIL import Image
from hybrid import encode,decode,coefficients,correction,HEADER,unpack_coefficients


def main():
    setup(); meta=provenance()
    rng=np.random.default_rng(2003)
    residual=rng.integers(-255,256,size=(31,29,3)).astype(np.float32)
    coeff=coefficients(residual)
    restored=correction(coeff,1.0,31,29)
    transform_error={'max_abs':float(np.abs(restored-residual).max()),'mse':float(np.mean((restored-residual)**2,dtype=np.float64))}
    assert np.allclose(restored,residual,atol=.0002)
    warm=Image.new('RGB',(40,32),(100,140,170)); decode(encode(warm,64))
    rows=[]
    for name in IMAGES:
        image=load(name); original=np.asarray(image,dtype=np.float32)
        for step in (32,64,128):
            start=time.perf_counter(); data=encode(image,step); enc=time.perf_counter()-start
            path=ARTIFACTS/(name+f'.dct_q{step}.p17'); path.write_bytes(data)
            stored=path.read_bytes()
            start=time.perf_counter(); output=decode(stored); dec=time.perf_counter()-start
            checked,parts,delta,base=decode(stored,details=True)
            assert np.array_equal(output,checked)
            error=original-output.astype(np.float32)
            residual_error=original-base-delta
            row={'image':name,'architecture':'full_dct','step':step,'base_patch_count':10,'bytes':len(data),'bpp':8*len(data)/(image.width*image.height),'rgb_mse':float(np.mean(error**2,dtype=np.float64)),'encode_seconds':enc,'decode_seconds':dec,'residual_reconstruction_mse':float(np.mean(residual_error**2,dtype=np.float64)),'enhancement_rate_fraction':parts['enhancement_bytes']/len(data),'file_consistent':data==stored,'stream_sha256':sha(data),'pixels_sha256':sha(output.tobytes()),'file':str(path.relative_to(HERE)),**parts}
            assert row['base_bytes']+row['enhancement_bytes']+row['container_bytes']==row['bytes']
            rows.append(row)
            print(name,step,row['bytes'],round(row['rgb_mse'],3),flush=True)
    (HERE/'hybrid_results.json').write_text(json.dumps({'provenance':meta,'transform_roundtrip':transform_error,'rows':rows},indent=2)+'\n')

if __name__=='__main__': main()
