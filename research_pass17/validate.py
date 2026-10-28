import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_residuals import ROOT,ARTIFACTS,load,setup,sha,provenance
from hybrid import encode,encode_tiled,decode,pack_coefficients,unpack_coefficients,HEADER
from PBC3 import PBC3
import numpy as np
from PIL import Image


def main():
    setup(); rows=[]
    for name in ('hybrid_results.json','tiles_results.json'):
        for row in json.loads((HERE/name).read_text())['rows']:
            data=(HERE/row['file']).read_bytes()
            output,parts,delta,base=decode(data,details=True)
            assert sha(data)==row['stream_sha256'] and sha(output.tobytes())==row['pixels_sha256']
            assert len(data)==parts['base_bytes']+parts['container_bytes']+parts['enhancement_bytes']
            original=np.asarray(load(row['image']),dtype=np.float32)
            residual=original-base
            mse=float(np.mean((original-output.astype(np.float32))**2,dtype=np.float64))
            assert mse==row['rgb_mse']
            rows.append({'file':row['file'],'stream_sha256':sha(data),'pixels_sha256':sha(output.tobytes()),'rgb_mse':mse,'residual_min':float(residual.min()),'residual_max':float(residual.max())})
    rng=np.random.default_rng(83)
    quantized=rng.integers(-32000,32001,size=(2,3,3,8,8),dtype=np.int16)
    assert np.array_equal(quantized,unpack_coefficients(pack_coefficients(quantized),16,24))
    synthetic=[]
    for h,w in ((9,13),(35,37)):
        image=Image.fromarray(rng.integers(0,256,size=(h,w,3),dtype=np.uint8))
        for mode in ('full','tiles'):
            fn=(lambda:encode(image,64)) if mode=='full' else (lambda:encode_tiled(image,64,100000))
            a,b=fn(),fn(); assert a==b
            out=decode(a); assert out.shape==(h,w,3) and out.dtype==np.uint8
            for bad in (a[:10],b'FAIL'+a[4:],a[:-1]):
                try: decode(bad)
                except ValueError: pass
                else: raise AssertionError('malformed file accepted')
            synthetic.append({'shape':[h,w],'mode':mode,'sha256':sha(a)})
    try: PBC3.decompress((ARTIFACTS/'googlewiki.png.dct_q128.p17').read_bytes())
    except ValueError: pass
    else: raise AssertionError('ordinary v0 decoder accepted research magic')
    independent=[]
    code='''from pathlib import Path
import sys
from learned_filler import LearnedFiller
def forbidden(*args,**kwargs): raise AssertionError("decoder loaded a policy")
LearnedFiller.load=forbidden
from hybrid import decode
from PIL import Image
Image.fromarray(decode(Path(sys.argv[1]).read_bytes())).save(sys.argv[2])
'''
    with tempfile.TemporaryDirectory(prefix='decode-',dir=ARTIFACTS) as temporary:
        for name,step in (('googlewiki.png',128),('eiffel.png',64)):
            path=ARTIFACTS/(name+f'.dct_q{step}.p17'); output=Path(temporary)/(name+'.png')
            env={**os.environ,'PYTHONPATH':os.pathsep.join((str(ROOT),str(HERE)))}
            subprocess.run([sys.executable,'-c',code,str(path),str(output)],cwd=temporary,env=env,check=True)
            pixels=np.asarray(Image.open(output))
            expected=next(r for r in rows if r['file']==str(path.relative_to(HERE)))
            assert sha(pixels.tobytes())==expected['pixels_sha256']
            independent.append(expected['file'])
    result={'provenance':provenance(),'completed_files_checked':len(rows),'all_sizes_sha_pixels_mse_consistent':True,'signed_int16_extreme_payload_roundtrip':True,'synthetic_determinism_and_boundaries':synthetic,'independent_process_no_policy_load':independent,'v0_rejects_experimental_magic':True,'rows':rows}
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Validated',len(rows),'completed files; fresh-process file-only decoding, signed range, boundaries and determinism passed')

if __name__=='__main__': main()
