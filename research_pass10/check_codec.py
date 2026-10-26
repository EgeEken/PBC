import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
THREADS={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
os.environ.update(THREADS)
sys.path.insert(0,str(ROOT))


def cases():
    import numpy as np
    from PIL import Image,ImageDraw,ImageOps
    with Image.open(ROOT/'hpt_data/cat.jpg') as image:
        photo=ImageOps.exif_transpose(image).convert('RGB')
    rgba=np.zeros((64,96,4),dtype=np.uint8)
    rgba[:,:,:3]=(45,100,175)
    rgba[:,:,3]=np.where(np.arange(96)[None,:]<48,0,255)
    graphics=Image.new('RGB',(127,95),'white')
    draw=ImageDraw.Draw(graphics)
    draw.text((5,5),'PBC baseline 3.1',fill='black')
    draw.rectangle((13,29,98,69),outline='red',width=3)
    draw.line((0,94,126,0),fill='blue',width=2)
    small=Image.fromarray(np.random.default_rng(91).integers(0,256,(13,17,3),dtype=np.uint8))
    return [('photo','compression',photo,{}),('rgba','balanced',Image.fromarray(rgba),{}),
        ('graphics','quality',graphics,{}),('native','high_quality',small,dict(random_seed=17))]


def describe(codec,encoded,image):
    import numpy as np
    from pbc3_types import BitReader
    decoded=codec.decompress(encoded.data).image
    assert decoded.size==image.size
    assert np.array_equal(np.asarray(encoded.image),np.asarray(decoded))
    mse=float(np.mean(np.subtract(np.asarray(image),np.asarray(decoded),dtype=np.float64)**2))
    assert np.isclose(mse,encoded.mse,rtol=1e-6)
    _,body=codec._open_body(encoded.data)
    header=codec._read_header(BitReader(body))
    return dict(stream_sha256=hashlib.sha256(encoded.data).hexdigest(),pixels_sha256=hashlib.sha256(np.asarray(decoded).tobytes()).hexdigest(),
        mse=mse,final_bytes=len(encoded.data),bpp=8*len(encoded.data)/(image.width*image.height),patches=header[10],
        image_mode=image.mode,width=image.width,height=image.height)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=('release','previous'))
    args=parser.parse_args()
    if args.worker=='release':
        sys.path.insert(0,str(ROOT/'research_pass1/release'))
    if args.worker=='previous':
        module=types.ModuleType('previous_PBC3')
        source=subprocess.check_output(['git','show','e6c8eca726b3048a92514bd09d31fca0b3e00fd7:PBC3.py'])
        exec(compile(source,'previous_PBC3.py','exec'),module.__dict__)
        codec=module.PBC3
    else:
        from PBC3 import PBC3 as codec
    from pbc3_types import PBC3Config
    items=cases()
    configs=[getattr(PBC3Config,preset)(learned_filler_model_path=str(ROOT/'patch_policy.npz'),**extra) for _,preset,_,extra in items]
    if args.worker:
        rows=[]
        for (name,preset,image,_),config in zip(items,configs):
            if args.worker=='previous' and len(rows)==2:
                break
            result=codec.compress(image,config=config)
            rows.append(dict(name=name,preset=preset,**describe(codec,result,image),payload=base64.b64encode(result.data).decode()))
        print(json.dumps(rows)); return
    frozen={}
    for source in ('release','previous'):
        output=subprocess.check_output([sys.executable,str(Path(__file__)), '--worker',source],cwd=ROOT,text=True)
        frozen[source]=json.loads(output)
    rows=[]; expected=[]; custom=[]
    fixtures=HERE/'hf_runs/fixtures'; fixtures.mkdir(parents=True,exist_ok=True)
    for i,((name,preset,image,_),config) in enumerate(zip(items,configs)):
        implicit=codec.compress(image,config=config)
        explicit=codec.compress(image,config=config,rng_mode='release')
        assert implicit.data==explicit.data==base64.b64decode(frozen['release'][i]['payload']),name
        expected.append(implicit.data)
        if i<2:
            old=codec.compress(image,config=config,rng_mode='current')
            assert old.data==base64.b64decode(frozen['previous'][i]['payload'])
            custom.append(old.data)
        else:
            events=list(codec.compress_stream(image,config=config,frame_every=1))
            assert events[-1]['result'].data==implicit.data
        image.save(fixtures/(name+'.png'))
        rows.append(dict(name=name,preset=preset,**describe(codec,implicit,image)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(codec.compress,items[0][2],config=configs[0],rng_mode='current'),
            pool.submit(codec.compress,items[1][2],config=configs[1],rng_mode='release')]
        assert futures[0].result().data==custom[0]
        assert futures[1].result().data==expected[1]
    try:
        codec.compress(items[3][2],config=configs[3],rng_mode='invalid')
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid RNG accepted')
    old=Path(r'C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC\research_pass1\space_googlewiki.balanced.pbc3').read_bytes()
    import numpy as np
    assert hashlib.sha256(np.asarray(codec.decompress(old).image).tobytes()).hexdigest()=='6762b473012c82cf1de553a964dfed84e5eb990e6d8dc83065628fcab9445d5c'
    worker=(ROOT/'research_pass9/worker.py').read_text(encoding='utf-8')
    assert "'release' if source=='dev_release_rng' else 'current'" in worker
    outcome=dict(encodes=20,release_identical_cases=4,unseen_synthetic_cases=3,custom_preserved_cases=2,default_equals_explicit_release=True,
        streaming_cases=2,mixed_concurrent_calls=2,invalid_rng_rejected=True,existing_v0_decode=True,internal_mse_correct=True,
        worker_modes_explicit=True,rows=rows,release_reference='2e59f1b06980aa467da0cde94fa33c3f9a0d4d51',previous_default_reference='e6c8eca726b3048a92514bd09d31fca0b3e00fd7')
    (HERE/'local_checks.json').write_text(json.dumps(outcome,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':
    main()
