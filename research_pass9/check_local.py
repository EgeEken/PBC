from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from hf_release_job import INPUTS,HOLDOUT,PRESETS,EQUIVALENCE,validate,run
from worker import THREADS


def main():
    representative=['cat.jpg','eiffel.png','googlewiki.png']
    results=[]
    for label,images in (('smoke',representative),('extension',[x for x in INPUTS+HOLDOUT if x not in representative])):
        spec=dict(images=images,presets=list(PRESETS),variants=list(EQUIVALENCE),repeats=1)
        result=run(validate(spec))
        (HERE/('local_'+label+'_raw.json')).write_text(json.dumps(result),encoding='utf-8')
        results.append(result)
        print(label,len(result['equivalent']),'cases equivalent',flush=True)
    import os
    os.environ.update(THREADS)
    sys.path.insert(0,str(ROOT))
    import numpy as np
    from PIL import Image
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    image=Image.fromarray(np.random.default_rng(7).integers(0,256,(48,64,3),dtype=np.uint8))
    config=PBC3Config.balanced(learned_filler_model_path=str(ROOT/'patch_policy.npz'),patch_count=5)
    expected={mode:PBC3.compress(image,config=config,rng_mode=mode).data for mode in ('current','release')}
    assert PBC3.compress(image,config=config).data==expected['release']
    assert list(PBC3.compress_stream(image,config=config,rng_mode='release',frame_every=0))[-1]['result'].data==expected['release']
    modes=('release','current','release','current')
    with ThreadPoolExecutor(max_workers=4) as pool:
        values=list(pool.map(lambda mode:PBC3.compress(image,config=config,rng_mode=mode).data,modes))
    assert all(data==expected[mode] for mode,data in zip(modes,values))
    try:
        PBC3.compress(image,config=config,rng_mode='unknown')
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown RNG accepted')
    spec=dict(images=representative,presets=list(PRESETS),variants=['release','dev_release_rng','dev_current_rng'],repeats=3)
    invalid=[dict(spec,images=['../secret']),dict(spec,images=['cat.jpg','cat.jpg']),dict(spec,variants=['dev_current_rng']),dict(spec,repeats=True),dict(spec,repeats=4),dict(spec,code='x'),dict(spec,images=list(INPUTS)),dict(spec,images=['pigeon.JPG']),dict(spec,presets=['custom']),None,[]]
    for value in invalid:
        try:
            validate(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
    rows=[{k:v for k,v in row.items() if k not in ('trajectory','encode_seconds','decode_seconds','trial')} for r in results for row in r['rows']]
    summary=dict(cases=sum(len(r['equivalent']) for r in results),equivalent=True,metadata=results[0]['metadata'],rows=rows,
        explicit_default_equals_omitted=True,compress_stream_option=True,concurrent_mixed_rng_calls=4,invalid_specifications_rejected=len(invalid),unknown_rng_rejected=True)
    (HERE/'local_equivalence.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('100 release cases equivalent; default, streaming, mixed concurrent selection and invalid specifications passed.',flush=True)


if __name__=='__main__':
    main()
