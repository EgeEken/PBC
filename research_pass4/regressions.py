from collections import Counter
import json
import sys

from support import ROOT, OUT, PASS2, PASS3, save, sha, snapshot, load


def main():
    sys.path.insert(0,str(ROOT))
    import numpy as np
    import psutil
    from PIL import Image,ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    import learned_filler as filler
    psutil.Process().cpu_affinity([psutil.Process().cpu_affinity()[0]])
    select=load('pass4_regression_frozen',PASS3/'representation.py').select
    prior=json.loads((PASS3/'allocation_full.json').read_text())
    before=snapshot()
    results=[]
    for name in ('mario64.png','monet.jpg','eiffel.png','pigeon.JPG'):
        expected=next(r for r in prior if r['trial']==0 and r['image']==name and r['label']=='q500p35_integrated_t0.4')
        baseline=next(r for r in prior if r['trial']==0 and r['image']==name and r['label']=='q500p35_none')
        fixed=next(r for r in prior if r['trial']==0 and r['image']==name and r['label']=='q500p35_fixed_t0.4')
        with Image.open(ROOT/expected['dataset']/name) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
        decisions=[]
        original_build=filler.build_action
        counters={}
        def build(target,canvas,box,config,channel_bits,action,trace=None):
            reference=original_build(target,canvas,box,config,channel_bits,action,trace=trace)
            alternatives=[]
            raw_build=filler._build_action
            def alternative(*args,**kwargs):
                result=raw_build(*args,**kwargs)
                alternatives.append(result)
                return result
            try:
                filler._build_action=alternative
                chosen=select(target,canvas,box,config,channel_bits,reference,.4,counters)
            finally:
                filler._build_action=raw_build
            compact=lambda r:dict(cell=r[0]['cell_size'],precision=r[0]['max_bitcount'],
                actual_bitcount=r[0]['bitcount'],raw_bits=r[4],local_sse_gain=r[3])
            kind='original' if chosen is reference else 'coarser_grid' if chosen[0]['cell_size']!=reference[0]['cell_size'] else 'lower_precision'
            lam=.4*reference[3]/max(1.,reference[4])
            decisions.append(dict(step=len(decisions),box=box,action=action,original=compact(reference),
                alternatives=[compact(r) for r in alternatives],selected=compact(chosen),kind=kind,
                policy_stop=reference[3]<=0,lambda_raw=lam,
                retained_gain=chosen[3]/reference[3] if reference[3]>0 else None,
                retained_raw_bits=chosen[4]/reference[4]))
            return chosen
        cfg=PBC3Config.quality(patch_count=35,auto_downsample_max_pixels=500000,learned_filler_model_path=str(ROOT/'patch_policy.npz'))
        try:
            filler.build_action=build
            result=PBC3.compress(image,config=cfg)
        finally:
            filler.build_action=original_build
        assert sha(result.data)==expected['stream_sha256']
        accepted=[r for r in decisions if not r['policy_stop']]
        changed=[r for r in accepted if r['kind']!='original']
        first_payload=next((i for i,(a,b) in enumerate(zip(expected['patch_signatures'],baseline['patch_signatures'])) if a!=b),None)
        fields=('mse','bpp','final_bytes','raw_bytes','iterative_patches','experimental_stats')
        results.append(dict(dataset=expected['dataset'],image=name,decisions=decisions,
            selected_types=dict(Counter(r['kind'] for r in accepted)),changed=len(changed),
            changed_mean_retained_gain=float(np.mean([r['retained_gain'] for r in changed])) if changed else None,
            changed_mean_retained_raw_bits=float(np.mean([r['retained_raw_bits'] for r in changed])) if changed else None,
            ordinary={k:baseline.get(k) for k in fields},integrated={k:expected.get(k) for k in fields},fixed={k:fixed.get(k) for k in fields},
            first_changed_stream_patch=first_payload,byte_identical_to_frozen=True))
        print('regression',name,results[-1]['selected_types'],flush=True)
    save('regression_diagnostics',results)
    save('regression_provenance',dict(before=before,after=snapshot(),scope='Untimed instrumentation of frozen selector; no policy change or extra search/model evaluations. Local gains are working-channel SSE, not original RGB.'))


if __name__=='__main__':
    main()
