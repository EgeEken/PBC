import ast
import hashlib
import inspect
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')})
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import psutil
reference_spec=importlib.util.spec_from_file_location('pass16_reference_features',ROOT/'research_pass16/baseline/pbc3_features.py')
pbc3_features=importlib.util.module_from_spec(reference_spec)
reference_spec.loader.exec_module(pbc3_features)
from learned_filler import LearnedFiller
from feature_candidate import integer_order_stats
psutil.Process().cpu_affinity([12])


def candidate_source():
    source=inspect.getsource(pbc3_features.extract_features)
    source=source.replace('    p05, p25, p50, p75, p95 = np.percentile(flat, [5, 25, 50, 75, 95])', '''    lo, hi = flat.min(), flat.max()
    if target.dtype.kind in "iu" and canvas.dtype.kind in "iu" and -2**52 <= lo <= hi <= 2**52 and hi - lo <= 65535:
        percentiles, uniq = integer_order_stats(flat, int(lo), int(hi))
        p05, p25, p50, p75, p95 = percentiles
    else:
        p05, p25, p50, p75, p95 = np.percentile(flat, [5, 25, 50, 75, 95])
        uniq = np.unique(np.rint(flat / 2.0)).size / n''')
    source=source.replace('    uniq = np.unique(np.rint(flat / 2.0)).size / n\n    return', '    return')
    return source


def candidate_function():
    source=candidate_source()
    namespace={'np':np,'integer_order_stats':integer_order_stats}
    exec(compile(source,'<feature_candidate>','exec'),namespace)
    return namespace['extract_features']


def main():
    candidate=candidate_function()
    if '--integrated' in sys.argv:
        from pbc3_features import extract_features as candidate, _integer_order_stats
        expected=ast.parse(candidate_source().replace('integer_order_stats(', '_integer_order_stats('))
        assert ast.dump(ast.parse(inspect.getsource(candidate)))==ast.dump(expected), 'integrated feature AST drift'
        expected_helper=inspect.getsource(integer_order_stats.py_func).replace('integer_order_stats(', '_integer_order_stats(')
        assert ast.dump(ast.parse(inspect.getsource(_integer_order_stats.py_func)))==ast.dump(ast.parse(expected_helper)), 'integrated kernel AST drift'
    model=LearnedFiller.load(str(ROOT/'patch_policy.npz'))
    rng=np.random.default_rng(8128)
    rows=[]
    for channels in (3,4):
        for h,w in ((1,1),(2,3),(17,23),(64,71),(200,400)):
            for dtype in (np.int32,np.int64,np.float32,np.float64,np.uint8):
                target=rng.integers(0,256,size=(h,w,channels)).astype(dtype)
                canvas=rng.integers(-700,901,size=(h,w,channels)).astype(dtype)
                patterns=[('random',target,canvas),('constant',np.full_like(target,128),np.full_like(canvas,128)),('negative',np.zeros_like(target),np.full_like(canvas,255)),('contrast',target,np.where(target<128,0,255).astype(dtype)),('strided',target[::-1,::2],canvas[::-1,::2])]
                if dtype in (np.float32,np.float64):
                    patterns.append(('fractional',target+0.125,canvas-0.375))
                for pattern,t,cv in patterns:
                    ih,iw,_=t.shape
                    boxes=[(0,0,0,iw,ih),(channels-1,iw-1,ih-1,1,1),(1,0,0,max(1,iw//2),max(1,ih//2))]
                    for box in boxes:
                        args=(t,cv,box,7,0.8,iw,ih,50,channels)
                        a=pbc3_features.extract_features(*args); b=candidate(*args)
                        assert a.dtype==b.dtype==np.float32 and a.shape==b.shape==(32,)
                        assert a.tobytes()==b.tobytes(),(channels,h,w,str(dtype),pattern,box,np.where(a!=b),a,b)
                        na=(a-model.feat_mean)/model.feat_std; nb=(b-model.feat_mean)/model.feat_std
                        la,va=model._forward(na[None,:]); lb,vb=model._forward(nb[None,:])
                        assert la.tobytes()==lb.tobytes() and va.tobytes()==vb.tobytes()
                        rows.append({'channels':channels,'shape':[h,w],'dtype':np.dtype(dtype).name,'pattern':pattern,'box':box,'feature_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'action':int(la.argmax(1)[0])})
    result={'cases':len(rows),'features_byte_identical':True,'logits_values_actions_identical':True,'rows':rows}
    (ROOT/'research_pass16'/('integrated_checks.json' if '--integrated' in sys.argv else 'feature_checks.json')).write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__': main()
