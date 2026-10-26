import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import learned_filler as lf
from stop_policy import enabled


def main():
    model=object.__new__(lf.LearnedFiller)
    model.candidates=model.top_k=1
    model.q_override=.8
    model.feat_mean=np.zeros(1,dtype=np.float32)
    model.feat_std=np.ones(1,dtype=np.float32)
    model.lmin=.05; model.lmax=5.
    model._featurize=lambda *args: np.zeros((1,1),dtype=np.float32)
    model._forward=lambda *args,**kwargs: (np.zeros((1,30)),np.zeros((1,1)))
    boxes=[(0,0,0,1,1),(0,1,0,1,1),(0,0,1,1,1)]
    array=np.zeros((2,2,3),dtype=np.int16)
    outcomes=[]
    for mode,first,second,pool,accepted,build_count in [
        ('diagnose',0,1,boxes,False,1),('S1',0,1,boxes,False,1),
        ('S2',0,1,boxes,True,2),('S2',0,0,boxes,False,2),
        ('S2',1,1,boxes,True,1),('S2',0,1,boxes[:1],False,1),('S2',0,1,[],False,0)]:
        calls=[]; rng=[]; stats={}
        def propose(*args,**kwargs):
            rng.append('one ordinary pool')
            return pool
        def build(target,canvas,box,*args,**kwargs):
            calls.append(box)
            return dict(channel=0),np.zeros((1,1)),np.zeros((1,1)),first if box==boxes[0] else second,1.
        with patch.object(lf,'propose_boxes',propose),patch.object(lf,'build_action',build):
            original=lf.LearnedFiller.select_patch
            with enabled(mode,stats):
                result=model.select_patch(array,array,SimpleNamespace(patch_count=50),rng,2,1,0)
            assert lf.LearnedFiller.select_patch is original
        assert (result[0] is not None)==accepted and len(calls)==build_count and len(rng)==1
        assert all(box in boxes[:2] for box in calls)
        outcomes.append(dict(mode=mode,first=first,second=second,proposals=len(pool),accepted=accepted,builds=len(calls)))
    output=dict(passed=True,cases=outcomes,single_proposal_pool=True,no_third_proposal=True,hooks_restored=True)
    (ROOT/'research_pass7/stop_policy_checks.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':
    main()
