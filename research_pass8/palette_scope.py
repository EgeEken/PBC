from contextlib import contextmanager
import time

import numpy as np

import pbc3_ops as ops
from palette import bounded_palette,local_sse


def refine(target,canvas,config,patch,values,delta,channel_bits,stats,position):
    started=time.perf_counter()
    stats['considered']+=1
    c,x,y,w,h=(patch[k] for k in ('channel','x','y','w','h'))
    residual=target[y:y+h,x:x+w,c]-canvas[y:y+h,x:x+w,c]
    small=np.rint(ops.project_residual_to_grid(residual,patch['cell_size'],config.residual_projection_mode)).astype(np.int16)
    candidate,candidate_values=bounded_palette(small,patch,config,98)
    stats['bounds_changed']+=int((candidate['neg'],candidate['pos'])!=(patch['neg'],patch['pos']))
    cost=ops.patch_bits_for(candidate,channel_bits)
    original_cost=ops.patch_bits_for(patch,channel_bits)
    result=patch,values,delta
    if cost<=original_cost and not np.array_equal(candidate_values,values):
        before=target[y:y+h,x:x+w,c]-np.clip(canvas[y:y+h,x:x+w,c]+delta,0,255)
        before_sse=int(np.sum(before.astype(np.int64)**2))
        candidate_sse,candidate_delta=local_sse(target,canvas,candidate,candidate_values)
        stats['exact_candidates']+=1
        if candidate_sse<before_sse:
            stats['accepted']+=1
            stats['local_sse_reduction']+=before_sse-candidate_sse
            stats['raw_bits_saved']+=original_cost-cost
            stats['accepted_positions'].append(position)
            result=candidate,candidate_values,candidate_delta
    stats['palette_seconds']+=time.perf_counter()-started
    return result


@contextmanager
def enabled(ignored,stats):
    from pbc3_heads import DownsampleInitHead,FillerHead
    from learned_filler import LearnedFiller
    started=time.perf_counter()
    original_init,original_fill=DownsampleInitHead.select,FillerHead.select
    stats.update(percentile=98,considered=0,bounds_changed=0,exact_candidates=0,accepted=0,
        local_sse_reduction=0,raw_bits_saved=0,palette_seconds=0.,accepted_positions=[])

    def init(self,channel,target,canvas,w,h,config,channel_bits,trace=None):
        patch,values,delta,cell,bits=original_init(self,channel,target,canvas,w,h,config,channel_bits,trace=trace)
        patch,values,delta=refine(target,canvas,config,patch,values,delta,channel_bits,stats,['init',channel])
        return patch,values,delta,cell,bits

    def fill(self,target,canvas,config,rng,channel_bits,step,current_channel,boxes,canvas_patches,debug_lines,trace=None,abs_error=None):
        assert isinstance(self.learned,LearnedFiller)
        patch,values,delta=original_fill(self,target,canvas,config,rng,channel_bits,step,current_channel,boxes,
            canvas_patches,debug_lines,trace=trace,abs_error=abs_error)
        if patch is None:
            return patch,values,delta
        return refine(target,canvas,config,patch,values,delta,channel_bits,stats,['iterative',step])

    try:
        DownsampleInitHead.select=init
        FillerHead.select=fill
        yield
    finally:
        DownsampleInitHead.select=original_init
        FillerHead.select=original_fill
        stats['encode_scope_seconds']=time.perf_counter()-started
