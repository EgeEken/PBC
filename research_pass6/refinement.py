from contextlib import contextmanager
import time

import numpy as np


def refine(target,canvas,config,patch,values,delta,stats,position):
    import pbc3_ops as ops
    stats['attempted']=stats.get('attempted',0)+1
    c,x,y,w,h=(patch[k] for k in ('channel','x','y','w','h'))
    layer=canvas[y:y+h,x:x+w,c]
    desired=target[y:y+h,x:x+w,c]
    error=desired-np.clip(layer+delta,0,255)
    original_sse=int(np.sum(error.astype(np.int64)**2))
    feedback=ops.signed_resample(error,*values.shape)
    stats['extra_projections']=stats.get('extra_projections',0)+1
    palette=ops.palette_generator(patch['mask'],patch['max_bitcount'],patch['neg'],patch['pos'],config.positive_bias)
    proposed=np.rint(values.astype(np.float32)+.5*feedback).astype(np.int16)
    indices=ops.quantize_signed(proposed,palette)
    changed=int(np.count_nonzero(indices!=patch['indices']))
    stats['proposed_index_changes']=stats.get('proposed_index_changes',0)+changed
    stats['indices_considered']=stats.get('indices_considered',0)+indices.size
    if not changed:
        return patch,values,delta
    candidate_values=palette[indices]
    candidate_delta=ops.signed_resample(candidate_values,h,w)
    stats['extra_reconstructions']=stats.get('extra_reconstructions',0)+1
    candidate_error=desired-np.clip(layer+candidate_delta,0,255)
    candidate_sse=int(np.sum(candidate_error.astype(np.int64)**2))
    stats['evaluated_original_sse']=stats.get('evaluated_original_sse',0)+original_sse
    if candidate_sse>=original_sse:
        return patch,values,delta
    candidate={**patch,'indices':indices}
    assert ops.patch_bits_for(candidate,2)==ops.patch_bits_for(patch,2)
    stats['accepted']=stats.get('accepted',0)+1
    stats['accepted_index_changes']=stats.get('accepted_index_changes',0)+changed
    stats['local_sse_reduction']=stats.get('local_sse_reduction',0)+original_sse-candidate_sse
    stats.setdefault('accepted_positions',[]).append(position)
    return candidate,candidate_values,candidate_delta


@contextmanager
def enabled(mode,stats):
    from pbc3_heads import DownsampleInitHead,FillerHead
    started=time.perf_counter()
    stats.update(feedback_strength=.5,maximum_iterative_attempts=3)
    owner=DownsampleInitHead if mode=='init' else FillerHead
    original=owner.select
    if mode=='init':
        def select(self,channel,target,canvas,w,h,config,channel_bits,trace=None):
            patch,values,delta,cell,bits=original(self,channel,target,canvas,w,h,config,channel_bits,trace=trace)
            patch,values,delta=refine(target,canvas,config,patch,values,delta,stats,channel)
            return patch,values,delta,cell,bits
    else:
        def select(self,target,canvas,config,*args,**kwargs):
            patch,values,delta=original(self,target,canvas,config,*args,**kwargs)
            if patch is not None and stats.get('attempted',0)<3 and patch['cell_size']>1 and patch['indices'].size*patch['bitcount']>=8192:
                return refine(target,canvas,config,patch,values,delta,stats,int(args[5]))
            return patch,values,delta
    try:
        owner.select=select
        yield
    finally:
        owner.select=original
        stats['encode_scope_seconds']=time.perf_counter()-started
