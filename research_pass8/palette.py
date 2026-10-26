import math

import numpy as np

import pbc3_ops as ops


def bounded_palette(small,patch,config,percentile):
    magnitudes=(-small[small<0].astype(np.int32),small[small>0].astype(np.int32))
    neg,pos=[min(255,int(math.ceil(float(np.percentile(side,percentile))))) if side.size else 0 for side in magnitudes]
    mask=ops.mask_from_values(small,len(patch['mask']),neg,pos,config.positive_bias)
    palette=ops.palette_generator(mask,patch['max_bitcount'],neg,pos,config.positive_bias)
    indices=ops.quantize_signed(np.clip(small,-neg,pos),palette)
    candidate={**patch,'mask':mask,'neg':neg,'pos':pos,'indices':indices,
        'bitcount':ops.resolve_palette_bitcount(mask,patch['max_bitcount'],neg,pos,config.positive_bias)}
    return candidate,palette[indices]


def local_sse(target,canvas,patch,values):
    c,x,y,w,h=(patch[k] for k in ('channel','x','y','w','h'))
    delta=ops.signed_resample(values,h,w)
    error=target[y:y+h,x:x+w,c]-np.clip(canvas[y:y+h,x:x+w,c]+delta,0,255)
    return int(np.sum(error.astype(np.int64)**2)),delta
