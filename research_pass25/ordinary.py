import math
import numpy as np
from pbc3_heads import DownsampleInitHead
import pbc3_ops as ops
from research_pass2.stream_tools import serialize


def encode(prep,config):
    h,w,channels=prep['target'].shape
    means=[int(round(float(np.mean(prep['arr'][:,:,c])))) for c in range(channels)]
    canvas=np.empty((h,w,channels),dtype=np.int16);canvas[:]=means
    channel_bits=max(1,math.ceil(math.log2(channels)));patches=[];head=DownsampleInitHead()
    for c in range(channels):
        patch,values,delta,cell,bits=head.select(c,prep['target'],canvas,w,h,config,channel_bits)
        ops.apply_delta(canvas[:,:,c],0,0,w,h,delta);patches.append(patch)
    header=(prep['downsampled'],prep['original_w'] if prep['downsampled'] else None,prep['original_h'] if prep['downsampled'] else None,w,h,prep['color_space'],channels,channel_bits,config.positive_bias,False,channels,means,False,None,None,None)
    return serialize(header,patches)
