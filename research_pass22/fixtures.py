import lzma
import numpy as np
from pbc3_types import BitWriter
from pbc3_stream import write_header
from research_pass21 import sparse_hybrid as reference,zero_runs


def container(w,h,q,step=128,base_size=None):
    bw=BitWriter()
    bw_w,bw_h=base_size or (w,h)
    write_header(bw,bw_w,bw_h,bw_w,bw_h,False,0,3,2,True,False,0,[0,127,255])
    base=b'PBC3'+bytes([0,0])+bw.finish()
    payload=lzma.compress(zero_runs.pack(q),format=lzma.FORMAT_XZ,preset=0)
    return reference.HEADER.pack(b'P21S',w,h,len(base),len(payload),step,8,1)+base+payload


def small():
    q=np.zeros((2,2,3,8,8),dtype=np.int16)
    q[0,0,0,0,0]=1
    return container(13,9,q)
