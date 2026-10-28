import numpy as np
from scipy.fft import idctn
from research_pass21 import sparse_hybrid as reference
from PBC3 import PBC3


def reconstruct(q,step,h,w,sparse=False,reuse=False):
    if sparse:
        occupied=np.any(q!=0,axis=(-2,-1))
        selected=q[occupied].astype(np.float32)
        if reuse:np.multiply(selected,np.float32(step),out=selected)
        else:selected=selected*step
        inverse=idctn(selected,axes=(-2,-1),type=2,norm='ortho',workers=1,overwrite_x=reuse)
        spatial=np.zeros(q.shape,dtype=np.float32)
        spatial[occupied]=inverse
    else:
        scaled=q.astype(np.float32)
        if reuse:np.multiply(scaled,np.float32(step),out=scaled)
        else:scaled=scaled*step
        spatial=idctn(scaled,axes=(-2,-1),type=2,norm='ortho',workers=1,overwrite_x=reuse)
    by,bx=spatial.shape[:2]
    return spatial.transpose(0,3,1,4,2).reshape(by*8,bx*8,3)[:h,:w] @ reference.h17.COLOR


def decode(data,sparse=True,reuse=True):
    (w,h,step,n),base,payload=reference.split(data)
    pixels=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.uint8 if reuse else np.float32)
    if pixels.shape!=(h,w,3):raise ValueError('base dimensions')
    q=reference.read_coefficients(payload,w,h,n)
    delta=reconstruct(q,step,h,w,sparse,reuse)
    if reuse:
        np.add(delta,pixels,out=delta)
        np.rint(delta,out=delta)
        np.clip(delta,0,255,out=delta)
        return delta.astype(np.uint8)
    return np.clip(np.rint(pixels+delta),0,255).astype(np.uint8)
