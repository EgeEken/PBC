import ctypes as ct
from pathlib import Path
import numpy as np

LIB=ct.CDLL(str(Path(__file__).resolve().parents[1]/'artifacts/mse.dll'))
_FN=LIB.mse_u8
_FN.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int64,ct.c_int64,ct.c_int64,ct.POINTER(ct.c_int64),ct.POINTER(ct.c_int64),ct.POINTER(ct.c_double)]
_FN.restype=ct.c_int

def mse_u8(reference,reconstructed):
    a,b=np.asarray(reference),np.asarray(reconstructed)
    if a.dtype!=np.uint8 or b.dtype!=np.uint8: raise TypeError('native metric requires uint8 pixels')
    if a.shape!=b.shape or a.ndim not in (2,3): raise ValueError('matching 2D or 3D image arrays required')
    if a.ndim==2: a,b=a[:,:,None],b[:,:,None]
    h,w,c=a.shape
    if not (h and w and 1<=c<=4 and h*w<=32_000_000): raise ValueError('invalid metric dimensions')
    sa=(ct.c_int64*3)(*a.strides); sb=(ct.c_int64*3)(*b.strides); result=ct.c_double()
    if _FN(a.ctypes.data,b.ctypes.data,h,w,c,sa,sb,ct.byref(result)): raise ValueError('native metric rejected dimensions')
    return result.value

def install():
    import pbc3_ops as ops
    original=ops.final_mse
    def final_mse(reference,reconstructed):
        a,b=np.asarray(reference),np.asarray(reconstructed)
        if a.dtype==np.uint8 and b.dtype==np.uint8: return mse_u8(a,b)
        return original(reference,reconstructed)
    ops.final_mse=final_mse
    return original
