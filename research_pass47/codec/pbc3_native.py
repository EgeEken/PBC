import ctypes as ct
from functools import lru_cache
from pathlib import Path
import numpy as np

DLL=Path(__file__).resolve().parent/'pbc3/mse.dll'

@lru_cache(maxsize=1)
def _function():
    lib=ct.CDLL(str(DLL))
    fn=lib.mse_u8
    fn.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int64,ct.c_int64,ct.c_int64,ct.POINTER(ct.c_int64),ct.POINTER(ct.c_int64),ct.POINTER(ct.c_double)]
    fn.restype=ct.c_int
    return lib,fn

class NativeMSE:
    def __init__(self,reference):
        self.reference=reference
        self.library,self.function=_function()

    def __call__(self,reference,reconstructed):
        a,b=np.asarray(reference),np.asarray(reconstructed)
        if a.dtype!=np.uint8 or b.dtype!=np.uint8 or a.shape!=b.shape or a.ndim not in (2,3):
            return self.reference(reference,reconstructed)
        if a.ndim==2: a,b=a[:,:,None],b[:,:,None]
        h,w,c=a.shape
        if not (h and w and 1<=c<=4 and h*w<=32_000_000): return self.reference(reference,reconstructed)
        sa=(ct.c_int64*3)(*a.strides);sb=(ct.c_int64*3)(*b.strides);out=ct.c_double()
        if self.function(a.ctypes.data,b.ctypes.data,h,w,c,sa,sb,ct.byref(out)):
            raise ValueError('native metric rejected dimensions')
        return out.value

def select_metric(backend,reference):
    if backend=='reference' or (backend=='auto' and not DLL.is_file()): return reference
    if backend in ('native','auto'): return NativeMSE(reference)
    raise ValueError('mse_backend must be reference, native or auto')
