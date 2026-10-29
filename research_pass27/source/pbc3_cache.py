from pathlib import Path
import sys
from numba.core.caching import UserProvidedCacheLocator

class FrozenCacheLocator(UserProvidedCacheLocator):
    @classmethod
    def from_function(cls,py_func,py_file):
        if not getattr(sys,'frozen',False): return None
        source=Path(sys._MEIPASS)/Path(py_file).name
        if not source.is_file(): return None
        return super().from_function(py_func,str(source))
