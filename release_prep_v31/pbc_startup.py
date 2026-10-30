import time

def initialize():
    started = time.perf_counter()
    import numpy as np
    from PIL import Image
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    from pbc3_native import NativeMSE, select_metric
    rng = np.random.default_rng(3101)
    for channels in (3, 4):
        image = Image.fromarray(rng.integers(0, 256, (37, 39, channels), dtype=np.uint8))
        PBC3.compress(image, config=PBC3Config.quality())
    return dict(seconds=time.perf_counter()-started,
        native=isinstance(select_metric("auto", lambda a, b: None), NativeMSE),
        signatures=signatures())



def signatures():
    import pbc3_ops,pbc3_features
    return {module.__name__+"."+name:[str(s) for s in value.signatures]
        for module in (pbc3_ops,pbc3_features) for name,value in vars(module).items()
        if hasattr(value,"signatures")}
