import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import dataclasses
import time
if sys.argv[1]=='native':
    from kernel import install
    install()
for line in sys.stdin:
    request=json.loads(line)
    image=load(request['image']) if request['image']!='rgba' else Image.fromarray(np.random.default_rng(26).integers(0,256,(91,127,4),dtype=np.uint8))
    cfg=getattr(PBC3Config,request['preset'])()
    t=time.perf_counter();r=PBC3.compress(image,config=cfg);enc=time.perf_counter()-t
    t=time.perf_counter();decoded=PBC3.decompress(r.data);dec=time.perf_counter()-t
    print(json.dumps(dict(encode_seconds=enc,decode_seconds=dec,stream_sha256=sha(r.data),pixel_sha256=sha(decoded.image.tobytes()),bytes=len(r.data),internal_mse=r.mse,rgb_mse=mse(image.convert('RGB'),decoded.image.convert('RGB')),config=dataclasses.asdict(cfg),peak_wset=psutil.Process().memory_info().peak_wset)),flush=True)
