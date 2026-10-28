import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass21.common import *


def main():
    setup();rows=[]
    previous=json.loads((ROOT/'research_pass17/baseline.json').read_text())['rows']
    warm=Image.fromarray(np.random.default_rng(2003).integers(0,256,(40,32,3),dtype=np.uint8))
    PBC3.decompress(PBC3.compress(warm,config=PBC3Config.high_quality()).data)
    for name in ('mario64.png','cat.jpg','monet.jpg','flower.jpg'):
        image=load(name)
        start=time.perf_counter();encoded=PBC3.compress(image,config=PBC3Config.high_quality());enc=time.perf_counter()-start
        start=time.perf_counter();decoded=PBC3.decompress(encoded.data);dec=time.perf_counter()-start
        expected=next(r for r in previous if r['image']==name and r['preset']=='high_quality')
        assert sha(encoded.data)==expected['stream_sha256'] and sha(np.asarray(decoded.image).tobytes())==expected['pixels_sha256']
        rows.append(dict(image=name,preset='high_quality',bytes=len(encoded.data),rgb_mse=expected['rgb_mse'],encode_seconds=enc,decode_seconds=dec,stream_sha256=sha(encoded.data)))
        print(name,enc,dec,flush=True)
    write_json('other_controls.json',dict(provenance=meta(),rows=rows,timing='One warmed full call on four remaining HQ controls, identical saved file hashes. No repeated speed claim and no new RD control configurations.'))


if __name__=='__main__':main()
