import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass20.common import *


def main():
    setup();rows=[]
    previous=json.loads((ROOT/'research_pass17/baseline.json').read_text())['rows']
    warm=Image.fromarray(np.random.default_rng(2003).integers(0,256,(40,32,3),dtype=np.uint8))
    PBC3.decompress(PBC3.compress(warm,config=PBC3Config.high_quality()).data)
    for name in INITIAL:
        image=load(name)
        start=time.perf_counter();data=PBC3.compress(image,config=PBC3Config.high_quality()).data;enc=time.perf_counter()-start
        start=time.perf_counter();output=PBC3.decompress(data).image.convert('RGB');dec=time.perf_counter()-start
        expected=next(r for r in previous if r['image']==name and r['preset']=='high_quality')
        assert sha(data)==expected['stream_sha256']
        rows.append(dict(image=name,encode_seconds=enc,decode_seconds=dec,stream_sha256=sha(data),bytes=len(data),rgb_mse=mse(image,output)))
        print(name,enc,dec,flush=True)
    provenance=meta();provenance['timing']='Three one-pass warmed complete HQ calls, reusing known RD controls and verifying identical bytes. Not repeated/interleaved speed evidence; failed multiscale settings receive no timing repetitions.'
    write_json('control_times.json',dict(provenance=provenance,rows=rows))


if __name__=='__main__':main()
