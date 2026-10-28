import json
import lzma
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass20.common import *
from research_pass20 import multiscale as codec


def main():
    setup();ARTIFACTS.mkdir(exist_ok=True)
    extension='--extend' in sys.argv
    names=EXTRA if extension else INITIAL
    previous=json.loads((HERE/'screen.json').read_text())['rows'] if extension else []
    provenance=meta();provenance['timing']='One warmed complete compress/decompress call per candidate. Base freshly encoded inside timed call for actual full latency; p10 base bytes verified against the reused frozen control. No instrumentation/scoring/hash/I/O included.'
    warm=Image.fromarray(np.random.default_rng(2003).integers(0,256,(40,32,3),dtype=np.uint8))
    codec.decode(codec.encode(warm,'lean',5));codec.decode(codec.encode(warm,'rich',10))
    rows=previous
    for name in names:
        image=load(name)
        bases={}
        for budget in (10,5):
            for setting in ('lean','rich'):
                start=time.perf_counter();data=codec.encode(image,setting,budget);enc=time.perf_counter()-start
                path=ARTIFACTS/(name+f'.p{budget}.{setting}.p20');path.write_bytes(data)
                stored=path.read_bytes()
                start=time.perf_counter();output=codec.decode(stored);dec=time.perf_counter()-start
                checked,parts,coarse_delta,fine_delta,pixels=codec.decode(stored,details=True)
                assert np.array_equal(output,checked)
                params,base,qc,qf=codec.split(stored)
                if budget==10:assert base==frozen_base(name)
                if budget in bases:assert base==bases[budget]
                bases[budget]=base
                first,second=codec.layer_bytes(qc,qf)
                coarse_prefix=lzma.compress(first,format=lzma.FORMAT_XZ,preset=0)
                remaining=np.asarray(image,dtype=np.float32)-pixels-coarse_delta
                row=dict(image=name,base_budget=budget,setting=setting,bytes=len(data),bpp=8*len(data)/(image.width*image.height),rgb_mse=mse(image,output),encode_seconds=enc,decode_seconds=dec,stream_sha256=sha(data),pixels_sha256=sha(output.tobytes()),base_sha256=sha(base),coarse_prefix_xz_bytes=len(coarse_prefix),fine_incremental_xz_bytes=parts['shared_enhancement_bytes']-len(coarse_prefix),coarse_only_rgb_mse=mse(image,np.clip(np.rint(pixels+coarse_delta),0,255).astype(np.uint8)),remaining_residual_mse=float(np.mean(remaining**2,dtype=np.float64)),coarse_zero_fraction=float(np.mean(qc==0)),fine_retained_zero_fraction=float(np.mean(qf[:,:,:,codec.BAND]==0)),process_peak_working_set=psutil.Process().memory_info().peak_wset,file=str(path.relative_to(HERE)),**parts)
                assert row['base_bytes']+row['shared_enhancement_bytes']+row['container_bytes']==len(data)
                assert row['coarse_prefix_xz_bytes']+row['fine_incremental_xz_bytes']==row['shared_enhancement_bytes']
                rows.append(row)
                print(name,budget,setting,row['bytes'],round(row['rgb_mse'],3),round(enc,3),round(dec,3),flush=True)
    write_json('screen.json',dict(provenance=provenance,rows=rows))
    meta()


if __name__=='__main__':main()
