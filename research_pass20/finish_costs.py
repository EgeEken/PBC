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
    setup();rows=[]
    def pack_normal(q):
        return lzma.compress(q.transpose(2,3,4,0,1).astype('<i2',copy=False).tobytes(),format=lzma.FORMAT_XZ,preset=0)
    original_pack=h17.pack_coefficients
    h17.pack_coefficients=pack_normal
    try:
        warm=Image.fromarray(np.random.default_rng(2003).integers(0,256,(40,32,3),dtype=np.uint8))
        h17.decode(h17.encode(warm,128))
        for name in INITIAL:
            image=load(name)
            start=time.perf_counter();data=h17.encode(image,128);enc=time.perf_counter()-start
            start=time.perf_counter();output=h17.decode(data);dec=time.perf_counter()-start
            frozen=(ROOT/'research_pass17/artifacts'/(name+'.dct_q128.p17')).read_bytes()
            assert np.array_equal(output,h17.decode(frozen))
            (ARTIFACTS/(name+'.normal0.p17')).write_bytes(data)
            base=frozen_base(name)
            rows.append(dict(image=name,bytes=len(data),bpp=8*len(data)/(image.width*image.height),rgb_mse=mse(image,output),encode_seconds=enc,decode_seconds=dec,stream_sha256=sha(data),pixels_sha256=sha(output.tobytes()),base_bytes=len(base),enhancement_bytes=len(data)-len(base)-h17.HEADER.size,container_bytes=h17.HEADER.size,process_peak_working_set=psutil.Process().memory_info().peak_wset))
            print('normal full',name,len(data),round(enc,3),round(dec,3),flush=True)
    finally:h17.pack_coefficients=original_pack
    diagnostic=[]
    for name in INITIAL:
        image=load(name)
        data=(ARTIFACTS/(name+'.p10.lean.p20')).read_bytes()
        params,base,qc,qf=codec.split(data)
        w,h,cw,ch,sc,sf=params
        pixels=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32)
        coarse=codec.resize_float(h17.correction(qc,sc,ch,cw),w,h)
        remaining=np.asarray(image,dtype=np.float32)-pixels-coarse
        coeff=h17.coefficients(remaining)
        all_q=h17.quantize(coeff,sf)
        high_q=all_q.copy();high_q[:,:,:,~codec.BAND]=0
        assert np.array_equal(high_q,qf)
        energy=np.sum(coeff.astype(np.float64)**2,axis=(0,1,2))
        output=np.clip(np.rint(pixels+coarse+h17.correction(all_q,sf,h,w)),0,255).astype(np.uint8)
        diagnostic.append(dict(image=name,omitted_remaining_energy_fraction=float(energy[~codec.BAND].sum()/energy.sum()),omitted_quantized_nonzeros=int(np.count_nonzero(all_q[:,:,:,~codec.BAND])),coded_quantized_nonzeros=int(np.count_nonzero(high_q)),unencoded_all_band_rgb_mse=mse(image,output),coded_multiscale_rgb_mse=mse(image,codec.decode(data)),warning='The all-band output is a reconstruction diagnostic only; no file or rate claim, and not a third screened configuration.'))
    metadata=meta();metadata['timing']='Normal-preset completed controls: one warmed full encode/decode each. Unencoded omitted-band diagnosis is outside timers. No repeated or production-speed claim.'
    write_json('cost_details.json',dict(provenance=metadata,normal_full_rows=rows,omitted_band=diagnostic))
    print(diagnostic,flush=True)


if __name__=='__main__':main()
