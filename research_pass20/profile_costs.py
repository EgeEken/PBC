import gc
import lzma
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass20.common import *
from scipy.fft import dctn,idctn


def main():
    setup();ARTIFACTS.mkdir(exist_ok=True)
    provenance=meta();provenance['timing']='Disjoint instrumented spans plus separate complete calls. Entropy-only three interleaved repetitions on identical bytes. Full profiles one warmed call each; not repeated whole-codec speed claims.'
    warm=Image.fromarray(np.random.default_rng(2003).integers(0,256,(40,32,3),dtype=np.uint8))
    h17.decode(h17.encode(warm,128))
    records=[]
    for name in INITIAL:
        image=load(name);spans={}
        def call(label,fn):
            start=time.perf_counter();value=fn();spans[label]=time.perf_counter()-start;return value
        config=PBC3Config.quality(patch_count=10)
        base=call('base_encode',lambda:PBC3.compress(image,config=config).data)
        assert base==frozen_base(name)
        pixels=call('base_decode_inside_encode',lambda:np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32))
        residual=call('residual',lambda:np.asarray(image,dtype=np.float32)-pixels)
        color=call('color_transform',lambda:residual @ h17.COLOR.T)
        coeff=call('dct',lambda:dctn(blocks(color),axes=(-2,-1),type=2,norm='ortho',workers=1))
        quantized=call('quantization',lambda:h17.quantize(coeff,128))
        raw=call('reorder_copy',lambda:quantized.transpose(2,3,4,0,1).astype('<i2',copy=False).tobytes())
        payload=call('xz_extreme',lambda:lzma.compress(raw,format=lzma.FORMAT_XZ,preset=lzma.PRESET_EXTREME))
        data=call('container',lambda:h17.HEADER.pack(h17.MAGIC,image.width,image.height,len(base),len(payload),128,8,0,0)+base+payload)
        saved=(ROOT/'research_pass17/artifacts'/(name+'.dct_q128.p17')).read_bytes()
        assert data==saved
        call('decode_base',lambda:PBC3.decompress(base))
        recovered=call('decode_entropy',lambda:lzma.decompress(payload,format=lzma.FORMAT_XZ))
        by,bx=quantized.shape[:2]
        readback=call('decode_view',lambda:np.frombuffer(recovered,dtype='<i2').reshape(3,8,8,by,bx).transpose(3,4,0,1,2))
        scaled=call('decode_dequant',lambda:readback.astype(np.float32)*128)
        inverse=call('decode_idct',lambda:idctn(scaled,axes=(-2,-1),type=2,norm='ortho',workers=1))
        rgb=call('decode_color_crop',lambda:inverse.transpose(0,3,1,4,2).reshape(by*8,bx*8,3)[:image.height,:image.width] @ h17.COLOR)
        output=call('decode_round_clip',lambda:np.clip(np.rint(pixels+rgb),0,255).astype(np.uint8))
        assert np.array_equal(output,h17.decode(data))
        start=time.perf_counter();full=h17.encode(image,128);enc=time.perf_counter()-start
        start=time.perf_counter();out=h17.decode(full);dec=time.perf_counter()-start
        assert full==saved and np.array_equal(out,output)
        entropy=[]
        for rep in range(3):
            for setting in (('extreme','normal') if rep%2==0 else ('normal','extreme')):
                preset=lzma.PRESET_EXTREME if setting=='extreme' else 0
                start=time.perf_counter();packed=lzma.compress(raw,format=lzma.FORMAT_XZ,preset=preset);elapsed=time.perf_counter()-start
                start=time.perf_counter();unpacked=lzma.decompress(packed,format=lzma.FORMAT_XZ);decode=time.perf_counter()-start
                assert unpacked==raw
                entropy.append(dict(rep=rep,setting=setting,bytes=len(packed),encode_seconds=elapsed,decode_seconds=decode,process_peak_working_set=psutil.Process().memory_info().peak_wset))
        summary={setting:{key:statistics.median(r[key] for r in entropy if r['setting']==setting) for key in ('bytes','encode_seconds','decode_seconds')} for setting in ('extreme','normal')}
        u,v=np.indices((8,8));low=u+v<4
        energy=np.sum(coeff.astype(np.float64)**2,axis=(0,1,2))
        bands=[]
        for kind,keep in [('low_only',low),('high_only',~low)]:
            q=quantized.copy();q[:,:,:,~keep]=0
            packed=h17.pack_coefficients(q)
            candidate=h17.HEADER.pack(h17.MAGIC,image.width,image.height,len(base),len(packed),128,8,0,0)+base+packed
            reconstructed=h17.decode(candidate)
            path=ARTIFACTS/(name+'.'+kind+'.p17');path.write_bytes(candidate)
            bands.append(dict(kind=kind,bytes=len(candidate),enhancement_bytes=len(packed),rgb_mse=mse(image,reconstructed),stream_sha256=sha(candidate),pixels_sha256=sha(reconstructed.tobytes())))
        (ARTIFACTS/(name+'.q128.npy')).write_bytes(b'')
        np.save(ARTIFACTS/(name+'.q128.npy'),quantized)
        records.append(dict(image=name,spans=spans,full_encode_seconds=enc,full_decode_seconds=dec,full_bytes=len(data),full_mse=mse(image,output),entropy=entropy,entropy_summary=summary,zero_fraction=float(np.mean(quantized==0)),low_band_energy_fraction=float(energy[low].sum()/energy.sum()),high_band_energy_fraction=float(energy[~low].sum()/energy.sum()),low_band_nonzeros=int(np.count_nonzero(quantized[:,:,:,low])),high_band_nonzeros=int(np.count_nonzero(quantized[:,:,:,~low])),raw_coefficient_bytes=len(raw),band_repacks=bands,stream_sha256=sha(data),process_peak_working_set=psutil.Process().memory_info().peak_wset))
        print(name,'spans',spans,'entropy',summary,'bands',[(r['kind'],r['bytes'],round(r['rgb_mse'],2)) for r in bands],flush=True)
        del coeff,quantized,raw,pixels,residual,color,recovered,readback,scaled,inverse,rgb
        gc.collect()
    write_json('costs.json',dict(provenance=provenance,rows=records))
    meta()


if __name__=='__main__':main()
