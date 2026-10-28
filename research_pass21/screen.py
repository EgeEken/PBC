import json
import lzma
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass21.common import *
from research_pass21 import sparse_hybrid as codec,zero_runs


def main():
    setup();ARTIFACTS.mkdir(exist_ok=True)
    extension='--extend' in sys.argv
    names=EXTRA if extension else INITIAL
    previous=json.loads((HERE/'files.json').read_text())['rows'] if extension else []
    profile=meta()
    warm=Image.fromarray(np.random.default_rng(2003).integers(0,256,(40,32,3),dtype=np.uint8))
    codec.decode(codec.encode(warm,128))
    rows=previous
    for name,step in names:
        image=load(name);old,old_base,expected=frozen(name,step)
        start=time.perf_counter();data=codec.encode(image,step);enc=time.perf_counter()-start
        path=ARTIFACTS/(name+f'.q{step}.p21');path.write_bytes(data)
        stored=path.read_bytes()
        start=time.perf_counter();output=codec.decode(stored);dec=time.perf_counter()-start
        params,base,payload=codec.split(stored);w,h,stored_step,n=params
        assert base==old_base and stored_step==step and sha(output.tobytes())==expected['pixels_sha256']
        q=codec.read_coefficients(payload,w,h,n)
        assert np.array_equal(q,coefficients(name,step))
        error=np.asarray(image,dtype=np.float32)-output.astype(np.float32)
        rgb_mse=float(np.mean(error**2,dtype=np.float64))
        assert rgb_mse==expected['rgb_mse']
        raw=zero_runs.decompress(payload,n)
        dense=q.transpose(2,3,4,0,1).astype('<i2',copy=False).tobytes()
        fast=lzma.compress(dense,format=lzma.FORMAT_XZ,preset=0)
        fast_file=h17.HEADER.pack(h17.MAGIC,w,h,len(base),len(fast),step,8,0,0)+base+fast
        fast_path=ARTIFACTS/(name+f'.q{step}.dense_normal.p17');fast_path.write_bytes(fast_file)
        assert np.array_equal(h17.decode(fast_file),output)
        row=dict(image=name,step=step,bytes=len(data),bpp=8*len(data)/(w*h),rgb_mse=rgb_mse,encode_seconds=enc,decode_seconds=dec,base_bytes=len(base),enhancement_bytes=len(payload),container_bytes=codec.HEADER.size,enhancement_fraction=len(payload)/len(data),sparse_raw_bytes=len(raw),coefficient_count=n,nonzero_count=int(np.count_nonzero(q)),stream_sha256=sha(data),pixels_sha256=sha(output.tobytes()),base_sha256=sha(base),file=str(path.relative_to(HERE)),frozen_bytes=len(old),frozen_enhancement_bytes=expected['enhancement_bytes'],dense_normal_bytes=len(fast_file),dense_normal_enhancement_bytes=len(fast),dense_normal_sha256=sha(fast_file),process_peak_working_set=psutil.Process().memory_info().peak_wset)
        assert row['base_bytes']+row['enhancement_bytes']+row['container_bytes']==len(data)
        rows.append(row)
        print(name,step,'bytes',len(data),'old',len(old),'payload change',round(100*(len(payload)/expected['enhancement_bytes']-1),2),'encode/decode',round(enc,3),round(dec,3),flush=True)
    write_json('files.json',dict(provenance=profile,extended=extension,rows=rows,timing='One warmed full deterministic encode/decode per sparse file; includes fresh p10 base encoding, base decode inside encode, all transforms, sparse pack and XZ. File I/O, validation, hashes and external RGB scoring excluded. Dense-normal files reused/repacked for RD outside timers.'))
    meta()


if __name__=='__main__':main()
