import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass21.common import *
from research_pass21 import sparse_hybrid as codec,zero_runs
from scipy.fft import idctn


def main():
    setup();rows=[]
    measured=json.loads((HERE/'files.json').read_text())['rows']
    for name,step in (('googlewiki.png',128),('eiffel.png',32),('cat.jpg',128)):
        row=next(r for r in measured if r['image']==name)
        data=(HERE/row['file']).read_bytes();codec.decode(data)
        spans={}
        def call(label,fn):
            start=time.perf_counter();value=fn();spans[label]=time.perf_counter()-start;return value
        params,base,payload=call('parse_container',lambda:codec.split(data))
        w,h,step,n=params
        pixels=call('base_decode_to_float',lambda:np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32))
        raw=call('enhancement_decompress',lambda:zero_runs.decompress(payload,n))
        flat=call('sparse_unpack',lambda:zero_runs.unpack(raw,n))
        q=call('coefficient_view',lambda:flat.reshape(3,8,8,(h+7)//8,(w+7)//8).transpose(3,4,0,1,2))
        scaled=call('dequant_float_copy',lambda:q.astype(np.float32)*step)
        inverse=call('inverse_dct',lambda:idctn(scaled,axes=(-2,-1),type=2,norm='ortho',workers=1))
        color=call('color_crop_reconstruction',lambda:inverse.transpose(0,3,1,4,2).reshape((h+7)//8*8,(w+7)//8*8,3)[:h,:w] @ h17.COLOR)
        output=call('final_round_clip',lambda:np.clip(np.rint(pixels+color),0,255).astype(np.uint8))
        assert sha(output.tobytes())==row['pixels_sha256']
        recovery=spans['enhancement_decompress']+spans['sparse_unpack']+spans['coefficient_view']
        residual_work=sum(spans[k] for k in ('dequant_float_copy','inverse_dct','color_crop_reconstruction','final_round_clip'))
        rows.append(dict(image=name,step=step,spans=spans,coefficient_recovery_seconds=recovery,reconstruction_excluding_recovery_seconds=spans['base_decode_to_float']+residual_work,enhancement_reconstruction_seconds=residual_work,profile_total_seconds=sum(spans.values()),coefficient_output_bytes=flat.nbytes,sparse_raw_bytes=len(raw),process_peak_working_set=psutil.Process().memory_info().peak_wset))
        print(name,spans,flush=True)
    write_json('decode_profile.json',dict(provenance=meta(),rows=rows,timing='One warmed disjoint decoder profile per case; does not replace complete-call timing. Reconstruction floor excludes coefficient recovery for this unchanged implementation, not a theoretical lower bound on all possible decoders.'))


if __name__=='__main__':main()
