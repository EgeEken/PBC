import time
import statistics
from scipy.fft import idctn
from common import *
from research_pass22 import decoder


def measure(data,reuse,sparse):
    spans={};buffers={};sharing={}
    def call(label,fn):
        start=time.perf_counter();value=fn();spans[label]=time.perf_counter()-start;return value
    def info(label,array):
        buffers[label]=dict(shape=list(array.shape),bytes=array.nbytes,dtype=str(array.dtype),strides=list(array.strides),owns=bool(array.flags.owndata),c_contiguous=bool(array.flags.c_contiguous),f_contiguous=bool(array.flags.f_contiguous))
    (w,h,step,n),base,payload=call('parse',lambda:reference.split(data))
    pixels=call('base_decode',lambda:np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.uint8 if reuse else np.float32));info('base',pixels)
    raw=call('decompress',lambda:zero_runs.decompress(payload,n))
    flat=call('unpack',lambda:zero_runs.unpack(raw,n));info('dense_int16',flat)
    q=call('view',lambda:flat.reshape(3,8,8,(h+7)//8,(w+7)//8).transpose(3,4,0,1,2))
    sharing['q_flat_view']=np.shares_memory(q,flat)
    if sparse:
        occupied=call('block_selection',lambda:np.any(q!=0,axis=(-2,-1)));info('occupancy',occupied)
        selected=call('gather',lambda:q[occupied]);info('gather_int16',selected)
        scaled=call('dequantize',lambda:selected.astype(np.float32));np_start=time.perf_counter();np.multiply(scaled,np.float32(step),out=scaled);spans['dequantize']+=time.perf_counter()-np_start
        info('selected_float',scaled)
        inverse=call('idct',lambda:idctn(scaled,axes=(-2,-1),type=2,norm='ortho',workers=1,overwrite_x=True));info('inverse',inverse)
        spatial=call('zero_output',lambda:np.zeros(q.shape,dtype=np.float32));info('spatial',spatial)
        start=time.perf_counter();spatial[occupied]=inverse;spans['scatter']=time.perf_counter()-start
    else:
        scaled=call('dequantize',lambda:q.astype(np.float32)*step);info('scaled_float',scaled)
        inverse=call('idct',lambda:idctn(scaled,axes=(-2,-1),type=2,norm='ortho',workers=1));info('inverse',inverse)
        spatial=inverse
    sharing['idct_scaled']=np.shares_memory(inverse,scaled)
    transposed=spatial.transpose(0,3,1,4,2);info('transposed_view',transposed)
    reordered=call('reorder_copy',lambda:transposed.reshape((h+7)//8*8,(w+7)//8*8,3));info('reordered',reordered)
    sharing['transpose_spatial']=np.shares_memory(transposed,spatial)
    sharing['reshape_spatial']=np.shares_memory(reordered,spatial)
    cropped=reordered[:h,:w];sharing['crop_reorder']=np.shares_memory(cropped,reordered)
    delta=call('color',lambda:cropped @ reference.h17.COLOR);info('color',delta)
    if reuse:
        start=time.perf_counter();np.add(delta,pixels,out=delta);np.rint(delta,out=delta);np.clip(delta,0,255,out=delta);output=delta.astype(np.uint8);spans['final_output']=time.perf_counter()-start
    else:output=call('final_output',lambda:np.clip(np.rint(pixels+delta),0,255).astype(np.uint8))
    info('output',output)
    return output,spans,buffers,sharing


def main():
    setup();rows=[]
    for name in CASES:
        data,record=saved(name)
        reference.decode(data);decoder.decode(data)
        for rep in range(3):
            for label in (('reference','optimized') if rep%2==0 else ('optimized','reference')):
                output,spans,buffers,sharing=measure(data,label=='optimized',label=='optimized')
                assert sha(output.tobytes())==record['pixels_sha256']
                rows.append(dict(image=name,variant=label,rep=rep,spans=spans,buffers=buffers,sharing=sharing,total_seconds=sum(spans.values())))
        print('profile',name,flush=True)
    write('profiles.json',dict(provenance=meta(),rows=rows,note='Three disjoint instrumented profiles per variant, not full-call timings. Profiling retains arrays for inspection and thus is not a memory peak measurement. Gather conversion separated from scaling; all spans included in totals. np.shares_memory measures actual buffer aliasing.'))


if __name__=='__main__':main()
