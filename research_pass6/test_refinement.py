import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from refinement import refine


def main():
    import numpy as np
    import pbc3_ops as ops
    import pbc3_stream as stream
    from pbc3_types import PBC3Config,BitReader,BitWriter
    rng=np.random.default_rng(2003)
    target=rng.integers(0,256,(13,17,3),dtype=np.int32)
    cfg=PBC3Config.quality()
    for base in (-32760,128,32760):
        canvas=np.full(target.shape,base,dtype=np.int16)
        palette=ops.palette_generator([1,1,1,1],4,255,255,True)
        indices=rng.integers(0,len(palette),(5,6),dtype=np.uint16)
        original_indices=indices.copy()
        patch=dict(channel=0,x=0,y=0,w=17,h=13,cell_size=3,indices=indices,
            bitcount=4,max_bitcount=4,mask=[1,1,1,1],neg=255,pos=255,palette_mode=0,palette=None)
        values=palette[indices]
        delta=ops.signed_resample(values,13,17)
        stats={}
        candidate,new_values,new_delta=refine(target,canvas,cfg,patch,values,delta,stats,0)
        for key in patch:
            if key!='indices':
                assert candidate[key]==patch[key]
        assert np.array_equal(patch['indices'],original_indices)
        assert ops.patch_bits_for(candidate,2)==ops.patch_bits_for(patch,2)
        before=(target[:,:,0]-np.clip(canvas[:,:,0]+delta,0,255)).astype(np.int64)
        applied=canvas.copy()
        ops.apply_delta(applied[:,:,0],0,0,17,13,new_delta)
        after=(target[:,:,0]-np.clip(applied[:,:,0],0,255)).astype(np.int64)
        assert np.sum(after**2)<=np.sum(before**2)
        writer=BitWriter()
        stream.write_patch(writer,candidate,2)
        decoded=stream.read_patch(BitReader(writer.finish()),2,True)
        assert np.array_equal(decoded[6],new_values)
        assert np.array_equal(ops.signed_resample(decoded[6],13,17),new_delta)
    print(json.dumps(dict(clipping_and_int16_cases=3,palette_geometry_and_bit_cost_preserved=True,normal_stream_decoder=True)))


if __name__=='__main__':
    main()
