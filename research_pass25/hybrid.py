import lzma
import struct
import time
import numpy as np
from PBC3 import PBC3
from pbc3_heads import FillerHead
from learned_filler import LearnedFiller
from pbc3_types import PBC3Result,BitWriter,BitReader
import pbc3_ops as ops
from research_pass25 import base

HEADER=struct.Struct('<4sIIBI')
MAX_PATCHES=128
MAX_RAW_BYTES=16777216


def frame(initial,patches):
    writer=BitWriter()
    for patch in patches:PBC3._write_patch(writer,patch,2)
    raw=writer.finish();method,payload=PBC3._entropy_pack(raw)
    return HEADER.pack(b'P25B',len(initial),len(raw),method,len(patches))+initial+payload,len(raw),len(payload)


def split(data):
    if not HEADER.size<=len(data)<=base.MAX_FILE_BYTES:raise ValueError('hybrid file resource/length')
    magic,nb,nraw,method,count=HEADER.unpack_from(data)
    if magic!=b'P25B' or nb<base.HEADER.size or nb>len(data)-HEADER.size or nraw>MAX_RAW_BYTES or count>MAX_PATCHES or method not in (0,2):raise ValueError('hybrid framing')
    initial=data[HEADER.size:HEADER.size+nb];payload=data[HEADER.size+nb:]
    if method==0:raw=payload
    else:
        reader=lzma.LZMADecompressor(format=lzma.FORMAT_RAW,filters=PBC3._LZMA_FILTERS)
        raw=reader.decompress(payload,max_length=nraw+1)
        if not reader.eof or reader.unused_data:raise ValueError('patch entropy framing')
    if len(raw)!=nraw or (not count and nraw):raise ValueError('patch raw length')
    return initial,raw,count


def decode_canvas(data):
    initial,raw,count=split(data);canvas,info=base.decode_canvas(initial)
    reader=BitReader(raw)
    for _ in range(count):
        try:c,x,y,w,h,cell,values,bits=PBC3._read_patch(reader,2,True)
        except ZeroDivisionError as error:raise ValueError('invalid cell size') from error
        if not 0<=c<3 or min(w,h,cell)<1 or x+w>info['w'] or y+h>info['h']:raise ValueError('patch outside working canvas')
        canvas[y:y+h,x:x+w,c]+=ops.signed_resample(values,h,w)
    padding=len(raw)*8-(reader.i*8-reader.nbits)
    if not 0<=padding<8 or reader.acc:raise ValueError('trailing patch data')
    return canvas,info


def decode(data):
    canvas,info=decode_canvas(data)
    return base.render(canvas,info)


def encode(image,config,step=128):
    start=time.perf_counter();prep=PBC3.prepare(image,config)
    if config.warmup_ratio>0 or config.patch_count<0 or config.patch_count>MAX_PATCHES or not config.positive_bias:raise ValueError('unsupported research configuration')
    initial=base.encode(prep,step);canvas,info=base.decode_canvas(initial)
    target=prep['target'];h,w,channels=target.shape;patches=[];policy_class=None
    if config.patch_count:
        state=ops.SearchState();scores=[state.score(target,canvas,c) for c in range(channels)]
        filler=FillerHead(config,2,(h,w,channels),[],(prep['original_w'],prep['original_h']))
        if type(filler.learned) is not LearnedFiller or hasattr(filler.learned,'begin_encode'):raise ValueError('requires verified legacy no-history policy')
        policy_class=type(filler.learned).__name__;rng=np.random.default_rng(config.random_seed)
        for index in range(1,config.patch_count+1):
            channel=PBC3._choose_channel(scores,index,channels,config.channel_cycle)
            patch,values,delta=filler.select(target,canvas,config,rng,2,index,channel,None,len(patches),[],abs_error=state.error(target,canvas,channel))
            if patch is None:break
            c=patch['channel'];ops.apply_delta(canvas[:,:,c],patch['x'],patch['y'],patch['w'],patch['h'],delta);patches.append(patch)
            state.update(target,canvas,c,patch['x'],patch['y'],patch['w'],patch['h']);scores[c]=state.score(target,canvas,c)
            if config.quality_target_mae>0 and max(scores)/(w*h)<=config.quality_target_mae:break
    data,raw_bytes,packed_bytes=frame(initial,patches)
    output=base.render(canvas,info);mse=ops.final_mse(prep['orig_compare'],output)
    result=PBC3Result(output,data,config,mse,time.perf_counter()-start,len(data)*8,prep['original_w'],prep['original_h'],w,h,channels=3)
    details=dict(base_bytes=len(initial),base_header_bytes=base.HEADER.size,spatial_raw_bytes=raw_bytes,spatial_payload_bytes=packed_bytes,hybrid_header_bytes=HEADER.size,spatial_patches=len(patches),step=step,policy_class=policy_class,working_component_mse=float(np.mean((target.astype(np.float64)-np.clip(canvas,0,255))**2)))
    return result,details
