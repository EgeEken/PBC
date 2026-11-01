from __future__ import annotations
import lzma
import math
import struct
import time

import numpy as np
import pbc33_codec as p33

MAGIC=b'P34S'
VERSION=1
HEADER=struct.Struct('<4sBIIHBBBBBIII')
BAND_IDS={'F':0,'L':1,'H':2}
BAND_NAMES={value:key for key,value in BAND_IDS.items()}


def eligible(u:int,v:int,band:str)->bool:
    return band=='F' or (band=='L' and u+v<4) or (band=='H' and u+v>=4)


def _block_options(source:np.ndarray,base:np.ndarray,step:int,band:str)->list[tuple[int,list[tuple[int,int]]]]:
    h,w=source.shape[:2]
    if h<=0 or w<=0: return [(0,[])]
    residual=np.zeros((8,8,3),np.float64)
    residual[:h,:w]=source.astype(np.float64)-base.astype(np.float64)
    coeff=p33.dctn(residual,axes=(0,1),type=2,norm='ortho',workers=1)
    quantized=np.rint(coeff/step).astype(np.int32)
    if np.any(quantized < -32768) or np.any(quantized > 32767): raise ValueError('quantized coefficient exceeds int16')
    base64=base.astype(np.float64);source64=source.astype(np.int32)
    base_sse=int(np.square(source64-base.astype(np.int32),dtype=np.int64).sum())
    proposals=[]
    for channel in range(3):
        for u in range(8):
            for v in range(8):
                value=int(quantized[u,v,channel])
                if value and eligible(u,v,band):
                    position=channel*64+u*8+v
                    record=p33._coefficient_record(position,value)
                    energy=float(np.square(p33._BASIS[u*8+v,:h,:w]*(step*value)).sum())
                    proposals.append((energy/(8*len(record)),position,value))
    proposals.sort(key=lambda item:(-item[0],item[1]))
    shortlist=proposals[:p33.SHORTLIST]
    measured=[]
    for _,position,value in shortlist:
        channel,rem=divmod(position,64);u,v=divmod(rem,8)
        corrected=np.clip(np.rint(base64[:,:,channel]+p33._BASIS[u*8+v,:h,:w]*(step*value)),0,255)
        test=source64.copy();test[:,:,channel]=corrected.astype(np.int32)
        gain=base_sse-int(np.square(source64-test,dtype=np.int64).sum())
        if gain>0: measured.append((gain/(8*len(p33._coefficient_record(position,value))),position,value))
    measured.sort(key=lambda item:(-item[0],item[1]))
    selected=measured[:p33.MAX_BLOCKS]
    options=[(base_sse,[])];selected_rows=[]
    for _,position,value in selected:
        channel,rem=divmod(position,64);u,v=divmod(rem,8)
        selected_rows.append((position,value))
        delta=np.zeros((8,8,3),np.float64)
        for selected_position,selected_value in selected_rows:
            selected_channel,selected_rem=divmod(selected_position,64)
            selected_u,selected_v=divmod(selected_rem,8)
            delta[:,:,selected_channel]+=p33._BASIS[selected_u*8+selected_v]*(step*selected_value)
        rec=np.clip(np.rint(base64+delta[:h,:w]),0,255).astype(np.int32)
        sse=int(np.square(source64-rec,dtype=np.int64).sum())
        options.append((sse,sorted(selected_rows)))
    return options


def _wrap_p33(data:bytes,band:str)->bytes:
    old=p33.HEADER.unpack_from(data)
    magic,version,width,height,cell,dc_step,detail_step,qstep,mode_bits,n,raw_len,xz_len=old
    if magic!=p33.MAGIC or len(data)!=p33.HEADER.size+xz_len: raise ValueError('invalid P33 input')
    new=HEADER.pack(MAGIC,VERSION,width,height,cell,dc_step,detail_step,qstep,BAND_IDS[band],mode_bits,n,raw_len,xz_len)
    return new+data[p33.HEADER.size:]


def encode_family(source:np.ndarray,base:np.ndarray,lambdas=(400.0,1600.0),qstep:int=32,band:str='F')->list[dict]:
    if band not in BAND_IDS: raise ValueError('unknown frequency band')
    if qstep not in p33.QUANT_STEPS: raise ValueError('invalid sparse quantization step')
    original=p33._block_options
    p33._block_options=lambda src,b,step:_block_options(src,b,step,band)
    try: rows=p33.encode_family(source,base,'S',lambdas,qstep)
    finally: p33._block_options=original
    for row in rows:
        row['p33_data']=row['data']
        row['data']=_wrap_p33(row['data'],band)
        row['band']=band
        row['header_bytes']=HEADER.size
    return rows


def _split(data:bytes):
    if len(data)<HEADER.size: raise ValueError('short P34S header')
    fields=HEADER.unpack_from(data)
    magic,version,width,height,cell,dc_step,detail_step,qstep,band_id,mode_bits,n,raw_len,xz_len=fields
    if magic!=MAGIC or version!=VERSION or band_id not in BAND_NAMES: raise ValueError('invalid P34S header or band')
    if len(data)!=HEADER.size+xz_len: raise ValueError('P34S size mismatch')
    p33header=p33.HEADER.pack(p33.MAGIC,version,width,height,cell,dc_step,detail_step,qstep,mode_bits,n,raw_len,xz_len)
    return fields,p33header+data[HEADER.size:]


def decode(data:bytes,base:np.ndarray,profile:dict|None=None):
    fields,p33data=_split(data)
    band=BAND_NAMES[fields[8]]
    original=p33.idctn
    def checked_idct(coeff,*args,**kwargs):
        if band!='F':
            outside=np.zeros((8,8),bool)
            for u in range(8):
                for v in range(8):
                    if not eligible(u,v,band): outside[u,v]=True
            if np.any(coeff[outside,:]!=0): raise ValueError('coefficient violates signaled frequency band')
        return original(coeff,*args,**kwargs)
    p33.idctn=checked_idct
    try: return p33.decode(p33data,base,profile)
    finally: p33.idctn=original


def unpack_raw(data:bytes)->tuple[dict,bytes]:
    fields,_=_split(data)
    magic,version,width,height,cell,dc_step,detail_step,qstep,band_id,mode_bits,n,raw_len,xz_len=fields
    decomp=lzma.LZMADecompressor(format=lzma.FORMAT_XZ,memlimit=64*1024*1024)
    raw=decomp.decompress(data[HEADER.size:],max_length=raw_len+1)
    if not decomp.eof or decomp.unused_data or len(raw)!=raw_len: raise ValueError('invalid P34S XZ payload')
    return {'width':width,'height':height,'qstep':qstep,'band':BAND_NAMES[band_id],'mode_bits':mode_bits,'n':n,'raw_len':raw_len},raw


def wrap_raw(width:int,height:int,qstep:int,band:str,modes:np.ndarray,params:bytes)->bytes:
    if band not in BAND_IDS or qstep not in p33.QUANT_STEPS: raise ValueError('invalid counterfactual metadata')
    packed=p33.pack_modes(modes);raw=packed+params
    xz=lzma.compress(raw,format=lzma.FORMAT_XZ,preset=6)
    n=len(modes)
    return HEADER.pack(MAGIC,VERSION,width,height,p33.CELL,p33.DC_STEP,p33.DETAIL_STEP,qstep,BAND_IDS[band],p33.MODE_BITS,n,len(raw),len(xz))+xz


def counterfactual_remove(data:bytes,remove_band:str)->bytes:
    if remove_band not in ('L','H'): raise ValueError('counterfactual must remove L or H')
    meta,raw=unpack_raw(data)
    if meta['band']!='F': raise ValueError('counterfactual reference must be Full spectrum')
    mode_len=math.ceil(meta['n']*p33.MODE_BITS/8)
    modes=p33.unpack_modes(raw[:mode_len],meta['n']).copy();params=raw[mode_len:]
    out=bytearray();cursor=0
    for index,mode_v in enumerate(modes):
        mode=int(mode_v)
        if mode==1:
            out.extend(params[cursor:cursor+3]);cursor+=3
        elif mode==2:
            old_mask=params[cursor];cursor+=1
            new_mask=0;blocks=[]
            for block in range(4):
                if not old_mask&(1<<block): continue
                count=params[cursor];cursor+=1;kept=[]
                for _ in range(count):
                    pos=params[cursor];cursor+=1;value_start=cursor
                    encoded,cursor=p33._read_varint(params,cursor)
                    channel,rem=divmod(pos,64);u,v=divmod(rem,8)
                    if not eligible(u,v,remove_band):
                        kept.append((pos,params[value_start:cursor]))
                if kept:
                    new_mask|=1<<block;blocks.append((count,kept))
            if new_mask:
                out.append(new_mask)
                for _,kept in blocks:
                    out.append(len(kept))
                    for pos,value_bytes in kept:
                        out.append(pos);out.extend(value_bytes)
            else: modes[index]=0
        elif mode!=0: raise ValueError('unexpected mode in sparse Full stream')
    if cursor!=len(params): raise ValueError('counterfactual parse mismatch')
    keep='H' if remove_band=='L' else 'L'
    return wrap_raw(meta['width'],meta['height'],meta['qstep'],keep,modes,bytes(out))


def accounting(data:bytes)->dict:
    meta,raw=unpack_raw(data)
    mode_len=math.ceil(meta['n']*p33.MODE_BITS/8)
    modes=p33.unpack_modes(raw[:mode_len],meta['n']);params=raw[mode_len:]
    cursor=0;dc=mask_bytes=count_bytes=position_bytes=value_bytes=0;blocks=coefficients=active_cells=0
    freq=[0]*15
    for mode_v in modes:
        mode=int(mode_v)
        if mode==0: continue
        active_cells+=1
        if mode==1: dc+=3;cursor+=3
        elif mode==2:
            mask=params[cursor];cursor+=1;mask_bytes+=1
            for block in range(4):
                if mask&(1<<block):
                    count=params[cursor];cursor+=1;count_bytes+=1;blocks+=1
                    for _ in range(count):
                        pos=params[cursor];cursor+=1;position_bytes+=1
                        _,end=p33._read_varint(params,cursor);value_bytes+=end-cursor;cursor=end
                        _,rem=divmod(pos,64);u,v=divmod(rem,8);freq[u+v]+=1;coefficients+=1
        else: raise ValueError('invalid sparse mode accounting')
    if cursor!=len(params): raise ValueError('parameter accounting mismatch')
    raw_len=len(raw);xz_len=len(data)-HEADER.size
    return {'band':meta['band'],'qstep':meta['qstep'],'lambda':None,'base_bytes':None,'enhancement_bytes':len(data),
        'header_bytes':HEADER.size,'mode_bytes':mode_len,'dc_bytes':dc,'cell_mode_counts':{str(int(m)):int(np.count_nonzero(modes==m)) for m in (0,1,2)},
        'block_mask_bytes':mask_bytes,'block_count_bytes':count_bytes,'coefficient_position_bytes':position_bytes,
        'coefficient_value_varint_bytes':value_bytes,'raw_parameter_bytes':len(params),'raw_payload_bytes':raw_len,
        'xz_payload_bytes':xz_len,'xz_vs_raw_delta_bytes':xz_len-raw_len,'active_cells':active_cells,
        'active_blocks':blocks,'coefficients':coefficients,'mean_coefficients_per_active_block':coefficients/blocks if blocks else 0,
        'frequency_diagonal_counts':freq}
