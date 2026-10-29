import io
import struct
import zlib
import numpy as np
from PIL import Image
from PBC3 import PBC3
from pbc3_types import PBC3Config

HEADER=struct.Struct('<4sB6H')
RECORD=struct.Struct('<BII')
MAX_BYTES=64*1024*1024

def inflate(payload,size):
    d=zlib.decompressobj()
    raw=d.decompress(payload,size+1)
    if len(raw)!=size or not d.eof or d.unused_data or d.unconsumed_tail:
        raise ValueError('invalid residual size or compressed framing')
    return raw

def encode(frames,mode='temporal',interval=8,fps=12):
    if mode not in ('intra','temporal','lossless'): raise ValueError('unknown video mode')
    frames=[np.asarray(f,dtype=np.uint8) for f in frames]
    if not frames or len(frames)>512 or not 1<=interval<=512 or not 1<=fps<=240: raise ValueError('invalid frame count or timing')
    h,w,c=frames[0].shape
    if c!=3 or w*h>2_073_600 or w*h*len(frames)>32_000_000: raise ValueError('invalid clip dimensions')
    parts=[HEADER.pack(b'P26V',1,w,h,len(frames),fps,1,interval)]
    previous=None
    for i,frame in enumerate(frames):
        if frame.shape!=(h,w,3): raise ValueError('inconsistent frame shape')
        if mode=='intra' or i%interval==0:
            if mode=='lossless':
                out=io.BytesIO();Image.fromarray(frame).save(out,format='PNG',compress_level=6)
                payload=out.getvalue();kind=2;previous=frame.copy()
            else:
                result=PBC3.compress(Image.fromarray(frame),config=PBC3Config.balanced())
                payload=result.data;kind=0;previous=np.asarray(PBC3.decompress(payload).image).copy()
        else:
            delta=frame.astype(np.int16)-previous.astype(np.int16)
            payload=zlib.compress(delta.astype('<i2').tobytes(),6);kind=1
            previous=(previous.astype(np.int16)+delta).astype(np.uint8)
        parts.extend((RECORD.pack(kind,len(payload),zlib.crc32(payload)),payload))
    data=b''.join(parts)
    if len(data)>MAX_BYTES: raise ValueError('clip exceeds file limit')
    return data

def decode(data):
    if not HEADER.size<=len(data)<=MAX_BYTES: raise ValueError('invalid video size')
    magic,version,w,h,count,num,den,interval=HEADER.unpack_from(data)
    if magic!=b'P26V' or version!=1 or not (w and h and w*h<=2_073_600 and 1<=count<=512 and w*h*count<=32_000_000 and num and den and 1<=interval<=512):
        raise ValueError('invalid video header')
    offset=HEADER.size;frames=[];ledger=[]
    for i in range(count):
        if offset+RECORD.size>len(data): raise ValueError('truncated frame header')
        kind,size,crc=RECORD.unpack_from(data,offset);offset+=RECORD.size
        if size>len(data)-offset: raise ValueError('truncated frame payload')
        payload=data[offset:offset+size];offset+=size
        if zlib.crc32(payload)!=crc: raise ValueError('frame CRC mismatch')
        if kind==0:
            import sys
            from pathlib import Path
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'packaging'))
            from cli import checked_body
            meta=checked_body(payload)
            if (meta['width'],meta['height'],meta['channels'])!=(w,h,3): raise ValueError('keyframe dimensions differ')
            frame=np.asarray(PBC3.decompress(payload).image).copy()
        elif kind==2:
            with Image.open(io.BytesIO(payload)) as image:
                if image.size!=(w,h) or image.mode!='RGB': raise ValueError('invalid PNG keyframe')
                frame=np.asarray(image).copy()
        elif kind==1 and frames:
            delta=np.frombuffer(inflate(payload,w*h*6),dtype='<i2').reshape(h,w,3)
            if np.any(delta<-255) or np.any(delta>255): raise ValueError('signed residual exceeds RGB range')
            summed=frames[-1].astype(np.int32)+delta
            if np.any(summed<0) or np.any(summed>255): raise ValueError('residual reconstructs outside RGB range')
            frame=summed.astype(np.uint8)
        else: raise ValueError('invalid frame type or missing reference')
        frames.append(frame);ledger.append(dict(type=kind,bytes=size))
    if offset!=len(data): raise ValueError('trailing video bytes')
    return frames,dict(width=w,height=h,frames=count,fps=num/den,interval=interval,ledger=ledger)
