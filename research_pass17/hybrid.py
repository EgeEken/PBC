import argparse
import lzma
from pathlib import Path
import struct
import sys

import numpy as np
from scipy.fft import dctn, idctn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PBC3 import PBC3
from pbc3_types import PBC3Config
from analyze_residuals import blocks

MAGIC=b'P17D'
HEADER=struct.Struct('<4sIIIIfBBI')
COLOR=np.array([[1,1,1],[1,0,-1],[1,-2,1]],dtype=np.float32)/np.sqrt(np.array([3,2,6],dtype=np.float32))[:,None]


def coefficients(residual):
    color=residual @ COLOR.T
    return dctn(blocks(color),axes=(-2,-1),type=2,norm='ortho',workers=1)


def correction(quantized,step,h,w):
    spatial=idctn(quantized.astype(np.float32)*step,axes=(-2,-1),type=2,norm='ortho',workers=1)
    by,bx,channels,_,_=spatial.shape
    color=spatial.transpose(0,3,1,4,2).reshape(by*8,bx*8,channels)
    return color[:h,:w] @ COLOR


def pack_coefficients(quantized):
    ordered=quantized.transpose(2,3,4,0,1).astype('<i2',copy=False)
    return lzma.compress(ordered.tobytes(),format=lzma.FORMAT_XZ,preset=lzma.PRESET_EXTREME)


def unpack_coefficients(payload,h,w):
    by,bx=(h+7)//8,(w+7)//8
    raw=lzma.decompress(payload,format=lzma.FORMAT_XZ)
    if len(raw)!=by*bx*3*8*8*2: raise ValueError('coefficient payload size')
    return np.frombuffer(raw,dtype='<i2').reshape(3,8,8,by,bx).transpose(3,4,0,1,2)


def quantize(coeff,step):
    values=np.rint(coeff/step)
    if values.min()<-32768 or values.max()>32767: raise ValueError('coefficient overflow')
    return values.astype(np.int16)


def encode(image,step,patch_count=10):
    config=PBC3Config.quality(patch_count=patch_count,learned_filler_model_path=str(ROOT/'patch_policy.npz'))
    base=PBC3.compress(image,config=config).data
    decoded=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32)
    residual=np.asarray(image,dtype=np.float32)-decoded
    coeff=coefficients(residual)
    payload=pack_coefficients(quantize(coeff,step))
    header=HEADER.pack(MAGIC,image.width,image.height,len(base),len(payload),step,8,0,0)
    return header+base+payload


def encode_tiled(image,step,budget,patch_count=10):
    config=PBC3Config.quality(patch_count=patch_count,learned_filler_model_path=str(ROOT/'patch_policy.npz'))
    base=PBC3.compress(image,config=config).data
    decoded=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32)
    original=np.asarray(image,dtype=np.float32)
    residual=original-decoded
    candidates=[]
    columns=(image.width+31)//32
    for y in range(0,image.height,32):
        for x in range(0,image.width,32):
            tile=residual[y:y+32,x:x+32]
            h,w,_=tile.shape
            quantized=quantize(coefficients(tile),step)
            if not np.any(quantized): continue
            delta=correction(quantized,step,h,w)
            output=np.clip(np.rint(decoded[y:y+h,x:x+w]+delta),0,255)
            gain=float(np.sum(tile**2,dtype=np.float64)-np.sum((original[y:y+h,x:x+w]-output)**2,dtype=np.float64))
            if gain<=0: continue
            packed=pack_coefficients(quantized)
            index=(y//32)*columns+x//32
            cost=8+len(packed)
            candidates.append((gain/cost,index,packed,gain))
    candidates.sort(key=lambda row:(-row[0],row[1]))
    remaining=budget-HEADER.size-len(base)
    if remaining<0: raise ValueError('base exceeds total budget')
    selected=[]
    for ratio,index,packed,gain in candidates:
        cost=8+len(packed)
        if cost<=remaining:
            selected.append((index,packed)); remaining-=cost
    selected.sort()
    payload=b''.join(struct.pack('<II',index,len(packed))+packed for index,packed in selected)
    header=HEADER.pack(MAGIC,image.width,image.height,len(base),len(payload),step,8,1,len(selected))
    return header+base+payload


def decode(data,details=False):
    if len(data)<HEADER.size: raise ValueError('short container')
    magic,w,h,base_size,payload_size,step,block,mode,count=HEADER.unpack_from(data)
    if magic!=MAGIC or block!=8 or mode not in (0,1) or (mode==0 and count!=0) or not np.isfinite(step) or step<=0: raise ValueError('unsupported experimental stream')
    if len(data)!=HEADER.size+base_size+payload_size: raise ValueError('container size')
    base=data[HEADER.size:HEADER.size+base_size]
    pixels=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32)
    if pixels.shape!=(h,w,3): raise ValueError('base dimensions')
    payload=data[HEADER.size+base_size:]
    if mode==0:
        delta=correction(unpack_coefficients(payload,h,w),step,h,w)
    else:
        delta=np.zeros((h,w,3),dtype=np.float32)
        offset=0; seen=set(); columns=(w+31)//32
        for _ in range(count):
            if offset+8>len(payload): raise ValueError('tile header')
            index,size=struct.unpack_from('<II',payload,offset); offset+=8
            if index in seen or index>=((h+31)//32)*columns or offset+size>len(payload): raise ValueError('tile record')
            seen.add(index)
            y,x=divmod(index,columns); y*=32; x*=32
            th,tw=min(32,h-y),min(32,w-x)
            quantized=unpack_coefficients(payload[offset:offset+size],th,tw)
            delta[y:y+th,x:x+tw]=correction(quantized,step,th,tw)
            offset+=size
        if offset!=len(payload): raise ValueError('trailing tile data')
    unbounded=np.rint(pixels+delta)
    output=np.clip(unbounded,0,255).astype(np.uint8)
    if details: return output,{'base_bytes':base_size,'enhancement_bytes':payload_size-8*count,'container_bytes':HEADER.size+8*count,'selected_tiles':count,'mode':mode,'quantization':step,'clipped_pixel_components':int(np.count_nonzero((unbounded<0)|(unbounded>255)))},delta,pixels
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('input',type=Path); parser.add_argument('output',type=Path)
    args=parser.parse_args()
    from PIL import Image
    Image.fromarray(decode(args.input.read_bytes())).save(args.output)
