import argparse
import lzma
import struct
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_pass17'))
sys.path.insert(0,str(ROOT))
import hybrid as h17
import numpy as np
from PIL import Image
from PBC3 import PBC3
from pbc3_types import PBC3Config

MAGIC=b'P20M'
HEADER=struct.Struct('<4sIIIIIIffBBBB')
BAND=np.add.outer(np.arange(8),np.arange(8))>=4
SETTINGS={'lean':(32,128),'rich':(16,64)}


def resize_float(array,width,height):
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(array[:,:,c],dtype=np.float32)).resize((width,height),Image.Resampling.BICUBIC,reducing_gap=None),dtype=np.float32) for c in range(3)],axis=2)


def layer_bytes(coarse,fine):
    first=coarse.transpose(2,3,4,0,1).astype('<i2',copy=False).tobytes()
    second=fine.transpose(2,3,4,0,1)[:,BAND].astype('<i2',copy=False).tobytes()
    return first,second


def read_layers(payload,width,height,cw,ch):
    by,bx=(height+7)//8,(width+7)//8
    cy,cx=(ch+7)//8,(cw+7)//8
    nc=cy*cx*3*64
    nf=by*bx*3*int(BAND.sum())
    reader=lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    raw=reader.decompress(payload,max_length=(nc+nf)*2+1)
    if not reader.eof or reader.unused_data or len(raw)!=(nc+nf)*2: raise ValueError('coefficient framing')
    values=np.frombuffer(raw,dtype='<i2')
    coarse=values[:nc].reshape(3,8,8,cy,cx).transpose(3,4,0,1,2)
    ordered=np.zeros((3,8,8,by,bx),dtype=np.int16)
    ordered[:,BAND]=values[nc:].reshape(3,int(BAND.sum()),by,bx)
    return coarse,ordered.transpose(3,4,0,1,2)


def split(data):
    if len(data)<HEADER.size: raise ValueError('short header')
    magic,w,h,nb,npayload,cw,ch,qc,qf,block,scale,band,filter_id=HEADER.unpack_from(data)
    if magic!=MAGIC or not 0<w<=65535 or not 0<h<=65535 or w*h>64000000 or block!=8 or scale!=4 or band!=4 or filter_id!=1 or (cw,ch)!=((w+3)//4,(h+3)//4) or not np.isfinite(qc) or not np.isfinite(qf) or qc<=0 or qf<=0:
        raise ValueError('unsupported multiscale header')
    if nb<6 or len(data)!=HEADER.size+nb+npayload: raise ValueError('container length')
    base=data[HEADER.size:HEADER.size+nb]
    coarse,fine=read_layers(data[HEADER.size+nb:],w,h,cw,ch)
    return (w,h,cw,ch,qc,qf),base,coarse,fine


def enhance(image,base,setting):
    qc,qf=SETTINGS[setting]
    pixels=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32)
    original=np.asarray(image,dtype=np.float32)
    if pixels.shape!=original.shape: raise ValueError('base dimensions')
    residual=original-pixels
    w,h=image.size;cw,ch=(w+3)//4,(h+3)//4
    coarse=h17.quantize(h17.coefficients(resize_float(residual,cw,ch)),qc)
    coarse_delta=resize_float(h17.correction(coarse,qc,ch,cw),w,h)
    fine=h17.quantize(h17.coefficients(residual-coarse_delta),qf)
    fine[:,:,:,~BAND]=0
    first,second=layer_bytes(coarse,fine)
    payload=lzma.compress(first+second,format=lzma.FORMAT_XZ,preset=0)
    return HEADER.pack(MAGIC,w,h,len(base),len(payload),cw,ch,qc,qf,8,4,4,1)+base+payload


def encode(image,setting='lean',patch_count=10):
    base=PBC3.compress(image,config=PBC3Config.quality(patch_count=patch_count,learned_filler_model_path=str(ROOT/'patch_policy.npz'))).data
    return enhance(image,base,setting)


def decode(data,details=False):
    (w,h,cw,ch,qc,qf),base,coarse,fine=split(data)
    pixels=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32)
    if pixels.shape!=(h,w,3): raise ValueError('decoded base dimensions')
    coarse_delta=resize_float(h17.correction(coarse,qc,ch,cw),w,h)
    fine_delta=h17.correction(fine,qf,h,w)
    output=np.clip(np.rint(pixels+coarse_delta+fine_delta),0,255).astype(np.uint8)
    if details:return output,dict(base_bytes=len(base),shared_enhancement_bytes=len(data)-HEADER.size-len(base),container_bytes=HEADER.size,coarse_size=[cw,ch],coarse_step=qc,fine_step=qf,coarse_raw_bytes=coarse.size*2,fine_raw_bytes=fine.shape[0]*fine.shape[1]*3*int(BAND.sum())*2),coarse_delta,fine_delta,pixels
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    Image.fromarray(decode(args.input.read_bytes())).save(args.output)
