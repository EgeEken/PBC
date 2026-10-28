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
from research_pass21 import zero_runs

MAGIC=b'P21S'
HEADER=struct.Struct('<4sIIIIfBB')
MAX_PIXELS=16777216
MAX_FILE_BYTES=268435456


def split(data):
    if not HEADER.size<=len(data)<=MAX_FILE_BYTES:raise ValueError('container resource/length')
    magic,w,h,nb,ne,step,block,version=HEADER.unpack_from(data)
    n=((w+7)//8)*((h+7)//8)*3*64
    if magic!=MAGIC or not 0<w<=65535 or not 0<h<=65535 or w*h>MAX_PIXELS or n>zero_runs.MAX_COEFFICIENTS or not np.isfinite(step) or step<=0 or block!=8 or version!=1:
        raise ValueError('unsupported sparse container')
    if nb<6 or len(data)!=HEADER.size+nb+ne:raise ValueError('container framing')
    return (w,h,step,n),data[HEADER.size:HEADER.size+nb],data[HEADER.size+nb:]


def encode(image,step):
    if image.mode!='RGB' or image.width*image.height>MAX_PIXELS:raise ValueError('research RGB dimensions')
    base=PBC3.compress(image,config=PBC3Config.quality(patch_count=10,learned_filler_model_path=str(ROOT/'patch_policy.npz'))).data
    pixels=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32)
    q=h17.quantize(h17.coefficients(np.asarray(image,dtype=np.float32)-pixels),step)
    payload=lzma.compress(zero_runs.pack(q),format=lzma.FORMAT_XZ,preset=0)
    return HEADER.pack(MAGIC,image.width,image.height,len(base),len(payload),step,8,1)+base+payload


def read_coefficients(payload,w,h,n):
    raw=zero_runs.decompress(payload,n)
    values=zero_runs.unpack(raw,n)
    return values.reshape(3,8,8,(h+7)//8,(w+7)//8).transpose(3,4,0,1,2)


def decode(data):
    (w,h,step,n),base,payload=split(data)
    pixels=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.float32)
    if pixels.shape!=(h,w,3):raise ValueError('base dimensions')
    q=read_coefficients(payload,w,h,n)
    return np.clip(np.rint(pixels+h17.correction(q,step,h,w)),0,255).astype(np.uint8)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    Image.fromarray(decode(args.input.read_bytes())).save(args.output)
