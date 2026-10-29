import lzma
import struct
import numpy as np
from scipy.fft import dctn,idctn
from research_pass21 import zero_runs
from research_pass17.analyze_residuals import blocks
from PBC3 import PBC3

HEADER=struct.Struct('<4sHHHHB3BfI')
MAX_WORKING_PIXELS=1048576
MAX_ORIGINAL_PIXELS=16777216
MAX_FILE_BYTES=67108864


def encode(prep,step):
    if prep['has_alpha'] or prep['warm_target'] is not None or prep['channels']!=3:raise ValueError('research RGB without alpha/warmup required')
    means=[int(round(float(np.mean(prep['arr'][:,:,c])))) for c in range(3)]
    residual=prep['target'].astype(np.float32)-np.array(means,dtype=np.float32)
    coefficients=dctn(blocks(residual),axes=(-2,-1),type=2,norm='ortho',workers=1)
    quantized=np.rint(coefficients/step)
    if not np.isfinite(step) or step<=0 or quantized.min()<-32768 or quantized.max()>32767:raise ValueError('quantization range')
    payload=lzma.compress(zero_runs.pack(quantized.astype(np.int16)),format=lzma.FORMAT_XZ,preset=0)
    return HEADER.pack(b'P25I',prep['w'],prep['h'],prep['original_w'],prep['original_h'],prep['color_id'],*means,step,len(payload))+payload


def split(data):
    if not HEADER.size<=len(data)<=MAX_FILE_BYTES:raise ValueError('initializer file length')
    magic,w,h,ow,oh,color,*tail=HEADER.unpack_from(data)
    means=tail[:3];step,length=tail[3:]
    if magic!=b'P25I' or min(w,h,ow,oh)<1 or w*h>MAX_WORKING_PIXELS or ow*oh>MAX_ORIGINAL_PIXELS or w>ow or h>oh or color not in (0,1) or not np.isfinite(step) or step<=0 or len(data)!=HEADER.size+length:raise ValueError('initializer header')
    return dict(w=w,h=h,original_w=ow,original_h=oh,color_space=PBC3.COLOR_SPACE_NAMES[color],means=means,step=step),data[HEADER.size:]


def decode_canvas(data):
    info,payload=split(data);w,h=info['w'],info['h'];by,bx=(h+7)//8,(w+7)//8;n=by*bx*3*64
    raw=zero_runs.decompress(payload,n)
    flat=zero_runs.unpack(raw,n)
    q=flat.reshape(3,8,8,by,bx).transpose(3,4,0,1,2)
    spatial=idctn(q.astype(np.float32)*np.float32(info['step']),axes=(-2,-1),type=2,norm='ortho',workers=1)
    working=spatial.transpose(0,3,1,4,2).reshape(by*8,bx*8,3)[:h,:w]
    canvas=np.rint(working+np.array(info['means'],dtype=np.float32))
    if not np.isfinite(canvas).all() or canvas.min()<-32768 or canvas.max()>32767:raise ValueError('initializer canvas range')
    return canvas.astype(np.int16),info


def render(canvas,info):
    image=PBC3._canvas_to_image(canvas,info['color_space'],False)
    size=(info['original_w'],info['original_h'])
    if image.size!=size:image=image.resize(size,PBC3.RESAMPLE_FILTER,reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
    return image


def decode(data):
    canvas,info=decode_canvas(data)
    return render(canvas,info)
