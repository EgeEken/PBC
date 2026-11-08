import lzma
from pathlib import Path
import numpy as np
from PIL import Image
import pbc3_stream as stream
from pbc3_types import BitReader
from pbc3_decode_ops import signed_resample
import pbc3_decode_ops as ops

MAX_FILE=64*1024*1024
MAX_BODY=128*1024*1024
MAX_PIXELS=32_000_000
FILTERS=[{"id":lzma.FILTER_LZMA2,"preset":lzma.PRESET_EXTREME}]

def _open(data):
    import lzma
    from pbc3_types import BitReader
    if len(data)<6 or len(data)>MAX_FILE or data[:5]!=b'PBC3\0':
        raise ValueError('invalid, unsupported or oversized PBC3 v0 file')
    if data[5]==0:
        body=data[6:]
    elif data[5]==2:
        d=lzma.LZMADecompressor(format=lzma.FORMAT_RAW,filters=FILTERS)
        try:
            body=d.decompress(data[6:],max_length=MAX_BODY+1)
        except lzma.LZMAError as e:
            raise ValueError('invalid LZMA payload') from e
        if len(body)>MAX_BODY or not d.eof or d.unused_data:
            raise ValueError('truncated, trailing or oversized LZMA payload')
    else:
        raise ValueError('unsupported entropy method')
    br=BitReader(body)
    header=stream.read_header(br,{0:'RGB',1:'YCbCr'})
    down,ow,oh,w,h,color,channels,cb,bias,alpha,count,bases,warm,ww,wh,split=header
    if not (w and h and w*h<=MAX_PIXELS and channels==(4 if alpha else 3) and cb==2 and count<=10000):
        raise ValueError('unsupported dimensions, channels or patch count')
    if down and not (ow and oh and ow*oh<=MAX_PIXELS):
        raise ValueError('unsupported output dimensions')
    if warm and not (ww and wh and ww*wh<=MAX_PIXELS and split<=count):
        raise ValueError('invalid warmup dimensions or patch split')
    return br,header,dict(version=0,working_width=w,working_height=h,width=ow if down else w,height=oh if down else h,
                color_space=color,channels=channels,alpha=alpha,patches=count,warmup=warm,warmup_width=ww,warmup_height=wh,warmup_split=split,entropy=data[5],bytes=len(data))

def read_file(path):
    path=Path(path)
    if path.stat().st_size>MAX_FILE: raise ValueError('file exceeds 64 MiB limit')
    return path.read_bytes()

def _patch_header(br,header,i):
    cb,bias=header[7],header[8]
    channel=br.read(cb)
    x,y,w,h=(br.read(16) for _ in range(4))
    if br.read(1): raise ValueError('unsupported explicit palette')
    size=br.read(10)
    if not size: raise ValueError('empty palette mask')
    mask=[br.read(1) for _ in range(size)]
    negative,positive=br.read(8),br.read(8)
    maximum=br.read(4)
    if not 1<=maximum<=9: raise ValueError('invalid palette bit depth')
    bits=ops.resolve_palette_bitcount(mask,maximum,negative,positive,bias)
    cell=br.read(16)
    cw,ch=(header[13],header[14]) if header[12] and i>=header[15] else (header[3],header[4])
    if not (channel<header[6] and w and h and cell and x+w<=cw and y+h<=ch):
        raise ValueError('invalid patch geometry')
    gw,gh=(w+cell-1)//cell,(h+cell-1)//cell
    if gw*gh*bits>br.nbits+8*(len(br.data)-br.i): raise EOFError('truncated patch indices')
    return channel,x,y,w,h,cell,mask,negative,positive,maximum,bits,gw,gh

def _skip(br,bits):
    buffered=min(bits,br.nbits);br.read(buffered)
    remaining=bits-buffered;br.i+=remaining//8;br.read(remaining%8)

def _finish(br):
    if br.i!=len(br.data) or br.acc: raise ValueError('trailing stream data or nonzero padding')

def inspect(data):
    br,header,metadata=_open(data)
    for i in range(header[10]):
        patch=_patch_header(br,header,i)
        _skip(br,patch[10]*patch[11]*patch[12])
    _finish(br)
    return metadata

def _resize(canvas,w,h):
    out=np.empty((h,w,canvas.shape[2]),dtype=np.int16)
    for c in range(canvas.shape[2]):
        layer=np.ascontiguousarray(canvas[:,:,c],dtype=np.float32)
        image=Image.frombuffer('F',(canvas.shape[1],canvas.shape[0]),layer,'raw','F',0,1)
        out[:,:,c]=np.rint(np.asarray(image.resize((w,h),Image.Resampling.BICUBIC),dtype=np.float32)).astype(np.int16)
    return out

def decode(data,max_patches=None):
    br,header,metadata=_open(data)
    down,ow,oh,w,h,color,channels,cb,bias,alpha,count,bases,warm,ww,wh,split=header
    if max_patches is not None and max_patches<0: raise ValueError('negative patch limit')
    limit=count if max_patches is None else min(count,max_patches)
    canvas=np.zeros((h,w,channels),dtype=np.int16)
    for c,base in enumerate(bases): canvas[:,:,c]=base
    for i in range(count):
        c,x,y,pw,ph,cell,mask,neg,pos,maximum,bits,gw,gh=_patch_header(br,header,i)
        if i>=limit:
            _skip(br,bits*gw*gh)
            continue
        if warm and i==split: canvas=_resize(canvas,ww,wh)
        palette=ops.palette_generator(mask,maximum,neg,pos,bias)
        indices=stream.read_grid(br,gh*gw,bits).reshape(gh,gw)
        canvas[y:y+ph,x:x+pw,c]+=signed_resample(palette[indices],ph,pw)
    _finish(br)
    arr=np.clip(canvas,0,255).astype(np.uint8)
    if alpha:
        image=Image.fromarray(arr[:,:,:3],color).convert('RGB').convert('RGBA')
        image.putalpha(Image.fromarray(arr[:,:,3],'L'))
    else: image=Image.fromarray(arr,color).convert('RGB')
    if down and image.size!=(ow,oh): image=image.resize((ow,oh),Image.Resampling.BICUBIC,reducing_gap=None)
    return image
