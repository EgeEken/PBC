import lzma
from pathlib import Path
import numpy as np
from PIL import Image
import pbc3_stream as stream
from pbc3_types import BitReader
from pbc3_decode_ops import signed_resample

MAX_FILE=64*1024*1024
MAX_BODY=128*1024*1024
MAX_PIXELS=32_000_000
FILTERS=[{"id":lzma.FILTER_LZMA2,"preset":lzma.PRESET_EXTREME}]

def _scan(data):
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
    for i in range(count):
        import pbc3_decode_ops as ops
        channel=br.read(cb)
        x,y,pw,ph=(br.read(16) for _ in range(4))
        if br.read(1): raise ValueError('unsupported explicit palette')
        mask_size=br.read(10)
        if not mask_size: raise ValueError('empty palette mask')
        mask=[br.read(1) for _ in range(mask_size)]
        negative,positive=br.read(8),br.read(8)
        maximum=br.read(4)
        if not 1<=maximum<=9: raise ValueError('invalid palette bit depth')
        bits=ops.resolve_palette_bitcount(mask,maximum,negative,positive,bias)
        cell=br.read(16)
        cw,ch=(ww,wh) if warm and i>=split else (w,h)
        if not (channel<channels and pw and ph and cell and x+pw<=cw and y+ph<=ch):
            raise ValueError('invalid patch geometry')
        grid_bits=((pw+cell-1)//cell)*((ph+cell-1)//cell)*bits
        if grid_bits>br.nbits+8*(len(body)-br.i): raise EOFError('truncated patch indices')
        buffered=min(grid_bits,br.nbits)
        br.read(buffered)
        remaining=grid_bits-buffered
        br.i+=remaining//8
        br.read(remaining%8)
    if br.i!=len(body) or br.acc:
        raise ValueError('trailing stream data or nonzero padding')
    return body,header,dict(version=0,working_width=w,working_height=h,width=ow if down else w,height=oh if down else h,
                color_space=color,channels=channels,alpha=alpha,patches=count,warmup=warm,warmup_width=ww,warmup_height=wh,warmup_split=split,entropy=data[5],bytes=len(data))

def read_file(path):
    path=Path(path)
    if path.stat().st_size>MAX_FILE: raise ValueError('file exceeds 64 MiB limit')
    return path.read_bytes()

def inspect(data):
    return _scan(data)[2]

def _resize(canvas,w,h):
    out=np.empty((h,w,canvas.shape[2]),dtype=np.int16)
    for c in range(canvas.shape[2]):
        layer=np.ascontiguousarray(canvas[:,:,c],dtype=np.float32)
        image=Image.frombuffer('F',(canvas.shape[1],canvas.shape[0]),layer,'raw','F',0,1)
        out[:,:,c]=np.rint(np.asarray(image.resize((w,h),Image.Resampling.BICUBIC),dtype=np.float32)).astype(np.int16)
    return out

def decode(data,max_patches=None):
    body,header,metadata=_scan(data)
    down,ow,oh,w,h,color,channels,cb,bias,alpha,count,bases,warm,ww,wh,split=header
    if max_patches is not None and max_patches<0: raise ValueError('negative patch limit')
    br=BitReader(body);stream.read_header(br,{0:'RGB',1:'YCbCr'})
    canvas=np.zeros((h,w,channels),dtype=np.int16)
    for c,base in enumerate(bases): canvas[:,:,c]=base
    for i in range(count if max_patches is None else min(count,max_patches)):
        if warm and i==split: canvas=_resize(canvas,ww,wh)
        c,x,y,pw,ph,cell,values,bits=stream.read_patch(br,cb,bias)
        canvas[y:y+ph,x:x+pw,c]+=signed_resample(values,ph,pw)
    arr=np.clip(canvas,0,255).astype(np.uint8)
    if alpha:
        image=Image.fromarray(arr[:,:,:3],color).convert('RGB').convert('RGBA')
        image.putalpha(Image.fromarray(arr[:,:,3],'L'))
    else: image=Image.fromarray(arr,color).convert('RGB')
    if down and image.size!=(ow,oh): image=image.resize((ow,oh),Image.Resampling.BICUBIC,reducing_gap=None)
    return image
