import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

if not getattr(sys,'frozen',False):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'artifacts/published'))
else:
    cache=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'PBC32Research/numba'
    os.environ.setdefault('NUMBA_CACHE_DIR',str(cache))

PRESETS=('compression','balanced','quality','high_quality')
MAX_FILE=64*1024*1024
MAX_BODY=128*1024*1024
MAX_PIXELS=32_000_000

def checked_body(data):
    import lzma
    from PBC3 import PBC3
    from pbc3_types import BitReader
    if len(data)<6 or len(data)>MAX_FILE or data[:5]!=b'PBC3\0':
        raise ValueError('invalid, unsupported or oversized PBC3 v0 file')
    if data[5]==0:
        body=data[6:]
    elif data[5]==2:
        d=lzma.LZMADecompressor(format=lzma.FORMAT_RAW,filters=PBC3._LZMA_FILTERS)
        try:
            body=d.decompress(data[6:],max_length=MAX_BODY+1)
        except lzma.LZMAError as e:
            raise ValueError('invalid LZMA payload') from e
        if len(body)>MAX_BODY or not d.eof or d.unused_data:
            raise ValueError('truncated, trailing or oversized LZMA payload')
    else:
        raise ValueError('unsupported entropy method')
    br=BitReader(body)
    header=PBC3._read_header(br)
    down,ow,oh,w,h,color,channels,cb,bias,alpha,count,bases,warm,ww,wh,split=header
    if not (w and h and w*h<=MAX_PIXELS and channels==(4 if alpha else 3) and cb==2 and count<=10000):
        raise ValueError('unsupported dimensions, channels or patch count')
    if down and not (ow and oh and ow*oh<=MAX_PIXELS):
        raise ValueError('unsupported output dimensions')
    if warm and not (ww and wh and ww*wh<=MAX_PIXELS and split<=count):
        raise ValueError('invalid warmup dimensions or patch split')
    for i in range(count):
        import pbc3_ops as ops
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
    return dict(version=0,working_width=w,working_height=h,width=ow if down else w,height=oh if down else h,
                color_space=color,channels=channels,alpha=alpha,patches=count,warmup=warm,warmup_width=ww,warmup_height=wh,warmup_split=split,entropy=data[5],bytes=len(data))

def main():
    parser=argparse.ArgumentParser(prog='pbc',description='Standalone research CLI for the published PBC 3.1 codec')
    parser.add_argument('--version',action='version',version='pbc 3.2 packaging prototype; codec 3.1.0, stream v0')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('encode','decode'):
        p=sub.add_parser(command)
        p.add_argument('input',type=Path); p.add_argument('output',type=Path)
        p.add_argument('--force',action='store_true')
        if command=='encode': p.add_argument('--preset',choices=PRESETS,default='balanced')
    p=sub.add_parser('inspect'); p.add_argument('input',type=Path)
    p=sub.add_parser('benchmark',help='research: warmed in-process codec timing')
    p.add_argument('input',type=Path); p.add_argument('--preset',choices=PRESETS,default='quality')
    p.add_argument('--repeats',type=int,choices=range(1,11),default=3)
    args=parser.parse_args()
    try:
        if hasattr(args,'output') and args.output.exists() and not args.force:
            raise FileExistsError('output exists; use --force to replace it')
        if hasattr(args,'output') and args.input.resolve()==args.output.resolve():
            raise ValueError('input and output paths must differ')
        t=time.perf_counter()
        from PBC3 import PBC3
        from pbc3_types import PBC3Config
        from PIL import Image, ImageOps
        imports=time.perf_counter()-t
        if args.command=='inspect':
            if args.input.stat().st_size>MAX_FILE: raise ValueError('file exceeds 64 MiB research limit')
            result=checked_body(args.input.read_bytes())
        elif args.command in ('encode','benchmark'):
            with Image.open(args.input) as opened:
                image=ImageOps.exif_transpose(opened).copy()
            if image.width*image.height>MAX_PIXELS: raise ValueError('image exceeds 32 megapixel research limit')
            config=getattr(PBC3Config,args.preset)()
            t=time.perf_counter(); encoded=PBC3.compress(image,config=config); first=time.perf_counter()-t
            if args.command=='benchmark':
                import hashlib
                timings=[]
                for _ in range(args.repeats):
                    t=time.perf_counter(); r=PBC3.compress(image,config=config); enc=time.perf_counter()-t
                    assert r.data==encoded.data
                    t=time.perf_counter(); decoded=PBC3.decompress(r.data); dec=time.perf_counter()-t
                    timings.append(dict(encode_seconds=enc,decode_seconds=dec))
                result=dict(first_encode_seconds=first,rows=timings,bytes=len(encoded.data),stream_sha256=hashlib.sha256(encoded.data).hexdigest(),pixel_sha256=hashlib.sha256(decoded.image.tobytes()).hexdigest())
            else:
                checked_body(encoded.data)
                with args.output.open('wb' if args.force else 'xb') as f: f.write(encoded.data)
                result=dict(bytes=len(encoded.data),codec_seconds=first,mse=encoded.mse,preset=args.preset)
        else:
            if args.input.stat().st_size>MAX_FILE: raise ValueError('file exceeds 64 MiB research limit')
            data=args.input.read_bytes(); metadata=checked_body(data)
            t=time.perf_counter(); decoded=PBC3.decompress(data); elapsed=time.perf_counter()-t
            output=io.BytesIO(); decoded.image.save(output,format=Image.registered_extensions().get(args.output.suffix.lower(),'PNG'))
            with args.output.open('wb' if args.force else 'xb') as f: f.write(output.getvalue())
            result=dict(codec_seconds=elapsed,**metadata)
        print(json.dumps(dict(import_seconds=imports,**result)))
        return 0
    except (OSError,ValueError,EOFError,IndexError,KeyError,OverflowError,RuntimeError) as e:
        print('pbc: '+str(e),file=sys.stderr)
        return 1

if __name__=='__main__':
    sys.exit(main())
