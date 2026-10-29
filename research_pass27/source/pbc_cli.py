import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

if getattr(sys,'frozen',False):
    cache=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'PBC/3.2rc1/numba'
    os.environ.setdefault('NUMBA_CACHE_DIR',str(cache))
    os.environ.setdefault('NUMBA_CACHE_LOCATOR_CLASSES','pbc3_cache.FrozenCacheLocator,UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator')

PRESETS=('compression','balanced','quality','high_quality')
MAX_FILE=64*1024*1024
MAX_BODY=128*1024*1024
MAX_PIXELS=32_000_000

def main(decoder_only=False):
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(prog='pbc',description='PBC 3.2 release candidate (v0 compression preserved)')
    parser.add_argument('--version',action='version',version='pbc 3.2.0rc1; stream v0')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in (('decode',) if decoder_only else ('encode','decode')):
        p=sub.add_parser(command)
        p.add_argument('input',type=Path); p.add_argument('output',type=Path)
        p.add_argument('--force',action='store_true')
        if command=='encode':
            p.add_argument('--preset',choices=PRESETS,default='balanced')
            p.add_argument('--mse-backend',choices=('reference','native','auto'),default='auto')
    p=sub.add_parser('inspect'); p.add_argument('input',type=Path)
    p=sub.add_parser('benchmark',help='research: warmed in-process codec timing')
    p.add_argument('input',type=Path); p.add_argument('--preset',choices=PRESETS,default='quality')
    p.add_argument('--repeats',type=int,choices=range(1,11),default=3)
    p.add_argument('--mse-backend',choices=('reference','native','auto'),default='auto')
    p=sub.add_parser('benchdecode',help='research: warmed decode timing');p.add_argument('input',type=Path);p.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if decoder_only and args.command in ('encode','benchmark'): parser.error('encoding is unavailable')
    try:
        if hasattr(args,'output') and args.output.exists() and not args.force:
            raise FileExistsError('output exists; use --force to replace it')
        if hasattr(args,'output') and args.input.resolve()==args.output.resolve():
            raise ValueError('input and output paths must differ')
        t=time.perf_counter()
        from pbc3_decoder import decode,inspect,read_file
        from PIL import Image, ImageOps
        imports=time.perf_counter()-t
        if args.command=='inspect':
            if args.input.stat().st_size>MAX_FILE: raise ValueError('file exceeds 64 MiB research limit')
            result=inspect(read_file(args.input))
        elif args.command=='benchdecode':
            import hashlib
            data=read_file(args.input);decode(data);times=[]
            for _ in range(args.repeats):
                t=time.perf_counter();image=decode(data);times.append(time.perf_counter()-t)
            result=dict(seconds=times,pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
        elif args.command in ('encode','benchmark'):
            encoder_import_started=time.perf_counter()
            from PBC3 import PBC3
            from pbc3_types import PBC3Config
            imports+=time.perf_counter()-encoder_import_started
            with Image.open(args.input) as opened:
                image=ImageOps.exif_transpose(opened).copy()
            if image.width*image.height>MAX_PIXELS: raise ValueError('image exceeds 32 megapixel research limit')
            config=getattr(PBC3Config,args.preset)()
            t=time.perf_counter(); encoded=PBC3.compress(image,config=config,mse_backend=args.mse_backend); first=time.perf_counter()-t
            if args.command=='benchmark':
                import hashlib
                timings=[]
                for _ in range(args.repeats):
                    t=time.perf_counter(); r=PBC3.compress(image,config=config,mse_backend=args.mse_backend); enc=time.perf_counter()-t
                    assert r.data==encoded.data
                    t=time.perf_counter(); decoded=PBC3.decompress(r.data); dec=time.perf_counter()-t
                    timings.append(dict(encode_seconds=enc,decode_seconds=dec))
                result=dict(first_encode_seconds=first,rows=timings,bytes=len(encoded.data),stream_sha256=hashlib.sha256(encoded.data).hexdigest(),pixel_sha256=hashlib.sha256(decoded.image.tobytes()).hexdigest())
            else:
                inspect(encoded.data)
                with args.output.open('wb' if args.force else 'xb') as f: f.write(encoded.data)
                result=dict(bytes=len(encoded.data),codec_seconds=first,mse=encoded.mse,preset=args.preset)
        else:
            if args.input.stat().st_size>MAX_FILE: raise ValueError('file exceeds 64 MiB research limit')
            data=read_file(args.input)
            t=time.perf_counter(); image=decode(data); elapsed=time.perf_counter()-t
            metadata=dict(width=image.width,height=image.height,mode=image.mode,bytes=len(data))
            fmt=Image.registered_extensions().get(args.output.suffix.lower())
            if fmt is None: raise ValueError('unknown output image extension')
            output=io.BytesIO(); image.save(output,format=fmt)
            with args.output.open('wb' if args.force else 'xb') as f: f.write(output.getvalue())
            result=dict(codec_seconds=elapsed,**metadata)
        print(json.dumps(dict(import_seconds=imports,**result)))
        return 0
    except (OSError,ValueError,EOFError,IndexError,KeyError,OverflowError,RuntimeError) as e:
        print('pbc: '+str(e),file=sys.stderr)
        return 1

if __name__=='__main__':
    sys.exit(main())
