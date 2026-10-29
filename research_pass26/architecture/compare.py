import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import io
import math
import subprocess
import time
from PIL import features

out=PASS/'artifacts/architecture';out.mkdir(exist_ok=True)
rows=[]
dimensions={}
for name in ('googlewiki.png','mario64.png','monet.jpg','cat.jpg','flower.jpg'):
    image=load(name)
    dimensions[name]=image.size
    targets=[]
    PBC3.compress(image,config=PBC3Config.quality())
    for preset in ('quality','high_quality'):
        t=time.perf_counter();r=PBC3.compress(image,config=getattr(PBC3Config,preset)());enc=time.perf_counter()-t
        t=time.perf_counter();decoded=PBC3.decompress(r.data).image;dec=time.perf_counter()-t
        (out/(name+'.'+preset+'.pbc')).write_bytes(r.data)
        rows.append(dict(image=name,codec='PBC31',setting=preset,bytes=len(r.data),mse=mse(image,decoded),encode_seconds=enc,decode_seconds=dec,sha256=sha(r.data),pixel_sha256=sha(decoded.tobytes())))
        targets.append(len(r.data))
    prep=PBC3.prepare(image,config=PBC3Config.quality())
    working=Image.fromarray(prep['arr'],'YCbCr')
    for codec in ('JPEG','WEBP','AVIF','working_JPEG','working_WEBP'):
        if codec=='AVIF' and not features.check('avif'): continue
        is_working=codec.startswith('working_');fmt=codec.removeprefix('working_')
        source=working if is_working and fmt=='JPEG' else working.convert('RGB') if is_working else image
        measured={}
        def point(q):
            if q in measured: return measured[q]
            opts=dict(quality=q)
            if fmt=='JPEG': opts.update(subsampling=2,optimize=True,progressive=False)
            elif fmt=='WEBP': opts.update(method=4)
            else: opts.update(speed=8,max_threads=1,subsampling='4:2:0')
            if is_working:
                exif=Image.Exif();exif[270]=f'P26_ORIGINAL={image.width}x{image.height}'
                opts['exif']=exif
            t=time.perf_counter();buf=io.BytesIO();source.save(buf,format=fmt,**opts);data=buf.getvalue();enc=time.perf_counter()-t
            t=time.perf_counter();decoded=Image.open(io.BytesIO(data));decoded.load()
            if is_working:
                marker=decoded.getexif()[270];assert marker.startswith('P26_ORIGINAL=')
                size=tuple(map(int,marker.split('=')[1].split('x')))
                decoded=decoded.convert('RGB').resize(size,Image.Resampling.BICUBIC,reducing_gap=None)
            else: decoded=decoded.convert('RGB')
            dec=time.perf_counter()-t
            file=out/(name+'.'+codec+'.'+str(q)+'.'+fmt.lower());file.write_bytes(data)
            row=dict(image=name,codec=codec,setting=q,bytes=len(data),mse=mse(image,decoded),encode_seconds=enc,decode_seconds=dec,sha256=sha(data),pixel_sha256=sha(decoded.tobytes()),working_size=source.size,metadata_original_size=is_working)
            rows.append(row);measured[q]=row;return row
        low=1 if fmt=='JPEG' else 0
        point(low);point(100)
        for target in targets:
            for _ in range(4):
                below=[q for q,r in measured.items() if r['bytes']<=target]
                above=[q for q,r in measured.items() if r['bytes']>=target]
                if not below or not above: break
                lo=max(below);hi=min(above)
                if hi-lo<=1: break
                point((lo+hi)//2)
        print(name,codec,len(measured),'points',flush=True)
    raw=out/(name+'.rgb');raw.write_bytes(np.asarray(image).tobytes())
    for distance in (8,15):
        file=out/(name+f'.d{distance}.jxl')
        t=time.perf_counter()
        subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{image.width}x{image.height}','-i',str(raw),'-frames:v','1','-c:v','libjxl','-effort','3','-distance',str(distance),'-threads','1',str(file)],check=True,capture_output=True)
        enc=time.perf_counter()-t;t=time.perf_counter()
        decoded=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(file),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'],capture_output=True,check=True).stdout
        dec=time.perf_counter()-t;array=np.frombuffer(decoded,np.uint8).reshape(image.height,image.width,3)
        data=file.read_bytes();rows.append(dict(image=name,codec='JXL',setting=distance,bytes=len(data),mse=mse(image,array),encode_seconds=enc,decode_seconds=dec,sha256=sha(data),pixel_sha256=sha(decoded)))
    for row in rows:
        width,height=dimensions[row['image']]
        row.update(width=width,height=height,bpp=row['bytes']*8/(width*height))
    save('architecture/results.json',rows)
print('architecture comparison complete',len(rows),flush=True)
