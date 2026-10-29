import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
import time
from container import encode,decode

out=PASS/'artifacts/video';out.mkdir(exist_ok=True)
w,h,n,fps=256,144,16,12
photo=load('cat.jpg').resize((w+32,h+16),Image.Resampling.LANCZOS)
flower=np.asarray(load('flower.jpg').resize((w,h),Image.Resampling.LANCZOS))
static=[flower.copy() for _ in range(n)]
y,x=np.mgrid[:h,:w]
background=np.stack((40+x//2+0*y,50+y+0*x,80+x//4+y//4),axis=-1).astype(np.uint8)
motion=[]
for i in range(n):
    f=background.copy();xx=8+9*i;f[48:88,xx:xx+32]=(235,55,45);motion.append(f)
pan=[np.asarray(photo.crop((2*i,i,w+2*i,h+i))) for i in range(n)]
clips=dict(static_flower=static,moving_shape=motion,photo_pan=pan)
PBC3.compress(Image.fromarray(background),config=PBC3Config.balanced())
rows=[]
for name,frames in clips.items():
    source=np.stack(frames);(out/(name+'.rgb')).write_bytes(source.tobytes())
    for mode in ('intra','temporal','lossless','h264'):
        t=time.perf_counter()
        if mode=='h264':
            file=out/(name+'.mp4')
            subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','28','-threads','1','-pix_fmt','yuv420p',str(file)],input=source.tobytes(),check=True)
            data=file.read_bytes()
        else:
            data=encode(frames,mode=mode,interval=8,fps=fps);file=out/(name+'.'+mode+'.p26v');file.write_bytes(data)
        enc=time.perf_counter()-t;t=time.perf_counter()
        if mode=='h264':
            raw=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(file),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout
            decoded=np.frombuffer(raw,np.uint8).reshape(n,h,w,3);meta=None
        else: decoded,meta=decode(data)
        dec=time.perf_counter()-t
        errors=[mse(a,b) for a,b in zip(frames,decoded)]
        row=dict(clip=name,mode=mode,bytes=len(data),bpp_per_frame=len(data)*8/(w*h*n),mean_mse=float(np.mean(errors)),worst_mse=max(errors),final_mse=errors[-1],frame_mse=errors,encode_seconds=enc,decode_seconds=dec,encode_fps=n/enc,decode_fps=n/dec,sha256=sha(data),frame_hashes=[sha(np.asarray(f).tobytes()) for f in decoded],source_sha256=sha(source.tobytes()),metadata=meta)
        if meta:
            row['residual_fraction']=sum(r['bytes'] for r in meta['ledger'] if r['type']==1)/len(data)
            if mode=='temporal': assert all(errors[i]==0 for i in range(n) if i%8)
            if mode=='lossless': assert max(errors)==0
        rows.append(row);print(name,mode,len(data),row['mean_mse'],enc,dec,flush=True)
save('video/results.json',rows)
