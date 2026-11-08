import hashlib, json, os, pathlib, subprocess, time
import numpy as np
from PIL import Image, ImageOps

ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=pathlib.Path(__file__).resolve().parents[1]/"outputs"
BIN=ROOT/"research_pass35"/"outputs"/"binaries"/"reference"
ENV=dict(os.environ,PATH=str(BIN)+";C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
QS=[0,5,20,35,50,65,80,90,95]
IMAGES=["googlewiki.png","cat.jpg","coast.jpg"]
def run(cmd):
 t=time.perf_counter(); p=subprocess.run(cmd,capture_output=True,text=True,env=ENV); sec=time.perf_counter()-t
 if p.returncode: raise RuntimeError(f"{cmd[0]} failed: {p.stderr} {p.stdout}")
 return sec
def main():
 rows=[]
 for name in IMAGES:
  src=ROOT/"hpt_data"/name
  with Image.open(src) as im: rgb=ImageOps.exif_transpose(im).convert("RGB"); ref=np.asarray(rgb,dtype=np.int16); w,h=rgb.size
  sha=hashlib.sha256(src.read_bytes()).hexdigest()
  for q in QS:
   dest=OUT/f"{pathlib.Path(name).stem}_q{q}.avif"; png=OUT/f"{pathlib.Path(name).stem}_q{q}.png"
   et=run([str(BIN/"avifenc.exe"),"--codec","aom","--qcolor",str(q),"--qalpha","0","--speed","8","--jobs","1","--depth","8","--yuv","420","--range","full","--cicp","1/13/1","--ignore-exif","--ignore-xmp","--ignore-profile",str(src),str(dest)])
   dt=run([str(BIN/"avifdec.exe"),"--jobs","1","--ignore-icc",str(dest),str(png)])
   with Image.open(png) as im: got=np.asarray(im.convert("RGB"),dtype=np.int16)
   if got.shape!=ref.shape: raise RuntimeError(f"shape mismatch {name} q{q}: {got.shape} {ref.shape}")
   d=got-ref; mse=float(np.mean(d.astype(np.int32)**2,dtype=np.float64)); b=dest.stat().st_size
   rows.append({"image":name,"source_sha256":sha,"width":w,"height":h,"qcolor":q,"speed":8,"threads":1,"bytes":b,"bpp":b*8/(w*h),"rgb_mse":mse,"rgb_psnr":10*float(np.log10(255**2/mse)) if mse else None,"avif_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"pixel_sha256":hashlib.sha256(got.tobytes()).hexdigest(),"subprocess_encode_seconds_including_launch_and_io":et,"subprocess_decode_seconds_including_launch_png_io":dt})
   print(f"{name:16} q{q:2} {b:8}B {rows[-1]['bpp']:.4f}bpp mse={mse:.3f} enc={et:.3f}s dec={dt:.3f}s",flush=True)
 result={"configuration":{"libavif":"1.3.0","libaom":"3.12.1","source_manifest":"research_pass35/SOURCE_MANIFEST.json","quality":"qcolor","speed":8,"threads":1,"depth":8,"yuv":"420","range":"full","cicp":"1/13/1","metadata":"ignored","timing_note":"subprocess launch, file IO, and PNG output included; timing not compared with in-process PBC calls"},"rows":rows}
 (OUT/"avif_grid_cli.json").write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__": main()
