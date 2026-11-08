import hashlib, json, os, pathlib, statistics, subprocess, time
import numpy as np
from PIL import Image, ImageOps

ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'research_pass35'/'outputs'/'baseline'
REF=ROOT/'research_pass35'/'outputs'/'binaries'/'reference'
RES=ROOT/'research_pass35'/'outputs'/'binaries'
ENV=dict(os.environ,PATH=str(REF)+';C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows')
QS={'googlewiki.png':[5,20],'mario64.png':[50,65],'flower.jpg':[80,90],'cat.jpg':[35,50],'coast.jpg':[80,90]}

def run(cmd):
 t=time.perf_counter(); p=subprocess.run(cmd,capture_output=True,text=True,env=ENV); dt=time.perf_counter()-t
 if p.returncode: raise RuntimeError(f'{cmd[0]} failed ({p.returncode}): {p.stderr}\n{p.stdout}')
 return dt,p.stdout,p.stderr

def enc(exe,src,dst,q,speed=8):
 return run([str(exe),'--codec','aom','--qcolor',str(q),'--qalpha','0','--speed',str(speed),'--jobs','1','--depth','8','--yuv','420','--range','full','--cicp','1/13/1','--ignore-exif','--ignore-xmp','--ignore-profile',str(src),str(dst)])

def dec(exe,src,dst): return run([str(exe),'--jobs','1','--ignore-icc',str(src),str(dst)])

def metrics(src,out):
 with Image.open(src) as im: a=np.asarray(ImageOps.exif_transpose(im).convert('RGB'),dtype=np.int16)
 with Image.open(out) as im: b=np.asarray(im.convert('RGB'),dtype=np.int16)
 if a.shape!=b.shape: raise RuntimeError(f'dimension mismatch {a.shape} vs {b.shape}')
 return float(np.mean((a-b)**2)),int(a.shape[1]),int(a.shape[0])

rows=[]
for name,qs in QS.items():
 src=ROOT/'hpt_data'/name
 with Image.open(src) as im: rgb=ImageOps.exif_transpose(im).convert('RGB'); w,h=rgb.size
 srcsha=hashlib.sha256(src.read_bytes()).hexdigest()
 for pi,q in enumerate(qs):
  key=f'{pathlib.Path(name).stem}_p{pi+1}_q{q}'; outdir=OUT/key; outdir.mkdir(parents=True,exist_ok=True)
  ref_avif=outdir/'reference.avif'; res_avif=outdir/'research-off.avif'
  enc(REF/'avifenc.exe',src,ref_avif,q)
  enc(RES/'research-off-avifenc.exe',src,res_avif,q)
  refbytes=ref_avif.read_bytes(); resbytes=res_avif.read_bytes(); equal=refbytes==resbytes
  if not equal: print('BYTE_MISMATCH',name,q,hashlib.sha256(refbytes).hexdigest(),hashlib.sha256(resbytes).hexdigest())
  ref_png=outdir/'decoded.png'; dec(REF/'avifdec.exe',ref_avif,ref_png); mse,_,_=metrics(src,ref_png)
  samples=[]
  # one warm-up pair, then three fresh-process paired samples
  warm_avif=outdir/'warm.avif'; warm_png=outdir/'warm.png'; enc(REF/'avifenc.exe',src,warm_avif,q); dec(REF/'avifdec.exe',warm_avif,warm_png)
  for rep in range(3):
   avif=outdir/f'timed_{rep}.avif'; png=outdir/f'timed_{rep}.png'
   et,_,_=enc(REF/'avifenc.exe',src,avif,q); dt,_,_=dec(REF/'avifdec.exe',avif,png)
   smse,_,_=metrics(src,png)
   samples.append({'encode_seconds':et,'decode_seconds':dt,'mse':smse,'avif_sha256':hashlib.sha256(avif.read_bytes()).hexdigest()})
  rows.append({'image':name,'source_sha256':srcsha,'width':w,'height':h,'point':pi+1,'qcolor':q,'speed':8,'ref_bytes':len(refbytes),'research_off_bytes':len(resbytes),'bpp':len(refbytes)*8/(w*h),'rgb_mse':mse,'reference_avif_sha256':hashlib.sha256(refbytes).hexdigest(),'research_off_avif_sha256':hashlib.sha256(resbytes).hexdigest(),'bytes_identical':equal,'timed_samples':samples,'median_encode_seconds':statistics.median(s['encode_seconds'] for s in samples),'median_decode_seconds':statistics.median(s['decode_seconds'] for s in samples)})
  print(f'{name:16} p{pi+1} q={q:2} {len(refbytes):8}B {rows[-1]["bpp"]:.4f}bpp mse={mse:.2f} byte_equal={equal} enc={rows[-1]["median_encode_seconds"]:.3f}s dec={rows[-1]["median_decode_seconds"]:.3f}s')
result={'configuration':{'av1':'libaom 3.12.1','libavif':'1.3.0','speed':8,'jobs':1,'depth':8,'yuv':'420','range':'full','cicp':'1/13/1','metadata':'ignored','sharp_yuv':False,'timed_repetitions':3,'warmup_pairs':1},'rows':rows}
(OUT/'baseline_results.json').write_text(json.dumps(result,indent=2)+'\n')
if not all(r['bytes_identical'] for r in rows): raise SystemExit('reference parity FAILED')
