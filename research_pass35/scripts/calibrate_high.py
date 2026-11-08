import hashlib, json, os, pathlib, subprocess, time
from PIL import Image, ImageOps
root=pathlib.Path(__file__).resolve().parents[2]; out=root/'research_pass35/outputs/calibration'; b=root/'research_pass35/outputs/binaries/reference'; env=dict(os.environ, PATH=str(b)+';C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows')
rows=[]
for name in ['flower.jpg','coast.jpg']:
 src=root/'hpt_data'/name
 with Image.open(src) as im: w,h=ImageOps.exif_transpose(im).size
 sha=hashlib.sha256(src.read_bytes()).hexdigest()
 for q in [80,90,95]:
  dst=out/f'{pathlib.Path(name).stem}_q{q}.avif'; cmd=[str(b/'avifenc.exe'),'--codec','aom','--qcolor',str(q),'--qalpha','0','--speed','8','--jobs','1','--depth','8','--yuv','420','--range','full','--cicp','1/13/1','--ignore-exif','--ignore-xmp','--ignore-profile',str(src),str(dst)]
  t=time.perf_counter(); p=subprocess.run(cmd,capture_output=True,text=True,env=env); sec=time.perf_counter()-t
  if p.returncode: raise RuntimeError(p.stderr)
  data=dst.read_bytes(); row={'image':name,'source_sha256':sha,'width':w,'height':h,'qcolor':q,'bytes':len(data),'bpp':len(data)*8/(w*h),'encode_seconds':sec,'avif_sha256':hashlib.sha256(data).hexdigest()}; rows.append(row); print(name,q,row['bytes'],f"{row['bpp']:.4f} bpp")
p=out/'calibration.json'; d=json.loads(p.read_text(encoding='utf-8-sig')); d['qcolor_grid'] += [80,90,95]; d['rows'] += rows; p.write_text(json.dumps(d,indent=2)+'\n')
