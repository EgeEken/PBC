import hashlib,json,os,pathlib,subprocess,time
import numpy as np
from PIL import Image,ImageOps
ROOT=pathlib.Path(__file__).resolve().parents[2]; BIN=ROOT/'research_pass35/outputs/binaries'; OUT=ROOT/'research_pass35/outputs/profile'; ENV=dict(os.environ,PATH=str(BIN/'reference')+';C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows')
baseline=json.loads((ROOT/'research_pass35/outputs/baseline/baseline_results.json').read_text(encoding='utf-8-sig'))
rows=[]
for r in json.loads((OUT/'profile_results.json').read_text(encoding='utf-8-sig'))['rows']:
 name,q,speed=r['image'],r['qcolor'],r['speed']; src=ROOT/'hpt_data'/name; case=OUT/f'{pathlib.Path(name).stem}_q{q}_s{speed}'; prof=case/'out.avif'; ref=case/'reference.avif'
 if speed==8:
  brow=next(x for x in baseline['rows'] if x['image']==name and x['qcolor']==q); same=hashlib.sha256(prof.read_bytes()).hexdigest()==brow['reference_avif_sha256']
 else:
  cmd=[str(BIN/'reference'/'avifenc.exe'),'--codec','aom','--qcolor',str(q),'--qalpha','0','--speed',str(speed),'--jobs','1','--depth','8','--yuv','420','--range','full','--cicp','1/13/1','--ignore-exif','--ignore-xmp','--ignore-profile',str(src),str(ref)]
  p=subprocess.run(cmd,capture_output=True,text=True,env=ENV)
  if p.returncode: raise RuntimeError(p.stderr)
  same=prof.read_bytes()==ref.read_bytes()
 png=case/'decoded_ref.png'; p=subprocess.run([str(BIN/'reference'/'avifdec.exe'),'--jobs','1','--ignore-icc',str(prof),str(png)],capture_output=True,text=True,env=ENV)
 if p.returncode: raise RuntimeError(f'avifdec failed {case}: {p.stderr}')
 with Image.open(src) as im: a=np.asarray(ImageOps.exif_transpose(im).convert('RGB'),dtype=np.int16)
 with Image.open(png) as im: b=np.asarray(im.convert('RGB'),dtype=np.int16)
 mse=float(np.mean((a-b)**2)); pillow_ok=False; pillow_mse=None; pillow_size=None; pillow_decoder=None
 try:
  with Image.open(prof) as im:
   im.load(); pillow_size=im.size; pb=np.asarray(im.convert('RGB'),dtype=np.int16); pillow_mse=float(np.mean((a-pb)**2)); pillow_decoder=im.format; pillow_ok=(pillow_size==(a.shape[1],a.shape[0]))
 except Exception as e: pillow_decoder=f'Unavailable: {type(e).__name__}: {e}'
 row={'image':name,'speed':speed,'qcolor':q,'reference_bytes_identical':same,'fresh_avifdec_ok':True,'decoded_rgb_mse':mse,'pillow_independent_decode_ok':pillow_ok,'pillow_format_or_error':pillow_decoder,'pillow_mse':pillow_mse,'pillow_size':pillow_size}
 rows.append(row); print(name,'speed',speed,'byte_equal',same,'avifdec_mse',round(mse,3),'pillow',pillow_decoder,'pillow_mse',None if pillow_mse is None else round(pillow_mse,3))
(OUT/'validation.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
if not all(x['reference_bytes_identical'] and x['fresh_avifdec_ok'] for x in rows): raise SystemExit('profile validation failed')
