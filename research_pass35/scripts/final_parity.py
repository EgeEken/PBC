import hashlib,json,os,pathlib,subprocess
import numpy as np
from PIL import Image
R=pathlib.Path(__file__).resolve().parents[2]; P=R/'research_pass35'; B=R/'research_pass35/outputs/binaries'; O=R/'research_pass35/outputs/equivalence'; REF=B/'reference'; ENV=dict(os.environ,PATH=str(REF)+';C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows')
QS={'googlewiki.png':[5,20],'mario64.png':[50,65],'flower.jpg':[80,90],'cat.jpg':[35,50],'coast.jpg':[80,90]}; rows=[]
def run(cmd):
 p=subprocess.run(cmd,capture_output=True,text=True,env=ENV)
 if p.returncode: raise RuntimeError(p.stderr)
for name,qs in QS.items():
 src=R/'hpt_data'/name
 for idx,q in enumerate(qs,1):
  d=O/f'{pathlib.Path(name).stem}_p{idx}'; d.mkdir(parents=True,exist_ok=True); files=[]
  for label,exe in [('reference',REF/'avifenc.exe'),('research-off',B/'research-off-avifenc.exe')]:
   av=d/f'{label}.avif'; run([str(exe),'--codec','aom','--qcolor',str(q),'--qalpha','0','--speed','8','--jobs','1','--depth','8','--yuv','420','--range','full','--cicp','1/13/1','--ignore-exif','--ignore-xmp','--ignore-profile',str(src),str(av)])
   png=d/f'{label}.png'; run([str(REF/'avifdec.exe'),'--jobs','1','--ignore-icc',str(av),str(png)])
   with Image.open(png) as im: pixels=np.asarray(im.convert('RGB'))
   files.append((av.read_bytes(),pixels))
  equal=files[0][0]==files[1][0]; pixels_equal=np.array_equal(files[0][1],files[1][1]); rows.append({'image':name,'qcolor':q,'bytes_identical':equal,'decoded_rgb_identical':pixels_equal,'avif_sha256':hashlib.sha256(files[0][0]).hexdigest(),'bytes':len(files[0][0])}); print(name,q,equal,pixels_equal)
(P/'outputs/equivalence/final_control_parity.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
if not all(x['bytes_identical'] and x['decoded_rgb_identical'] for x in rows): raise SystemExit('parity failed')

