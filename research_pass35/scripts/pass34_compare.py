import os,pathlib,subprocess,numpy as np
from PIL import Image,ImageOps
root=pathlib.Path.cwd(); b=root/'research_pass35/outputs/binaries/reference'; env=dict(os.environ,PATH=str(b)+';C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows'); src=root/'hpt_data/coast.jpg'
with Image.open(src) as im:a=np.asarray(ImageOps.exif_transpose(im).convert('RGB'),dtype=np.int16)
for q in [35,50]:
 avif=root/f'research_pass35/outputs/calibration/coast_q{q}.avif'; png=avif.with_suffix('.historical-check.png'); p=subprocess.run([str(b/'avifdec.exe'),'--jobs','1','--ignore-icc',str(avif),str(png)],capture_output=True,text=True,env=env)
 if p.returncode: raise RuntimeError(p.stderr)
 with Image.open(png) as im:c=np.asarray(im.convert('RGB'),dtype=np.int16)
 print(f'q={q} bytes={avif.stat().st_size} mse={np.mean((a-c)**2):.6f}')
