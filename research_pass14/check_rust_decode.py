import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from PBC3 import PBC3

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--rust-bin',required=True)
 args=ap.parse_args()
 fixtures=json.loads((HERE/'native_cases'/'fixtures.json').read_text(encoding='utf-8'))
 rows=[]
 for item in fixtures:
  path=HERE/'native_cases'/item['file']
  proc=subprocess.run([args.rust_bin,'decode',str(path)],capture_output=True,text=True)
  out=path.with_suffix('.png')
  if proc.returncode:
   rows.append(dict(name=item['name'],origin=item['origin'],status='rust_error',error=proc.stderr[-2000:])); continue
  python=PBC3.decompress(path.read_bytes()).image.convert('RGBA')
  with Image.open(out) as im: rust=im.convert('RGBA')
  a=np.asarray(python); b=np.asarray(rust)
  same_size=python.size==rust.size
  if same_size:
   neq=np.any(a!=b,axis=2); coords=np.argwhere(neq)
   first=None
   if len(coords):
    y,x=map(int,coords[0]); first=dict(x=x,y=y,python_rgba=a[y,x].tolist(),rust_rgba=b[y,x].tolist())
   rows.append(dict(name=item['name'],origin=item['origin'],format_version=item['format_version'],entropy_method=item['entropy_method'],channels=item['channels'],patch_count=item['patch_count'],iterative_patches=item['iterative_patches'],working_size=item['working_size'],original_size=item['original_size'],dimensions_equal=True,pixel_identical=not len(coords),different_pixels=int(len(coords)),different_channel_values=int(np.count_nonzero(a!=b)),first_difference=first))
  else:
   rows.append(dict(name=item['name'],origin=item['origin'],dimensions_equal=False,python_size=list(python.size),rust_size=list(rust.size),pixel_identical=False))
  print(rows[-1],flush=True)
 result=dict(rust_binary=args.rust_bin,fixtures=len(fixtures),exact_pixel_matches=sum(r.get('pixel_identical',False) for r in rows),results=rows)
 (HERE/'rust_correctness.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__': main()
