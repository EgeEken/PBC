import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = [("googlewiki.png", 5, 6), ("googlewiki.png", 20, 6), ("mario64.png", 50, 6),
         ("flower.jpg", 80, 6), ("cat.jpg", 35, 6), ("cat.jpg", 50, 6),
         ("coast.jpg", 80, 6), ("googlewiki.png", 5, 8), ("cat.jpg", 35, 8)]

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ref',type=Path,required=True); ap.add_argument('--ctrl',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); rows=[]
    for image,q,speed in CASES:
        src=ROOT/'hpt_data'/image; case=f'{Path(image).stem}_q{q}_s{speed}'; outputs={}
        for name,exe in [('REF',a.ref),('CTRL',a.ctrl)]:
            dst=a.out/f'{case}_{name.lower()}.avif'; env=os.environ.copy(); env['PATH']=str(exe.parent)+os.pathsep+r'C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows'+os.pathsep+env.get('PATH','')
            cmd=[str(exe),'--codec','aom','--qcolor',str(q),'--qalpha','0','--speed',str(speed),'--jobs','1','--depth','8','--yuv','420','--range','full','--cicp','1/13/1','--ignore-exif','--ignore-xmp','--ignore-profile',str(src),str(dst)]
            with open(a.out/f'{case}_{name.lower()}.log','wb') as log: p=subprocess.run(cmd,stdout=log,stderr=log,env=env)
            if p.returncode or not dst.is_file(): raise RuntimeError(f'{name} encode failed for {case}: {p.returncode}')
            outputs[name]={'bytes':dst.stat().st_size,'sha256':digest(dst),'path':str(dst)}
        row={'image':image,'qcolor':q,'speed':speed,'ref':outputs['REF'],'ctrl':outputs['CTRL'],'byte_identical':outputs['REF']['sha256']==outputs['CTRL']['sha256']}
        rows.append(row)
        if not row['byte_identical']: raise RuntimeError(f'CTRL differs from REF at {case}')
    result={'all_byte_identical':True,'cases':rows}; (a.out/'ctrl_parity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
