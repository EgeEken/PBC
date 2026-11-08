import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[2]

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save_cases(out):
    out.mkdir(parents=True,exist_ok=True); cases={}
    cases['constant']=Image.new('RGB',(320,240),(117,83,191))
    w,h=512,320
    grad=np.zeros((h,w,3),dtype=np.uint8)
    grad[:,:,0]=np.tile(np.linspace(0,255,w,dtype=np.uint8),(h,1))
    grad[:,:,1]=np.tile(np.linspace(0,255,h,dtype=np.uint8)[:,None],(1,w))
    grad[:,:,2]=(grad[:,:,0].astype(np.uint16)//2+grad[:,:,1].astype(np.uint16)//2).astype(np.uint8)
    cases['gradient']=Image.fromarray(grad)
    horizontal=Image.new('RGB',(320,256),(25,55,210)); ImageDraw.Draw(horizontal).rectangle((0,128,319,255),fill=(230,210,40)); cases['horizontal_edge']=horizontal
    vertical=Image.new('RGB',(320,256),(25,55,210)); ImageDraw.Draw(vertical).rectangle((160,0,319,255),fill=(230,210,40)); cases['vertical_edge']=vertical
    text=Image.new('RGB',(512,320),'white'); d=ImageDraw.Draw(text)
    for y in range(0,320,40): d.line((0,y,512,y),fill=(210,210,210),width=1)
    d.text((18,20),'PBC AV1 Candidate Test 0123456789',fill='black',stroke_width=0)
    d.text((18,80),'HORIZONTAL / VERTICAL / EDGE',fill=(20,60,180))
    d.rectangle((18,150,490,280),outline=(220,20,20),width=4)
    for x in range(28,490,24): d.line((x,160,x,270),fill=(30,120,50),width=2)
    cases['text_like']=text
    odd=np.zeros((257,319,3),dtype=np.uint8)
    odd[:,:,0]=np.tile(np.arange(319,dtype=np.uint16)%256,(257,1)).astype(np.uint8)
    odd[:,:,1]=np.tile(np.arange(257,dtype=np.uint16)[:,None]%256,(1,319)).astype(np.uint8)
    odd[:,:,2]=127
    cases['odd_dimensions']=Image.fromarray(odd)
    with Image.open(ROOT/'hpt_data/cat.jpg') as im:
        im=ImageOps.exif_transpose(im).convert('RGB'); x=(im.width-512)//2; y=(im.height-512)//2
        cases['photo_crop']=im.crop((x,y,x+512,y+512))
    paths={}
    for name,image in cases.items():
        p=out/f'{name}.png'; image.save(p,optimize=False); paths[name]=p
    return paths

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--exp',type=Path,required=True); ap.add_argument('--decoder',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    srcs=save_cases(a.out/'inputs'); rows=[]; env=os.environ.copy()
    for name,src in srcs.items():
        outputs=[]
        for repeat in (1,2):
            avif=a.out/f'{name}_r{repeat}.avif'; log=a.out/f'{name}_r{repeat}.log'
            cmd=[str(a.exp),'--codec','aom','--qcolor','35','--qalpha','0','--speed','6','--jobs','1','--depth','8','--yuv','420','--range','full','--cicp','1/13/1','--ignore-exif','--ignore-xmp','--ignore-profile',str(src),str(avif)]
            env['PATH']=str(a.exp.parent)+os.pathsep+r'C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows'+os.pathsep+env.get('PATH','')
            with open(log,'wb') as f: p=subprocess.run(cmd,stdout=f,stderr=f,env=env)
            if p.returncode or not avif.is_file(): raise RuntimeError(f'EXP encode failed: {name} exit={p.returncode}')
            png=a.out/f'{name}_decoded_r{repeat}.png'
            decenv=env.copy(); decenv['PATH']=str(a.decoder.parent)+os.pathsep+decenv['PATH']
            with open(a.out/f'{name}_decode_r{repeat}.log','wb') as f: d=subprocess.run([str(a.decoder),str(avif),str(png)],stdout=f,stderr=f,env=decenv)
            if d.returncode or not png.is_file(): raise RuntimeError(f'avifdec failed: {name}')
            with Image.open(src) as im: source=np.asarray(ImageOps.exif_transpose(im).convert('RGB'),dtype=np.uint8)
            with Image.open(png) as im: recon=np.asarray(im.convert('RGB'),dtype=np.uint8)
            if source.shape != recon.shape: raise RuntimeError(f'dimension mismatch: {name}: {source.shape} vs {recon.shape}')
            rows.append({'case':name,'repeat':repeat,'width':source.shape[1],'height':source.shape[0], 'source_sha256':digest(src),'avif_sha256':digest(avif),'decoded_rgb_sha256':hashlib.sha256(recon.tobytes()).hexdigest(),'bytes':avif.stat().st_size,'rgb_mse':float(np.mean((source.astype(np.float64)-recon.astype(np.float64))**2)),'encode_exit':p.returncode,'decode_exit':d.returncode})
    for name in srcs:
        pair=[r['avif_sha256'] for r in rows if r['case']==name]
        if pair[0] != pair[1]: raise RuntimeError(f'non-deterministic EXP output: {name}')
    result={'encoder':'libaom 3.12.1 PBC37_EXP','decoder':'unmodified Pass 35 libavif/libaom avifdec','cases':rows,'all_passed':True}
    (a.out/'synthetic_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('case','width','height','bytes','rgb_mse','avif_sha256')} for r in rows if r['repeat']==1],indent=2))
if __name__=='__main__': main()
