import hashlib,json,pathlib,shutil,subprocess,sys
import numpy as np
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=ROOT/'research_pass41'/'outputs'; BEN=OUT/'benchmark'; PL=OUT/'planes'; PROBE=OUT/'binaries'/'plane_probe.exe'; FFMPEG=pathlib.Path(r'C:\ffmpeg\bin\ffmpeg.exe')
r=json.loads((OUT/'phase_c_results.json').read_text())
# One decoder check per distinct image/rate/config output; preserve stderr/stdout evidence.
ffmpeg=[]
for image,q in [('googlewiki.png',5),('googlewiki.png',20),('cat.jpg',35),('cat.jpg',50),('coast.jpg',80),('coast.jpg',90)]:
  stem=pathlib.Path(image).stem
  for config,phase in [('A','phaseB_timed3' if q in (5,35,80) else 'phaseC_high_timed3'),('B','phaseC_low_timed3' if q in (5,35,80) else 'phaseC_high_timed3')]:
    avif=BEN/f's6_{stem}_q{q}_{config}_{phase}.avif'
    png=OUT/'ffmpeg'/f'{stem}_q{q}_{config}.png'; png.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.run([str(FFMPEG),'-y','-v','error','-i',str(avif),'-frames:v','1',str(png)],capture_output=True,text=True)
    if proc.returncode: raise RuntimeError(f'FFmpeg failed for {avif}: {proc.stderr}')
    with Image.open(png) as im:
      with Image.open(ROOT/'hpt_data'/image) as src: expected=src.size
      if im.size!=expected: raise RuntimeError(f'dimension mismatch {avif}: {im.size} vs {expected}')
    ffmpeg.append({'image':image,'qcolor':q,'config':config,'avif_sha256':hashlib.sha256(avif.read_bytes()).hexdigest(),'ffmpeg_png_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'dimensions':list(expected),'valid':True})
# Native decoded planes compared using same pinned AVIF decoder and exact source-derived references.
planes=[]
for image,q in [('googlewiki.png',5),('cat.jpg',35),('coast.jpg',80)]:
  stem=pathlib.Path(image).stem; source_prefix=PL/f'{stem}_q{q}'; data={}
  for config,phase in [('A','phaseB_timed3'),('B','phaseC_low_timed3')]:
    prefix=PL/'ab'/f'{stem}_q{q}_{config}'; prefix.parent.mkdir(parents=True,exist_ok=True)
    for label in ['Y','U','V']:
      shutil.copyfile(str(source_prefix)+f'_ref_{label}.bin',str(prefix)+f'_ref_{label}.bin')
    avif=BEN/f's6_{stem}_q{q}_{config}_{phase}.avif'
    proc=subprocess.run([str(PROBE),'compare',str(avif),str(prefix)],capture_output=True,text=True)
    if proc.returncode: raise RuntimeError(f'plane compare failed for {avif}: {proc.stderr}')
    shape=(812,814) if stem=='googlewiki' else ((1200,1600) if stem=='cat' else (1080,1920))
    meta=(PL/f'{stem}_q{q}.meta').read_text().splitlines(); d=dict(line.split('=',1) for line in meta if '=' in line); w=int(d['width']); h=int(d['height'])
    local={}
    for label,sz in [('Y',w*h),('U',((w+1)//2)*((h+1)//2)),('V',((w+1)//2)*((h+1)//2))]:
      ref=np.fromfile(str(prefix)+f'_ref_{label}.bin',dtype=np.uint8,count=sz)
      dec=np.fromfile(str(prefix)+f'_decoded_{label}.bin',dtype=np.uint8,count=sz)
      delta=dec.astype(np.int16)-ref.astype(np.int16)
      local[label]={'mse':float(np.mean(delta.astype(np.float64)**2)),'changed_samples':int(np.count_nonzero(delta)),'sample_count':int(sz),'decoded_sha256':hashlib.sha256(dec.tobytes()).hexdigest()}
    rgb=np.fromfile(str(prefix)+'_decoded_rgb.bin',dtype=np.uint8)
    local['rgb_decoded_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest(); data[config]=local
  planes.append({'image':image,'qcolor':q,'A':data['A'],'B':data['B'],'delta_samples':{p:data['A'][p]['changed_samples'] if data['A'][p]['decoded_sha256']!=data['B'][p]['decoded_sha256'] else 0 for p in ['Y','U','V']}})
r['decoder_compatibility']={'pinned_avifdec':'valid for every row (see decoder_ok)','ffmpeg':ffmpeg}
r['native_plane_comparisons']=planes
(OUT/'phase_c_results.json').write_text(json.dumps(r,indent=2)+'\n')
print('FFmpeg verified',len(ffmpeg),'distinct files')
for x in planes: print(x['image'],x['qcolor'],'Y A/B mse',x['A']['Y']['mse'],x['B']['Y']['mse'],'U',x['A']['U']['mse'],x['B']['U']['mse'],'V',x['A']['V']['mse'],x['B']['V']['mse'],'changed',x['delta_samples'])
