import hashlib, json, pathlib, statistics, subprocess, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / 'research_pass41'
OUT = PASS / 'outputs'
sys.path.insert(0, str(ROOT / 'research_pass39' / 'scripts'))
import benchmark as p39
REF_BIN = ROOT / 'research_pass35' / 'outputs' / 'binaries' / 'reference'
p39.BIN = REF_BIN
p39.ENV = dict(p39.os.environ, PATH=str(REF_BIN) + r';C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows')
p39.OUT = OUT / 'benchmark'
p39.OUT.mkdir(parents=True, exist_ok=True)
base = ['--advanced','aq-mode=0','--advanced','tune=psnr','--advanced','deltaq-mode=0','--advanced','sb-size=64']
p39.CONFIGS = {'A': base, 'B': base + ['--advanced','enable-chroma-deltaq=1']}
points = [('googlewiki.png',5),('googlewiki.png',20),('cat.jpg',35),('cat.jpg',50),('coast.jpg',80),('coast.jpg',90)]
result_path = OUT / 'phase_c_results.json'
old = json.loads(result_path.read_text()) if result_path.exists() else {'sources':{},'settings':{},'rows':[],'comparisons':[]}
old['sources'] = {'libaom_commit':'10aece4157eb79315da205f39e19bf6ab3ee30d0','libavif_commit':'1aadfad932c98c069a1204261b1856f81f3bc199'}
old['settings'] = {'speed':6,'jobs':1,'depth':8,'yuv':'420','range':'full','cicp':'1/13/1','warmups':1,'timed_repetitions':3,'configurations':{'A':base,'B':base+['--advanced','enable-chroma-deltaq=1']}}
existing = json.loads((OUT/'phase_b_results.json').read_text())
for row in existing['rows']:
    if row['phase']=='phaseB_timed' and row['repeat']==3:
        old['rows'] = [x for x in old['rows'] if not (x.get('image')==row['image'] and x.get('qcolor')==row['qcolor'] and x.get('config')=='A' and x.get('phase','').startswith('phaseC'))]
        copied = dict(row); copied['phase']='phaseC_reused_phaseB'; copied['point']='low'; old['rows'].append(copied)
old['reference_avifenc_sha256'] = p39.digest(REF_BIN/'avifenc.exe')
old['ssimulacra2'] = str(p39.SSIMULACRA)

def save(): result_path.write_text(json.dumps(old,indent=2)+'\n')
def encode_point(image,q,config,point_label):
    src=ROOT/'hpt_data'/image
    reps=[('warm',0),* [('timed',i) for i in range(1,4)]]
    for phase,rep in reps:
        row,_,_=p39.one(src,image,q,config,6,rep,f'phaseC_{point_label}_{phase}')
        row['source_sha256']=p39.digest(src); row['binary_sha256']=old['reference_avifenc_sha256']; row['point']=point_label
        old['rows'].append(row); save()
        print(f'{image:14} q{q:2} {config} {phase}{rep}: {row["bytes"]:8} B MSE {row["rgb_mse"]:9.4f} SS2 {row["ssimulacra2"]:8.3f} enc {row["encode_seconds"]:.3f}s',flush=True)

for image,q in [('googlewiki.png',5),('cat.jpg',35),('coast.jpg',80)]: encode_point(image,q,'B','low')
for image,q in [('googlewiki.png',20),('cat.jpg',50),('coast.jpg',90)]:
    order=['A','B']
    for phase,rep in [('warm',0),* [('timed',i) for i in range(1,4)]]:
        cfg_order=order if rep%2==1 else order[::-1]
        for config in cfg_order:
            src=ROOT/'hpt_data'/image
            row,_,_=p39.one(src,image,q,config,6,rep,f'phaseC_high_{phase}')
            row['source_sha256']=p39.digest(src); row['binary_sha256']=old['reference_avifenc_sha256']; row['point']='high'
            old['rows'].append(row); save()
            print(f'{image:14} q{q:2} {config} {phase}{rep}: {row["bytes"]:8} B MSE {row["rgb_mse"]:9.4f} SS2 {row["ssimulacra2"]:8.3f} enc {row["encode_seconds"]:.3f}s',flush=True)
# Native chroma planes for low-point B bitstreams; A planes already exist from Phase B.
probe=OUT/'binaries'/'plane_probe.exe'
for image,q in [('googlewiki.png',5),('cat.jpg',35),('coast.jpg',80)]:
    prefix=OUT/'planes'/f'{pathlib.Path(image).stem}_q{q}'
    bfile=p39.OUT/f's6_{pathlib.Path(image).stem}_q{q}_B_phaseC_low_timed3.avif'
    proc=subprocess.run([str(probe),'compare',str(bfile),str(prefix)],capture_output=True,text=True,env=p39.ENV)
    if proc.returncode: raise RuntimeError(f'plane probe failed: {proc.stderr}')
    old['comparisons'].append({'image':image,'qcolor':q,'config':'B','avif_sha256':p39.digest(bfile),'probe_stdout':proc.stdout.strip()})
    save(); print(f'plane B {image} q{q}: {proc.stdout.strip()}',flush=True)
