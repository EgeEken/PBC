import json,os,pathlib,statistics,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[2]; BIN=ROOT/'research_pass35/outputs/binaries'; OUT=ROOT/'research_pass35/outputs/profile'; ENV=dict(os.environ,PATH=str(BIN/'reference')+';C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows')
CASES=[('googlewiki.png',5,8),('mario64.png',50,8),('flower.jpg',80,8),('cat.jpg',35,8),('coast.jpg',80,8),('googlewiki.png',5,6),('cat.jpg',35,6)]
rows=[]
def encode(exe,src,dst,q,speed):
 cmd=[str(exe),'--codec','aom','--qcolor',str(q),'--qalpha','0','--speed',str(speed),'--jobs','1','--depth','8','--yuv','420','--range','full','--cicp','1/13/1','--ignore-exif','--ignore-xmp','--ignore-profile',str(src),str(dst)]
 t=time.perf_counter(); p=subprocess.run(cmd,capture_output=True,text=True,env=ENV); sec=time.perf_counter()-t
 if p.returncode: raise RuntimeError(p.stderr)
 return sec,dst.read_bytes()
for name,q,speed in CASES:
 key=f'{pathlib.Path(name).stem}_q{q}_s{speed}'; src=ROOT/'hpt_data'/name; d=OUT/key; d.mkdir(exist_ok=True)
 ref=BIN/'reference'/'avifenc.exe'; prof=BIN/'profile-avifenc.exe'
 encode(ref,src,d/'warm_ref.avif',q,speed); encode(prof,src,d/'warm_prof.avif',q,speed)
 times={'reference':[],'instrumented':[]}; hashes=[]
 for rep in range(3):
  for label,exe in [('reference',ref),('instrumented',prof)]:
   sec,data=encode(exe,src,d/f'overhead_{rep}_{label}.avif',q,speed); times[label].append(sec); hashes.append(data)
 same=all(hashes[i]==hashes[0] for i in range(1,len(hashes)))
 row={'image':name,'qcolor':q,'speed':speed,'reference_seconds':times['reference'],'instrumented_seconds':times['instrumented'],'reference_median':statistics.median(times['reference']),'instrumented_median':statistics.median(times['instrumented']),'overhead_fraction':statistics.median(times['instrumented'])/statistics.median(times['reference'])-1,'all_bytes_identical':same}; rows.append(row)
 print(key,'ref',round(row['reference_median'],4),'prof',round(row['instrumented_median'],4),'overhead%',round(row['overhead_fraction']*100,1),'identical',same)
(OUT/'overhead.json').write_text(json.dumps({'warmups_per_build':1,'paired_repetitions':3,'rows':rows},indent=2)+'\n')
