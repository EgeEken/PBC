import hashlib,json,os,pathlib,platform,subprocess,time
from PIL import Image,ImageOps
ROOT=pathlib.Path(__file__).resolve().parents[2]; BIN=ROOT/'research_pass35'/'outputs'/'binaries'; OUT=ROOT/'research_pass35'/'outputs'/'profile'; EXE=BIN/'profile-avifenc.exe'; ENV=dict(os.environ,PATH=str(BIN)+';C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows')
CASES=[('googlewiki.png',5,8),('mario64.png',50,8),('flower.jpg',80,8),('cat.jpg',35,8),('coast.jpg',80,8),('googlewiki.png',5,6),('cat.jpg',35,6)]
rows=[]
for name,q,speed in CASES:
 src=ROOT/'hpt_data'/name; key=f'{pathlib.Path(name).stem}_q{q}_s{speed}'; d=OUT/key; d.mkdir(parents=True,exist_ok=True); dst=d/'out.avif'
 cmd=[str(EXE),'--codec','aom','--qcolor',str(q),'--qalpha','0','--speed',str(speed),'--jobs','1','--depth','8','--yuv','420','--range','full','--cicp','1/13/1','--ignore-exif','--ignore-xmp','--ignore-profile',str(src),str(dst)]
 t=time.perf_counter(); p=subprocess.run(cmd,cwd=d,capture_output=True,text=True,env=ENV); sec=time.perf_counter()-t
 (d/'stdout.log').write_text(p.stdout,encoding='utf-8'); (d/'stderr.log').write_text(p.stderr,encoding='utf-8')
 if p.returncode: raise RuntimeError(f'{key} failed: {p.stderr}')
 stats=d/'fr_part_timing_data.csv'; csv=stats.read_text(encoding='utf-8-sig') if stats.exists() else ''
 data=dst.read_bytes()
 with Image.open(src) as im: w,h=ImageOps.exif_transpose(im).size
 rows.append({'image':name,'qcolor':q,'speed':speed,'width':w,'height':h,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'bytes':len(data),'bpp':len(data)*8/(w*h),'encode_seconds_instrumented':sec,'avif_sha256':hashlib.sha256(data).hexdigest(),'partition_stats_csv':csv,'stderr_path':str(d/'stderr.log'),'partition_stats_path':str(stats) if stats.exists() else None,'timing_lines':[x for x in p.stderr.splitlines() if 'time:' in x.lower() or 'partition' in x.lower()][:50]})
 print(key,'size',len(data),'profile_csv',bool(csv),'stderr chars',len(p.stderr))
(OUT/'profile_results.json').write_text(json.dumps({'instrumentation':{'CONFIG_COLLECT_PARTITION_STATS':2,'CONFIG_COLLECT_COMPONENT_TIMING':1},'machine':{'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count()},'rows':rows},indent=2)+'\n')
