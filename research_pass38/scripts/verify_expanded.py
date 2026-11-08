import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path.cwd()
cases=[("googlewiki","googlewiki.png",5),("mario64","mario64.png",50),("cat","cat.jpg",35)]
exp=ROOT/"research_pass38/outputs/binaries-expand/exp/avifenc.exe"
dec=ROOT/"research_pass35/outputs/binaries/reference/avifdec.exe"
ffmpeg=Path(r"C:\ffmpeg\bin\ffmpeg.exe")
rows=[]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for key,name,q in cases:
 d=ROOT/"research_pass38/outputs/expanded"/key
 src=ROOT/"hpt_data"/name
 original=d/"exp.avif"
 dup=d/"exp_repeat.avif"
 args=[str(exp),"--codec","aom","--qcolor",str(q),"--qalpha","0","--speed","6","--jobs","1","--depth","8","--yuv","420","--range","full","--cicp","1/13/1","--ignore-exif","--ignore-xmp","--ignore-profile",str(src),str(dup)]
 env=os.environ.copy(); env["PATH"]=str(exp.parent)+os.pathsep+r"C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows"
 t=time.perf_counter(); p=subprocess.run(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env); enc=time.perf_counter()-t
 if p.returncode or sha(original)!=sha(dup): raise RuntimeError(f"nondeterministic EXP: {key}")
 decs={}
 for label in ("ref","exp"):
  avif=d/f"{label}.avif"; png=d/f"{label}_timed.png"
  e=os.environ.copy(); e["PATH"]=str(dec.parent)+os.pathsep+r"C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows"
  t=time.perf_counter(); rc=subprocess.run([str(dec),"--jobs","1","--ignore-icc",str(avif),str(png)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=e).returncode; secs=time.perf_counter()-t
  if rc: raise RuntimeError(f"pinned decoder failed {key} {label}")
  decs[label]=secs
 ffcmd=[str(ffmpeg),"-v","error","-threads","1","-i",str(original),"-f","null","-"]
 fp=subprocess.run(ffcmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 if fp.returncode: raise RuntimeError(f"FFmpeg decode failed {key}: {fp.stderr.decode(errors='replace')}")
 rows.append({"image":name,"EXP_repeat_deterministic":True,"repeat_sha256":sha(dup),"repeat_encode_seconds_exploratory":enc,"avifdec_seconds":{"REF":decs["ref"],"EXP":decs["exp"]},"ffmpeg_decode":True})
out=ROOT/"research_pass38/outputs/expanded/verification.json"
out.write_text(json.dumps(rows,indent=2),encoding="utf-8")
print(json.dumps(rows,indent=2))
