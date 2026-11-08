import hashlib,json,math,os,pathlib,statistics,sys,time
import threadpoolctl,numpy as np
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parents[1];FROZEN=ROOT/"research_pass42"/"outputs"/"v31_frozen";os.environ["NUMBA_NUM_THREADS"]="1"
sys.path.insert(0,str(HERE.parent/"research_pass43"));sys.path.insert(0,str(ROOT/"research_pass42"));sys.path.insert(0,str(FROZEN))
import pbc4_codec as v1,pbc4_codec_v2 as v2,pbc3_heads
from PBC3 import PBC3
from pbc3_types import BitReader,BitWriter,PBC3Config
IMAGES=["googlewiki.png","cat.jpg","coast.jpg"];OUT=HERE/"outputs"
def encode_b(source):
 init,ref=[],[];oi,of=pbc3_heads.DownsampleInitHead.select,pbc3_heads.FillerHead.select
 def ci(self,*a,**k):r=oi(self,*a,**k);init.append(r[0]);return r
 def cf(self,*a,**k):r=of(self,*a,**k);
 # kept below as a normal wrapper to preserve all calls
 def capf(self,*a,**k):
  r=of(self,*a,**k)
  if r[0] is not None:ref.append(r[0])
  return r
 pbc3_heads.DownsampleInitHead.select=ci;pbc3_heads.FillerHead.select=capf
 try: full=PBC3.compress(Image.fromarray(source),config=PBC3Config.compression(),rng_mode="release")
 finally:pbc3_heads.DownsampleInitHead.select=oi;pbc3_heads.FillerHead.select=of
 _,raw=PBC3._open_body(full.data);header=PBC3._read_header(BitReader(raw));(_,ow,oh,ww,wh,cs,ch,cb,pb,alpha,_,_,warm,warmw,warmh,split)=header;cfg=PBC3Config.compression();prep=PBC3.prepare(Image.fromarray(source),cfg);prefixes=init+ref;count=min(3,len(prefixes));bw=BitWriter();PBC3._write_header(bw,ww,wh,ow,oh,True,prep["color_id"],ch,max(1,math.ceil(math.log2(ch))),cfg.positive_bias,alpha,count,[int(round(float(np.mean(prep["arr"][:,:,c])))) for c in range(ch)])
 for patch in prefixes[:count]:PBC3._write_patch(bw,patch,max(1,math.ceil(math.log2(ch))))
 rawp=bw.finish();prefix=PBC3.MAGIC+bytes((PBC3.VERSION,PBC3.ENTROPY_LZMA))+PBC3._entropy_pack(rawp,True)[1];return v2.encode_from_prefix(source,prefix,variant="D",point="low")["data"]
def main():
 rows=[];prev=os.getcwd();os.chdir(FROZEN)
 try:
  with threadpoolctl.threadpool_limits(limits=1):
   for name in IMAGES:
    source=v1.load_rgb(ROOT/"hpt_data"/name);expected=json.loads((ROOT/"research_pass43"/"outputs"/"v2_rd_comparison.json").read_text())[name]["points"]["low"]["candidates"];want=next(x["file_sha256"] for x in expected if x.get("coarse",{}).get("patch_count")==3)
    warm=encode_b(source)
    if hashlib.sha256(warm).hexdigest()!=want:raise RuntimeError(f"V4-B stream differs {name}")
    samples=[]
    for _ in range(3):
     t=time.perf_counter();data=encode_b(source);samples.append(time.perf_counter()-t)
     if hashlib.sha256(data).hexdigest()!=want:raise RuntimeError("repeat stream mismatch")
    rows.append({"image":name,"point":"low","coarse_patches":3,"file_bytes":len(warm),"file_sha256":want,"warmups":1,"samples_seconds":samples,"median_seconds":statistics.median(samples),"includes_p3_prefix_generation":True,"threads":1});print(name,samples,flush=True)
 finally:os.chdir(prev)
 (OUT/"v4b_complete_encode_timing.json").write_text(json.dumps({"rows":rows,"timing_boundary":"complete V3 coarse prefix generation plus V4 mode selection and encode; in-process; 1 warmup, 3 reps"},indent=2)+"\n")
if __name__=="__main__":main()

