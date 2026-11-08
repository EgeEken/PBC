import hashlib,json,math,os,pathlib,statistics,sys,time
import threadpoolctl,numpy as np
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parents[1];FROZEN=ROOT/"research_pass42"/"outputs"/"v31_frozen";os.environ["NUMBA_NUM_THREADS"]="1";sys.path.insert(0,str(HERE.parent/"research_pass43"));sys.path.insert(0,str(ROOT/"research_pass42"));sys.path.insert(0,str(FROZEN))
import pbc4_codec as v1,pbc4_codec_v2 as v2,pbc3_heads
from PBC3 import PBC3
from pbc3_types import BitReader,BitWriter,PBC3Config
OUT=HERE/"outputs"
def enc(source):
 init,ref=[],[];oi,of=pbc3_heads.DownsampleInitHead.select,pbc3_heads.FillerHead.select
 def ci(self,*a,**k):r=oi(self,*a,**k);init.append(r[0]);return r
 def cf(self,*a,**k):r=of(self,*a,**k);
 def capf(self,*a,**k):
  r=of(self,*a,**k)
  if r[0] is not None:ref.append(r[0])
  return r
 pbc3_heads.DownsampleInitHead.select=ci;pbc3_heads.FillerHead.select=capf
 try:full=PBC3.compress(Image.fromarray(source),config=PBC3Config.compression(),rng_mode="release")
 finally:pbc3_heads.DownsampleInitHead.select=oi;pbc3_heads.FillerHead.select=of
 _,raw=PBC3._open_body(full.data);hd=PBC3._read_header(BitReader(raw));(_,ow,oh,ww,wh,cs,ch,cb,pb,alpha,_,_,warm,warmw,warmh,split)=hd;cfg=PBC3Config.compression();prep=PBC3.prepare(Image.fromarray(source),cfg);patches=init+ref;count=min(3,len(patches));bw=BitWriter();PBC3._write_header(bw,ww,wh,ow,oh,True,prep["color_id"],ch,max(1,math.ceil(math.log2(ch))),cfg.positive_bias,alpha,count,[int(round(float(np.mean(prep["arr"][:,:,c]))) ) for c in range(ch)])
 for patch in patches[:count]:PBC3._write_patch(bw,patch,max(1,math.ceil(math.log2(ch))))
 rawp=bw.finish();prefix=PBC3.MAGIC+bytes((PBC3.VERSION,PBC3.ENTROPY_LZMA))+PBC3._entropy_pack(rawp,True)[1];return v2.encode_from_prefix(source,prefix,variant="D",point="intermediate")["data"]
def main():
 prev=os.getcwd();os.chdir(FROZEN)
 try:
  source=v1.load_rgb(ROOT/"hpt_data"/"googlewiki.png");expected=next(x["file_sha256"] for x in json.loads((ROOT/"research_pass43"/"outputs"/"v2_rd_comparison.json").read_text())["googlewiki.png"]["points"]["intermediate"]["candidates"] if x["coarse"].get("patch_count")==3)
  with threadpoolctl.threadpool_limits(limits=1):
   warm=enc(source);assert hashlib.sha256(warm).hexdigest()==expected;samples=[]
   for _ in range(3):t=time.perf_counter();data=enc(source);samples.append(time.perf_counter()-t);assert hashlib.sha256(data).hexdigest()==expected
  result={"image":"googlewiki.png","point":"intermediate","bytes":len(warm),"file_sha256":expected,"samples_seconds":samples,"median_seconds":statistics.median(samples),"warmups":1,"includes_coarse_prefix_generation":True,"numba_threads":1};(OUT/"v4b_intermediate_google_timing.json").write_text(json.dumps(result,indent=2)+"\n");print(result)
 finally:os.chdir(prev)
if __name__=="__main__":main()
