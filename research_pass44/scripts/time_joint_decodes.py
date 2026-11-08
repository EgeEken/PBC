import hashlib,json,pathlib,statistics,sys,time
import threadpoolctl
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parents[1];P43=ROOT/"research_pass43";sys.path.insert(0,str(ROOT/"research_pass42"));sys.path.insert(0,str(P43));import pbc4_codec_v2 as v2
def main():
 j=json.loads((HERE/"outputs"/"final_joint_allocation.json").read_text());rows=[]
 with threadpoolctl.threadpool_limits(limits=1):
  for group in j["results"]:
   x=group["C_joint_best"];data=(HERE/"outputs"/"selective"/x["path"]).read_bytes();v2.decode(data);samples=[]
   for _ in range(5):t=time.perf_counter();rec,info=v2.decode(data,return_info=True);samples.append(time.perf_counter()-t)
   rows.append({"image":group["image"],"point":group["point"],"bytes":len(data),"stream_sha256":hashlib.sha256(data).hexdigest(),"pixel_sha256":hashlib.sha256(rec.tobytes()).hexdigest(),"samples_seconds":samples,"median_seconds":statistics.median(samples),"warmups":1,"repetitions":5,"skip_exact":info["skip_exact"]});print(group["image"],group["point"],samples,flush=True)
 (HERE/"outputs"/"joint_decode_timing.json").write_text(json.dumps({"rows":rows,"timing_boundary":"complete v2 decode to RGB, stream preloaded, one BLAS thread"},indent=2)+"\n")
if __name__=="__main__":main()

