import hashlib,json,pathlib,statistics,sys,time
import threadpoolctl
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parents[1];P43=ROOT/"research_pass43";OUT=HERE/"outputs";sys.path.insert(0,str(ROOT/"research_pass42"));sys.path.insert(0,str(P43));import pbc4_codec_v2 as v2
def main():
 summary=json.loads((OUT/"strict_budget_refinement.json").read_text());rows=[]
 with threadpoolctl.threadpool_limits(limits=1):
  for group in summary["rows"]:
   x=group["best_within_cap"];path=OUT/"selective"/x["path"];data=path.read_bytes();v2.decode(data)
   samples=[]; pixel=None
   for _ in range(5):
    t=time.perf_counter();image,info=v2.decode(data,return_info=True);samples.append(time.perf_counter()-t);pixel=hashlib.sha256(image.tobytes()).hexdigest()
   rows.append({"image":x["image"],"point":x["point"],"bytes":len(data),"stream_sha256":hashlib.sha256(data).hexdigest(),"pixel_sha256":pixel,"samples_seconds":samples,"median_seconds":statistics.median(samples),"warmups":1,"repetitions":5,"in_process":True,"threadpool_limit":1,"skip_exact":info["skip_exact"]});print(x["image"],x["point"],samples,flush=True)
 (OUT/"selected_decode_timing.json").write_text(json.dumps({"rows":rows,"timing_boundary":"complete v2 decode to original-resolution RGB; stream preloaded; 1 warmup then 5 reps"},indent=2)+"\n")
if __name__=="__main__":main()

