import hashlib,json,pathlib,statistics,sys,time
import threadpoolctl
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parents[1];P43=ROOT/"research_pass43";sys.path.insert(0,str(ROOT/"research_pass42"));sys.path.insert(0,str(P43));import pbc4_codec_v2 as v2
def main():
 ref=json.loads((ROOT/"research_pass43"/"outputs"/"v2_rd_comparison.json").read_text());rows=[]
 with threadpoolctl.threadpool_limits(limits=1):
  for name in ["googlewiki.png","cat.jpg","coast.jpg"]:
   x=next(x for x in ref[name]["points"]["low"]["candidates"] if x["coarse"].get("patch_count")==3);data=pathlib.Path(x["stream_path"]).read_bytes();hash0=hashlib.sha256(v2.decode(data).tobytes()).hexdigest();v2.decode(data);samples=[]
   for _ in range(5):t=time.perf_counter();out,info=v2.decode(data,return_info=True);samples.append(time.perf_counter()-t)
   rows.append({"image":name,"bytes":len(data),"file_sha256":hashlib.sha256(data).hexdigest(),"pixel_sha256":hash0,"samples_seconds":samples,"median_seconds":statistics.median(samples),"warmups":1,"repetitions":5,"in_process":True,"skip_exact":info["skip_exact"]});print(name,samples,flush=True)
 (HERE/"outputs"/"v4b_full_decode_timing.json").write_text(json.dumps({"rows":rows,"timing_boundary":"complete v2 decode to original-resolution RGB; preloaded stream; 1 warmup, 5 reps"},indent=2)+"\n")
if __name__=="__main__":main()
