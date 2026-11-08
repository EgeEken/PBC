import hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parents[1];P43=ROOT/"research_pass43";OUT=HERE/"outputs";sys.path.insert(0,str(P43));sys.path.insert(0,str(ROOT/"research_pass42"));import pbc4_codec as v1,pbc4_codec_v2 as v2

def main():
 rows=[]
 for name,count in [("cat.jpg",19),("coast.jpg",50)]:
  stem=pathlib.Path(name).stem;source=v1.load_rgb(ROOT/"hpt_data"/name);prefix=(P43/"outputs"/f"{stem}_v31_prefix_{count:03d}.pbc").read_bytes()
  for point in ["low","intermediate"]:
   t=time.perf_counter();x=v2.encode_from_prefix(source,prefix,variant="D",point=point);elapsed=time.perf_counter()-t;path=OUT/"full_base_v4"/f"{stem}_v4_v2_{point}_p3{count:02d}.pbc";path.parent.mkdir(exist_ok=True);path.write_bytes(x["data"]);r=x["reconstruction"];d=source.astype(np.int32)-r.astype(np.int32);mse=float(np.mean(d.astype(np.int64)**2,dtype=np.float64));row={"image":name,"prefix_patch_count":count,"point":point,"bytes":len(x["data"]),"bpp":x["bpp"],"rgb_mse":mse,"base_mse":x["base_mse"],"encode_seconds_from_prefix":elapsed,"file_sha256":hashlib.sha256(x["data"]).hexdigest(),"pixel_sha256":hashlib.sha256(r.tobytes()).hexdigest(),"mode_counts":x["mode_counts"],"byte_accounting":x["byte_accounting"],"path":path.name};rows.append(row);print(row,flush=True)
 (OUT/"full_base_v4_candidates.json").write_text(json.dumps(rows,indent=2)+"\n")
if __name__=="__main__":main()
