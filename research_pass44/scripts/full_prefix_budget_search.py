import hashlib,json,math,pathlib,struct,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parents[1];P43=ROOT/"research_pass43";OUT=HERE/"outputs";sys.path.insert(0,str(P43));sys.path.insert(0,str(ROOT/"research_pass42"))
import pbc4_codec as v1,pbc4_codec_v2 as v2
from pbc4_v2_stream import unpack_v2,pack_v2
from pbc4_stream import pack_bits
from strict_budget_refinement import parse_stream,mse,p3_count
def main():
 bench=json.loads((ROOT/"research_pass42"/"outputs"/"benchmark_raw.json").read_text());results=[]
 for name,count in [("cat.jpg",19),("coast.jpg",50)]:
  src=v1.load_rgb(ROOT/"hpt_data"/name);stem=pathlib.Path(name).stem
  for point in ["low","intermediate"]:
   preset="compression" if point=="low" else "quality";target=bench["v31"][f"{name}|{preset}"]["rows"][0]["bytes"];cap=math.floor(target*1.05);path=OUT/"full_base_v4"/f"{stem}_v4_v2_{point}_p3{count:02d}.pbc";data=path.read_bytes();hdr,coarse_field,orig,records=parse_stream(data);n=int.from_bytes(coarse_field[:4],"little");base=v2.decode_coarse_body(coarse_field[4:4+n]);full=v2.decode(data);modes=orig.ravel();h,w=src.shape[:2];cols=orig.shape[1];gains=[]
   for i,mode in enumerate(modes):
    if not mode:gains.append(0.);continue
    row,col=divmod(i,cols);y0,x0=row*16,col*16;hh=min(16,h-y0);ww=min(16,w-x0);gains.append(mse(src[y0:y0+hh,x0:x0+ww],base[y0:y0+hh,x0:x0+ww])-mse(src[y0:y0+hh,x0:x0+ww],full[y0:y0+hh,x0:x0+ww]))
   order=sorted((i for i,m in enumerate(modes) if m and gains[i]>0),key=lambda i:gains[i]/max(1,len(records[i])),reverse=True);coarse=p3_count(coarse_field[4:4+n]);cache={}
   def evaluate(k):
    if k in cache:return cache[k]
    chosen=set(order[:k]);newm=np.zeros(len(modes),dtype=np.uint8)
    for i in chosen:newm[i]=modes[i]
    payload=coarse_field+pack_bits(newm,2)+b"".join(records[i] for i in range(len(modes)) if newm[i]);stream=pack_v2(hdr["width"],hdr["height"],hdr["coarse_width"],hdr["coarse_height"],hdr["procedure_step"],hdr["transform_step"],hdr["rd_lambda"],payload);rec=v2.decode(stream);row={"coarse_patch_count":coarse,"selected_regions":len(chosen),"bytes":len(stream),"rate_gap_percent":100*(len(stream)/target-1),"within_cap":len(stream)<=cap,"rgb_mse":mse(src,rec),"file_sha256":hashlib.sha256(stream).hexdigest(),"pixel_sha256":hashlib.sha256(rec.tobytes()).hexdigest(),"mode_counts":{"skip":int(np.count_nonzero(newm==0)),"procedural":int(np.count_nonzero(newm==1)),"transform":int(np.count_nonzero(newm==2))}};out=OUT/"selective"/f"{stem}_{point}_fullp{coarse}_k{k}.pbc";out.write_bytes(stream);row["path"]=out.name;cache[k]=row;return row
   low,high=0,512;tested=[]
   while low<=high and len(tested)<10:
    k=(low+high)//2;x=evaluate(k);tested.append(x)
    if x["within_cap"]:low=k+1
    else:high=k-1
   feasible=[x for x in tested if x["within_cap"]];best=min(feasible,key=lambda x:x["rgb_mse"]);basefile=evaluate(0);results.append({"image":name,"point":point,"target_preset":preset,"target_bytes":target,"hard_cap_bytes":cap,"coarse_patch_count":count,"source_stream":path.name,"base_only":basefile,"search_evaluations":tested,"best_feasible":best});print(name,point,"base",basefile["bytes"],round(basefile["rgb_mse"],3),"best",best["bytes"],best["selected_regions"],round(best["rgb_mse"],3),flush=True)
 (OUT/"full_prefix_budget_search.json").write_text(json.dumps({"method":"replace truncated p13 choice with complete V3.1 Compression prefix; bounded top-512 subset binary search, max 10 actual completed-file evaluations per target","results":results},indent=2)+"\n")
if __name__=="__main__":main()
