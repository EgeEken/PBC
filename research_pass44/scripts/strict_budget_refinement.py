import hashlib,json,math,pathlib,struct,sys
import numpy as np
from PIL import Image,ImageOps
ROOT=pathlib.Path(__file__).resolve().parents[2]; HERE=pathlib.Path(__file__).resolve().parents[1]; P43=ROOT/"research_pass43"; OUT=HERE/"outputs"; sys.path.insert(0,str(P43));sys.path.insert(0,str(ROOT/"research_pass42"))
import pbc4_codec as v1
import pbc4_codec_v2 as v2
from pbc4_v2_stream import unpack_v2,pack_v2
from pbc4_stream import unpack_bits,pack_bits
from pbc3_types import BitReader
from PBC3 import PBC3
IMAGES=["googlewiki.png","cat.jpg","coast.jpg"]; POINTS=["low","intermediate"]; K_CHOICES=[0,8,32,128,512]
def p3_count(raw):
 stream=PBC3.MAGIC+bytes((PBC3.VERSION,PBC3.ENTROPY_STORE))+raw
 _,body=PBC3._open_body(stream);h=PBC3._read_header(BitReader(body));return int(h[10])
def parse_stream(data):
 header,payload=unpack_v2(data);n=struct.unpack_from("<I",payload,0)[0];coarse_end=4+n;rows,cols=(header["height"]+15)//16,(header["width"]+15)//16;count=rows*cols;map_n=(count*2+7)//8;modes=np.asarray(unpack_bits(payload[coarse_end:coarse_end+map_n],count,2),dtype=np.uint8).reshape(rows,cols);pos=coarse_end+map_n;records=[]
 for mode in modes.ravel():
  start=pos
  if mode==1:
   subtype=payload[pos];pos+=1;pos+=6 if subtype==0 else 24
  elif mode==2:
   for _ in range(12):_,pos=v1._read_coefficients(payload,pos)
  records.append(bytes(payload[start:pos]) if mode else b"")
 if pos!=len(payload): raise RuntimeError("unexpected trailing bytes")
 return header,payload[:coarse_end],modes,records
def mse(a,b):
 d=a.astype(np.int32)-b.astype(np.int32);return float(np.mean(d.astype(np.int64)**2,dtype=np.float64))
def main():
 allrows=[]
 for name in IMAGES:
  src=v1.load_rgb(ROOT/"hpt_data"/name); stem=pathlib.Path(name).stem; outrows=[]
  for point in POINTS:
   target_preset="compression" if point=="low" else "quality"; target=json.loads((ROOT/"research_pass42"/"outputs"/"benchmark_raw.json").read_text())["v31"][f"{name}|{target_preset}"]["rows"][0]["bytes"]; cap=math.floor(target*1.05)
   files=sorted((P43/"outputs").glob(f"{stem}_v4_v2_{point}_p30*.pbc"))
   if not files: raise RuntimeError(f"no v2 candidates for {name} {point}")
   for path in files:
    data=path.read_bytes(); hdr,coarse_field,orig_modes,records=parse_stream(data); coarse_len=struct.unpack_from("<I",coarse_field,0)[0]; raw_body=coarse_field[4:4+coarse_len]; base=v2.decode_coarse_body(raw_body); full,info=v2.decode(data,return_info=True); count=orig_modes.size; original_flat=orig_modes.ravel(); gains=[]; region=16; h,w=src.shape[:2]
    for i,mode in enumerate(original_flat):
     if not mode: gains.append(0.0);continue
     row,col=divmod(i,orig_modes.shape[1]);y0,x0=row*region,col*region;hh=min(region,h-y0);ww=min(region,w-x0);g=mse(src[y0:y0+hh,x0:x0+ww],base[y0:y0+hh,x0:x0+ww])-mse(src[y0:y0+hh,x0:x0+ww],full[y0:y0+hh,x0:x0+ww]);gains.append(g)
    order=sorted((i for i,m in enumerate(original_flat) if m and gains[i]>0),key=lambda i:gains[i]/max(1,len(records[i])),reverse=True); coarse_count=p3_count(raw_body); stemout=f"{stem}_{point}_p3{coarse_count}"
    for k in K_CHOICES:
     selected=set(order[:k]); modes=np.zeros(count,dtype=np.uint8);outpayload=bytearray(coarse_field);records_bytes=0
     for i in sorted(selected):modes[i]=original_flat[i];outpayload.extend(records[i]);records_bytes+=len(records[i])
     # Records are serialized in region raster order, matching the map.
     mapbytes=pack_bits(modes,2); final_payload=coarse_field+mapbytes+bytes().join(records[i] for i in range(count) if modes[i]); packed=pack_v2(hdr["width"],hdr["height"],hdr["coarse_width"],hdr["coarse_height"],hdr["procedure_step"],hdr["transform_step"],hdr["rd_lambda"],final_payload); recon,decinfo=v2.decode(packed,return_info=True); b=len(packed);row={"image":name,"point":point,"reference_preset":target_preset,"target_bytes":target,"hard_cap_bytes":cap,"coarse_patch_count":coarse_count,"source_stream":path.name,"selected_regions":len(selected),"selected_payload_bytes":records_bytes,"bytes":b,"rate_gap_percent":100*(b/target-1),"within_cap":b<=cap,"bpp":b*8/(w*h),"rgb_mse":mse(src,recon),"mode_counts":decinfo["mode_counts"],"skip_exact":decinfo["skip_exact"],"file_sha256":hashlib.sha256(packed).hexdigest(),"pixel_sha256":hashlib.sha256(recon.tobytes()).hexdigest()}; outpath=OUT/"selective"/f"{stemout}_k{k}.pbc";outpath.parent.mkdir(parents=True,exist_ok=True);outpath.write_bytes(packed);row["path"]=outpath.name;outrows.append(row);print(name,point,"p3",coarse_count,"K",k,"bytes",b,"cap",cap,"mse",round(row["rgb_mse"],3),flush=True)
   eligible=[x for x in outrows if x["within_cap"]]; chosen=min(eligible,key=lambda x:x["rgb_mse"]); allrows.append({"image":name,"point":point,"target_preset":target_preset,"target_bytes":target,"hard_cap_bytes":cap,"tested":outrows,"best_within_cap":chosen})
 result={"method":"rank existing V4 mode records by measured per-region RGB MSE reduction per serialized record byte; test K=0,8,32,128 per existing coarse prefix; select only using final XZ file size and RGB MSE","format_changed":False,"decoder_changed":False,"new_mode_added":False,"k_choices":K_CHOICES,"rows":allrows};(OUT/"strict_budget_refinement.json").write_text(json.dumps(result,indent=2)+"\n");print("saved",OUT/"strict_budget_refinement.json")
if __name__=="__main__":main()

