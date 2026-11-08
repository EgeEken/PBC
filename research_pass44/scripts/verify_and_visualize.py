import hashlib,json,pathlib,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw,ImageOps
ROOT=pathlib.Path(__file__).resolve().parents[2]; HERE=pathlib.Path(__file__).resolve().parents[1]; P43=ROOT/"research_pass43"; OUT=HERE/"outputs"; VIS=HERE/"visuals"; sys.path.insert(0,str(P43));sys.path.insert(0,str(ROOT/"research_pass42"))
import pbc4_codec_v2 as v2
from pbc4_v2_stream import unpack_v2
from pbc4_stream import unpack_bits
from PBC3 import PBC3
def main():
 summary=json.loads((OUT/"strict_budget_refinement.json").read_text()); results=[]; code="import sys,hashlib,pathlib,json;sys.path.insert(0,r'"+str(P43)+"');sys.path.insert(0,r'"+str(ROOT/"research_pass42")+"');import pbc4_codec_v2 as v; p=pathlib.Path(sys.argv[1]); a=json.loads(pathlib.Path(sys.argv[2]).read_text()); x=v.decode(p.read_bytes(),return_info=True); expected=next(r['best_within_cap']['pixel_sha256'] for r in a['rows'] if r['best_within_cap']['path']==p.name); h=hashlib.sha256(x[0].tobytes()).hexdigest(); assert h==expected and x[1]['skip_exact']; print(json.dumps({'file':p.name,'pixel_sha256':h,'skip_exact':x[1]['skip_exact'],'dimensions':[x[0].shape[1],x[0].shape[0]]}))"
 for row in summary["rows"]:
  x=row["best_within_cap"]; path=OUT/"selective"/x["path"]; p=subprocess.run([sys.executable,"-c",code,str(path),str(OUT/"strict_budget_refinement.json")],capture_output=True,text=True)
  if p.returncode:raise RuntimeError(p.stderr)
  results.append(json.loads(p.stdout));print(p.stdout.strip())
 (OUT/"independent_decode_checks.json").write_text(json.dumps({"fresh_process_checks":results,"all_exact":True,"all_skip_exact":True},indent=2)+"\n")
 name="coast.jpg"; srcpath=ROOT/"hpt_data"/name; src=np.asarray(ImageOps.exif_transpose(Image.open(srcpath)).convert("RGB")); rr=next(x for x in summary["rows"] if x["image"]==name and x["point"]=="low");best=rr["best_within_cap"]; base_row=next(x for x in rr["tested"] if x["coarse_patch_count"]==best["coarse_patch_count"] and x["selected_regions"]==0); bestpath=OUT/"selective"/best["path"]; basepath=OUT/"selective"/base_row["path"]; corrected=v2.decode(bestpath.read_bytes()); base=v2.decode(basepath.read_bytes()); v3stream=(ROOT/"research_pass42"/"outputs"/"coast_v31_compression_r3.pbc").read_bytes(); v3=np.asarray(PBC3.decompress(v3stream).image.convert("RGB")); v4a=np.asarray(Image.open(ROOT/"research_pass42"/"outputs"/"coast_v4_D_low.png").convert("RGB")); images=[("Original",src),("V3.1 Compression",v3),("V4-A D-low",v4a),("V4-B coarse-only",base),("Pass44 selected",corrected)];x0,y0=1740,1080;size=384;panels=[]
 for label,arr in images:
  crop=Image.fromarray(arr[y0:y0+size,x0:x0+size]);crop.thumbnail((384,384));panel=Image.new("RGB",(400,420),"white");panel.paste(crop,(8,28));ImageDraw.Draw(panel).text((8,7),label,fill="black");panels.append(panel)
 sheet=Image.new("RGB",(400*len(panels),420),"white")
 for i,panel in enumerate(panels):sheet.paste(panel,(i*400,0))
 VIS.mkdir(exist_ok=True);sheet.save(VIS/"coast_crop_comparison.png")
 head,payload=unpack_v2(bestpath.read_bytes());n=int.from_bytes(payload[:4],"little");pos=4+n;count=((head["height"]+15)//16)*((head["width"]+15)//16);mapbytes=(count*2+7)//8;m=np.asarray(unpack_bits(payload[pos:pos+mapbytes],count,2),dtype=np.uint8).reshape((head["height"]+15)//16,(head["width"]+15)//16);colors=np.zeros((m.shape[0],m.shape[1],3),dtype=np.uint8);colors[m==0]=(90,90,90);colors[m==1]=(40,180,70);colors[m==2]=(40,100,230);mode=Image.fromarray(colors).resize((head["width"],head["height"]),Image.Resampling.NEAREST).crop((x0,y0,x0+size,y0+size)).resize((768,768),Image.Resampling.NEAREST);mode.save(VIS/"coast_mode_map_crop.png")
 print("visuals",VIS)
if __name__=="__main__":main()
