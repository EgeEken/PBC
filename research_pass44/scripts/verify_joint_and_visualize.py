import hashlib,json,pathlib,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw,ImageOps
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parents[1];P43=ROOT/"research_pass43";OUT=HERE/"outputs";VIS=HERE/"visuals";sys.path.insert(0,str(P43));sys.path.insert(0,str(ROOT/"research_pass42"))
import pbc4_codec_v2 as v2
from pbc4_v2_stream import unpack_v2
from pbc4_stream import unpack_bits
from PBC3 import PBC3
def main():
 data=json.loads((OUT/"final_joint_allocation.json").read_text());checks=[];code="import sys,hashlib,pathlib,json;sys.path.insert(0,r'"+str(ROOT/"research_pass42")+"');sys.path.insert(0,r'"+str(P43)+"');import pbc4_codec_v2 as v; d=pathlib.Path(sys.argv[1]).read_bytes(); x,info=v.decode(d,return_info=True);h=hashlib.sha256(x.tobytes()).hexdigest();assert h==sys.argv[2] and info['skip_exact'];print(json.dumps({'pixel_sha256':h,'skip_exact':info['skip_exact'],'shape':list(x.shape)}))"
 for row in data["results"]:
  x=row["C_joint_best"];path=OUT/"selective"/x["path"];p=subprocess.run([sys.executable,"-c",code,str(path),x["pixel_sha256"]],capture_output=True,text=True)
  if p.returncode:raise RuntimeError(p.stderr)
  checks.append({"image":row["image"],"point":row["point"],"file":path.name,**json.loads(p.stdout)});print(path.name,p.stdout.strip())
 (OUT/"independent_decode_checks.json").write_text(json.dumps({"checks":checks,"all_exact":True,"all_skip_exact":True},indent=2)+"\n")
 def panels(pathspec, box, outname):
  images=[]
  for label,array in pathspec:
   x0,y0,x1,y1=box;crop=Image.fromarray(array[y0:y1,x0:x1]);crop.thumbnail((420,380));panel=Image.new("RGB",(440,415),"white");panel.paste(crop,(10,28));ImageDraw.Draw(panel).text((10,7),label,fill="black");images.append(panel)
  sheet=Image.new("RGB",(440*len(images),415),"white")
  for i,panel in enumerate(images):sheet.paste(panel,(440*i,0))
  sheet.save(VIS/outname)
 def chosen(name,point):return next(x["C_joint_best"] for x in data["results"] if x["image"]==name and x["point"]==point)
 VIS.mkdir(exist_ok=True)
 # Text and sharp edges on Googlewiki.
 name="googlewiki.png";src=np.asarray(ImageOps.exif_transpose(Image.open(ROOT/"hpt_data"/name)).convert("RGB"));v3=np.asarray(PBC3.decompress((ROOT/"research_pass42"/"outputs"/"googlewiki_v31_compression_r3.pbc").read_bytes()).image.convert("RGB"));v4a=np.asarray(Image.open(ROOT/"research_pass42"/"outputs"/"googlewiki_v4_D_low.png").convert("RGB"));gx=chosen(name,"low");gv4=v2.decode((OUT/"selective"/gx["path"]).read_bytes());gb=v2.decode((P43/"outputs"/"googlewiki_v4_v2_low_base_p3012.pbc").read_bytes());panels([("Original",src),("V3.1 Compression",v3),("V4-A D-low",v4a),("V4-B coarse-only",gb),("Pass44 selective",gv4)],(20,110,450,490),"googlewiki_text_crop.png")
 # Smooth Coast sky and sparse correction seams.
 name="coast.jpg";src=np.asarray(ImageOps.exif_transpose(Image.open(ROOT/"hpt_data"/name)).convert("RGB"));v3=np.asarray(PBC3.decompress((ROOT/"research_pass42"/"outputs"/"coast_v31_compression_r3.pbc").read_bytes()).image.convert("RGB"));v4a=np.asarray(Image.open(ROOT/"research_pass42"/"outputs"/"coast_v4_D_low.png").convert("RGB"));cx=chosen(name,"low");cv=v2.decode((OUT/"selective"/cx["path"]).read_bytes());cb=v2.decode((P43/"outputs"/"coast_v4_v2_low_base_p3013.pbc").read_bytes());panels([("Original",src),("V3.1 Compression",v3),("V4-A D-low",v4a),("V4-B coarse-only",cb),("Pass44 selective",cv)],(1740,1080,2124,1464),"coast_crop_comparison.png")
 header,payload=unpack_v2((OUT/"selective"/cx["path"]).read_bytes());coarse_size=int.from_bytes(payload[:4],"little");pos=4+coarse_size;count=((header["height"]+15)//16)*((header["width"]+15)//16);mb=(count*2+7)//8;m=np.asarray(unpack_bits(payload[pos:pos+mb],count,2),dtype=np.uint8).reshape((header["height"]+15)//16,(header["width"]+15)//16);colors=np.zeros((*m.shape,3),dtype=np.uint8);colors[m==0]=(90,90,90);colors[m==1]=(40,180,70);colors[m==2]=(40,100,230);Image.fromarray(colors).resize((header["width"],header["height"]),Image.Resampling.NEAREST).crop((1740,1080,2124,1464)).resize((768,768),Image.Resampling.NEAREST).save(VIS/"coast_mode_map_crop.png")
 print(VIS)
if __name__=="__main__":main()


