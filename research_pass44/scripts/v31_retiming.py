import hashlib,json,os,pathlib,statistics,sys,time
import threadpoolctl
from PIL import Image,ImageOps
ROOT=pathlib.Path(__file__).resolve().parents[2]
FROZEN=ROOT/"research_pass42"/"outputs"/"v31_frozen"
os.environ["NUMBA_NUM_THREADS"]="1"
sys.path.insert(0,str(FROZEN))
os.chdir(FROZEN)
from PBC3 import PBC3
from pbc3_types import PBC3Config
IMAGES=["googlewiki.png","cat.jpg","coast.jpg"]
PRESETS=["compression","balanced","quality","high_quality"]
OUT=pathlib.Path(__file__).resolve().parents[1]/"outputs"
def main():
 rows=[]
 with threadpoolctl.threadpool_limits(limits=1):
  for name in IMAGES:
   src=ROOT/"hpt_data"/name
   with Image.open(src) as im: rgb=ImageOps.exif_transpose(im).convert("RGB")
   for preset in PRESETS:
    cfg=getattr(PBC3Config,preset)()
    expected=json.loads((ROOT/"research_pass42"/"outputs"/"benchmark_raw.json").read_text())["v31"][f"{name}|{preset}"]["rows"][0]["file_sha256"]
    warm=PBC3.compress(rgb,config=cfg,rng_mode="release").data
    if hashlib.sha256(warm).hexdigest()!=expected: raise RuntimeError(f"frozen output mismatch {name} {preset}")
    PBC3.decompress(warm)
    enc=[]; dec=[]
    for rep in range(3):
     t=time.perf_counter(); data=PBC3.compress(rgb,config=cfg,rng_mode="release").data; enc.append(time.perf_counter()-t)
     if hashlib.sha256(data).hexdigest()!=expected: raise RuntimeError("repeat stream mismatch")
     t=time.perf_counter(); rec=PBC3.decompress(data); dec.append(time.perf_counter()-t)
    rows.append({"image":name,"preset":preset,"bytes":len(warm),"file_sha256":expected,"encode_samples_seconds":enc,"decode_samples_seconds":dec,"median_encode_seconds":statistics.median(enc),"median_decode_seconds":statistics.median(dec),"reconstruction_hash_matches_pass42":True})
    print(name,preset,len(warm),f"enc={statistics.median(enc):.4f}",f"dec={statistics.median(dec):.4f}",flush=True)
 result={"source_commit":"8006f30debfc4a57289df06df21636aeb52baffd","model_sha256":"ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f","threads":1,"warmups_per_point":1,"repetitions":3,"timing_boundary":"in-process PBC3.compress and PBC3.decompress, excludes process launch and file I/O","rows":rows}
 (OUT/"v31_single_thread_timing.json").write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":main()


