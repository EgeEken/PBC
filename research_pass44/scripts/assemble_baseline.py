import hashlib,json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=pathlib.Path(__file__).resolve().parents[1]/"outputs"
def read(rel): return json.loads((ROOT/rel).read_text())
def main():
 manifest=read("research_pass35/SOURCE_MANIFEST.json"); raw=read("research_pass42/outputs/benchmark_raw.json"); v3time=read("research_pass44/outputs/v31_single_thread_timing.json"); v2=read("research_pass43/outputs/v2_rd_comparison.json"); coarse=read("research_pass43/outputs/coarse_budget_probe.json")
 avgrid=read("research_pass44/outputs/avif_grid_cli.json"); avtar=read("research_pass44/outputs/avif_targeted.json"); avextra=read("research_pass44/outputs/avif_extra_near_rate.json")
 source_rows=[]
 for name in ["googlewiki.png","cat.jpg","coast.jpg"]:
  src=ROOT/"hpt_data"/name; sha=hashlib.sha256(src.read_bytes()).hexdigest(); expected=next(x for x in manifest["assets"] if x["name"]==name)
  if sha!=expected["sha256"]: raise RuntimeError(f"source hash mismatch: {name}")
  source_rows.append({"name":name,"sha256":sha,"bytes":src.stat().st_size,"width":expected["width"],"height":expected["height"]})
 timings={(x["image"],x["preset"]):x for x in v3time["rows"]}; v31=[]
 for name in ["googlewiki.png","cat.jpg","coast.jpg"]:
  for preset in ["compression","balanced","quality","high_quality"]:
   m=raw["v31"][f"{name}|{preset}"]; x=m["rows"][0]; t=timings[(name,preset)]
   v31.append({"image":name,"preset":preset,"bytes":x["bytes"],"bpp":x["bytes"]*8/(source_rows[[s['name'] for s in source_rows].index(name)]["width"]*source_rows[[s['name'] for s in source_rows].index(name)]["height"]),"rgb_mse":x["rgb_mse"],"rgb_psnr":x["rgb_psnr"],"file_sha256":x["file_sha256"],"pixel_sha256":x["pixel_sha256"],"encode_samples_seconds":t["encode_samples_seconds"],"decode_samples_seconds":t["decode_samples_seconds"],"median_encode_seconds":t["median_encode_seconds"],"median_decode_seconds":t["median_decode_seconds"],"frozen_source_commit":m["frozen_source_commit"],"model_sha256":m["model_sha256"]})
 v4a=[]
 for name in ["googlewiki.png","cat.jpg","coast.jpg"]:
  for point in ["low","intermediate"]:
   x=raw["v4"][f"{name}|D|{point}"]; v4a.append({"image":name,"point":point,"bytes":x["bytes"],"bpp":x["bpp"],"rgb_mse":x["rgb_mse"],"rgb_psnr":x["rgb_psnr"],"encode_seconds_pass42":x["encode_seconds"],"fresh_decode_seconds_pass42":x["fresh_decode_seconds"],"file_sha256":x["file_sha256"],"mode_counts":x["mode_counts"],"byte_accounting":x["byte_accounting"],"stream_path":pathlib.Path(x["stream_path"]).name})
 v4b=[]; v4b_base=[]
 for name in ["googlewiki.png","cat.jpg","coast.jpg"]:
  for point in ["low","intermediate"]:
   xs=v2[name]["points"][point]["candidates"]; x=min(xs,key=lambda z:z["bytes"]); v4b.append({"image":name,"point":point,"bytes":x["bytes"],"bpp":x["bpp"],"rgb_mse":x["mse"],"rgb_psnr":x["psnr"],"encode_seconds_pass43":x.get("encode_seconds"),"file_sha256":x["file_sha256"],"mode_counts":x["mode_counts"],"byte_accounting":x["byte_accounting"],"coarse":x["coarse"],"stream_path":pathlib.Path(x["stream_path"]).name})
  target=raw["v31"][f"{name}|compression"]["rows"][0]["bytes"]; eligible=[x for x in coarse[name] if x["bytes"]<=target*1.05]; x=min(eligible,key=lambda z:z["mse"]); path=ROOT/"research_pass43"/"outputs"/f"{pathlib.Path(name).stem}_v4_v2_low_base_p{3000+x['coarse_patch_count']:04d}.pbc"
  v4b_base.append({"image":name,"reference_preset":"compression","budget_bytes":target,"hard_cap_bytes":int(target*1.05),"bytes":x["bytes"],"rate_gap_percent":100*(x["bytes"]/target-1),"rgb_mse":x["mse"],"pixel_sha256":x["pixel_sha256"],"file_sha256":x["file_sha256"],"coarse_patch_count":x["coarse_patch_count"],"skip_only":True,"path":path.name})
 avmap={}
 for group in [avgrid["rows"],avtar["rows"],avextra["rows"]]:
  for x in group:
   key=(x["image"],x["qcolor"]); avmap[key]=dict(avmap.get(key,{}),**x)
 avif=[]
 for (name,q),x in sorted(avmap.items()):
  if not x.get("source_sha256"):
   x["source_sha256"]=next(s["sha256"] for s in source_rows if s["name"]==name)
  x["timing_boundary"]="CLI process launch and file/PNG I/O included; not cross-compared"
  avif.append(x)
 matches=[]
 for ref in v31:
  points=[x for x in avif if x["image"]==ref["image"]]; nearest=min(points,key=lambda x:abs(x["bytes"]/ref["bytes"]-1)); matches.append({"image":ref["image"],"preset":ref["preset"],"target_bytes":ref["bytes"],"avif_qcolor":nearest["qcolor"],"avif_bytes":nearest["bytes"],"avif_mse":nearest["rgb_mse"],"gap_percent":100*(nearest["bytes"]/ref["bytes"]-1),"within_5_percent":abs(nearest["bytes"]/ref["bytes"]-1)<=.05})
 result={"pass":44,"phase_a_frozen":True,"repository":{"starting_head":"3723dc14de88c6c3dcbd98dd80a097e019a92a9f","branch":"research/experiments"},"sources":source_rows,"codecs":{"v31":{"source_commit":"8006f30debfc4a57289df06df21636aeb52baffd","model_sha256":"ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f","timing":"one thread; one warmup; three in-process repetitions; exact stream hash checked against Pass42"},"avif":{"libavif":"1.3.0","libaom":"3.12.1","speed":8,"threads":1,"depth":8,"yuv":"420","range":"full","cicp":"1/13/1","timing":"CLI launch and I/O included; incomparable"}},"v31_points":v31,"avif_points":avif,"avif_nearest_v31_targets":matches,"v4a_points":v4a,"v4b_native_points":v4b,"v4b_coarse_only_within_low_budget":v4b_base,"diagnosis":{"primary":"the earlier truncated coarse-prefix comparison understated the value of V3-style coarse procedural refinement; the complete V3.1 Compression prefixes reproduce its base pixels at near-matched rates, while current native detail adds only small gains","mode_map_raw_bytes_per_image":651,"header_bytes":32,"coarse_stage":"complete-prefix V4 all-SKIP bases match the V3.1 Compression reconstruction on all three images; sparse detail reduces MSE by only 0.18% to 1.01% within the 105% rate cap"},"caveats":["No matched AVIF encode/decode speed ranking: in-process C API probe failed before parity.","V4 timings retain Pass42/43 measurement boundaries; cross-codec timing claims withheld.","Only three development sources; no generalization claim."]}
 (OUT/"competitive_baseline.json").write_text(json.dumps(result,indent=2)+"\n")
 print("wrote",len(v31),"V3.1",len(avif),"unique AVIF",sum(x["within_5_percent"] for x in matches),"of",len(matches),"near rates")
if __name__=="__main__":main()
