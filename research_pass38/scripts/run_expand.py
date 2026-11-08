import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
import psutil
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "hpt_data"
CASES = [("googlewiki.png", 5), ("mario64.png", 50), ("cat.jpg", 35)]

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def encode(exe, src, q, out, log):
    args=[str(exe),"--codec","aom","--qcolor",str(q),"--qalpha","0","--speed","6",
          "--jobs","1","--depth","8","--yuv","420","--range","full","--cicp","1/13/1",
          "--ignore-exif","--ignore-xmp","--ignore-profile",str(src),str(out)]
    env=os.environ.copy()
    env["PATH"]=str(exe.parent)+os.pathsep+r"C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows"
    with open(str(log)+".stdout","wb") as so, open(str(log)+".stderr","wb") as se:
        start=time.perf_counter()
        proc=subprocess.Popen(args,stdout=so,stderr=se,env=env)
        peak=0
        process=psutil.Process(proc.pid)
        while proc.poll() is None:
            try: peak=max(peak,process.memory_info().rss)
            except psutil.Error: pass
            time.sleep(.005)
        elapsed=time.perf_counter()-start
        try: peak=max(peak,process.memory_info().rss)
        except psutil.Error: pass
    if proc.returncode or not out.is_file(): raise RuntimeError(f"encode failed: {exe}")
    return {"seconds_one_run":elapsed,"peak_sampled_rss_bytes":peak,
            "bytes":out.stat().st_size,"sha256":sha(out)}

def rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"),dtype=np.float64)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--expanded",type=Path,required=True)
    ap.add_argument("--decoder",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,q in CASES:
        src=DATA/name; case=a.out/Path(name).stem
        case.mkdir(exist_ok=True)
        results={}
        for label,exe in (("REF",a.reference),("EXP",a.expanded)):
            avif=case/f"{label.lower()}.avif"
            results[label]=encode(exe,src,q,avif,case/label.lower())
            decoded=case/f"{label.lower()}.png"
            env=os.environ.copy(); env["PATH"]=str(a.decoder.parent)+os.pathsep+r"C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows"
            with open(case/f"{label.lower()}.decode.stdout","wb") as so, open(case/f"{label.lower()}.decode.stderr","wb") as se:
                rc=subprocess.run([str(a.decoder),"--jobs","1","--ignore-icc",str(avif),str(decoded)],stdout=so,stderr=se,env=env).returncode
            if rc or not decoded.is_file(): raise RuntimeError(f"decode failed: {label} {name}")
            rec=rgb(decoded); orig=rgb(src)
            if rec.shape!=orig.shape: raise RuntimeError("decoded size mismatch")
            results[label]["rgb_mse"]=float(np.mean((orig-rec)**2))
        pixels=rgb(src).shape[0]*rgb(src).shape[1]
        for label in ("REF","EXP"):
            results[label]["bpp"]=results[label]["bytes"]*8/pixels
        results["delta"]={
            "bytes_percent":100*(results["EXP"]["bytes"]/results["REF"]["bytes"]-1),
            "bpp_percent":100*(results["EXP"]["bpp"]/results["REF"]["bpp"]-1),
            "rgb_mse_percent":100*(results["EXP"]["rgb_mse"]/results["REF"]["rgb_mse"]-1),
            "encode_time_percent":100*(results["EXP"]["seconds_one_run"]/results["REF"]["seconds_one_run"]-1),
        }
        results.update({"image":name,"qcolor":q,"speed":6,"input_sha256":sha(src),
                        "decoder":"pinned avifdec v1.3.0, unmodified"})
        rows.append(results)
    doc={"source":{"libaom":"10aece4157eb79315da205f39e19bf6ab3ee30d0",
                   "libavif":"1aadfad932c98c069a1204261b1856f81f3bc199"},
         "timings":"single encode per config, RSS polled every 5 ms; exploratory, not a speed claim",
         "cases":rows}
    (a.out/"expanded_comparison.json").write_text(json.dumps(doc,indent=2),encoding="utf-8")
    print(json.dumps(rows,indent=2))

if __name__=="__main__": main()
