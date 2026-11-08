import hashlib
import json
import pathlib
import subprocess
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research_pass36" / "outputs" / "symbol_parity"
BIN = ROOT / "research_pass35" / "outputs" / "binaries"
PROFILE = ROOT / "research_pass36" / "outputs" / "profiling" / "avifenc-symbol.exe"
DEC = BIN / "reference" / "avifdec.exe"
CASES = [("googlewiki.png", 5, 6), ("cat.jpg", 35, 6), ("googlewiki.png", 5, 8)]

OUT.mkdir(parents=True, exist_ok=True)
rows = []
for image, q, speed in CASES:
    src = ROOT / "hpt_data" / image
    outputs = []
    for name, exe in (("reference", BIN / "reference" / "avifenc.exe"), ("symbol", PROFILE)):
        avif = OUT / f"{pathlib.Path(image).stem}_q{q}_s{speed}_{name}.avif"
        cmd = [str(exe), "--codec", "aom", "--qcolor", str(q), "--qalpha", "0",
               "--speed", str(speed), "--jobs", "1", "--depth", "8", "--yuv", "420",
               "--range", "full", "--cicp", "1/13/1", "--ignore-exif", "--ignore-xmp",
               "--ignore-profile", str(src), str(avif)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"{name} encode failed: {run.stderr}")
        dec = OUT / f"{avif.stem}.png"
        run = subprocess.run([str(DEC), "--jobs", "1", "--ignore-icc", str(avif), str(dec)],
                             capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"{name} decode failed: {run.stderr}")
        with Image.open(dec) as im:
            pixels = im.convert("RGB")
            decoded = (pixels.size, pixels.tobytes())
        outputs.append({"name": name, "avif": avif, "decoded": decoded})
    a, b = outputs
    same_bytes = hashlib.sha256(a["avif"].read_bytes()).digest() == hashlib.sha256(b["avif"].read_bytes()).digest()
    same_rgb = a["decoded"] == b["decoded"]
    rows.append({"image": image, "qcolor": q, "speed": speed,
                 "reference_sha256": hashlib.sha256(a["avif"].read_bytes()).hexdigest(),
                 "symbol_sha256": hashlib.sha256(b["avif"].read_bytes()).hexdigest(),
                 "reference_bytes": a["avif"].stat().st_size,
                 "symbol_bytes": b["avif"].stat().st_size,
                 "byte_identical": same_bytes, "decoded_rgb_identical": same_rgb,
                 "dimensions": list(a["decoded"][0])})

result = {"cases": rows, "passed": all(r["byte_identical"] and r["decoded_rgb_identical"] for r in rows)}
(OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit("symbol build output parity failed")
