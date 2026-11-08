import csv
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / "research_pass40"
OUT = PASS / "outputs" / "qmaps"
sys.path.insert(0, str(ROOT / "research_pass39" / "scripts"))
import benchmark as p39

DIAG = ROOT / "research_pass39" / "outputs" / "binaries" / "avifenc-qmap.exe"
LEDGER = json.loads((PASS / "outputs" / "ablation" / "results.json").read_text())
CONFIGS = LEDGER["configurations"]
POINTS = {}
for row in LEDGER["rows"]:
    if row["phase"] in ("timed", "calibration", "followup"):
        POINTS.setdefault(row["image"], {}).setdefault(row["qcolor"], set()).add(row["config"])
OUT.mkdir(parents=True, exist_ok=True)
rows = []

for image, quality_configs in POINTS.items():
    source = ROOT / "hpt_data" / image
    for q, available_configs in quality_configs.items():
        for config in sorted(available_configs):
            options = CONFIGS[config]
            key = f"{pathlib.Path(image).stem}_q{q}_{config}"
            avif = OUT / f"{key}.avif"
            command = [str(DIAG), "--codec", "aom", "--qcolor", str(q), "--qalpha", "0", "--speed", "6",
                       "--jobs", "1", "--depth", "8", "--yuv", "420", "--range", "full", "--cicp", "1/13/1",
                       "--ignore-exif", "--ignore-xmp", "--ignore-profile"] + options + [str(source), str(avif)]
            proc = subprocess.run(command, capture_output=True, text=True, env=p39.ENV)
            if proc.returncode:
                raise RuntimeError(f"diagnostic encode failed ({key}):\n{proc.stderr}\n{proc.stdout}")
            expected = next(r for r in LEDGER["rows"]
                            if r["image"] == image and r["qcolor"] == q and r["config"] == config and r["phase"] in ("timed", "calibration", "followup"))
            digest = hashlib.sha256(avif.read_bytes()).hexdigest()
            if digest != expected["avif_sha256"]:
                raise RuntimeError(f"qmap build changed file bytes for {key}: {digest} != {expected['avif_sha256']}")
            samples = []
            markers = []
            for line in proc.stderr.splitlines():
                if line.startswith(("PBC_EXTRA_CFG,", "PBC_RECODE_SWITCH,", "PBC_NO_RECODE_AQ,")):
                    markers.append(line)
                if line.startswith("PBC_QMAP,"):
                    _, row, col, qindex, segment = line.split(",")
                    row, col = int(row), int(col)
                    if row % 16 == 0 and col % 16 == 0:
                        samples.append((row, col, int(qindex), int(segment)))
            with (OUT / f"{key}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(("mi_row", "mi_col", "effective_qindex", "segment_id"))
                writer.writerows(samples)
            rows.append({"image": image, "qcolor": q, "config": config, "avif_sha256": digest,
                         "matched_timed_output": True, "root_samples": len(samples), "path_markers": markers,
                         "qindex_min": min(s[2] for s in samples), "qindex_max": max(s[2] for s in samples),
                         "segments": sorted(set(s[3] for s in samples)), "qmap_csv": f"outputs/qmaps/{key}.csv"})
            print(f"{key}: parity=True roots={len(samples)} q={min(s[2] for s in samples)}..{max(s[2] for s in samples)} markers={markers}", flush=True)

(PASS / "outputs" / "qmaps" / "manifest.json").write_text(json.dumps({"diagnostic_avifenc_sha256": hashlib.sha256(DIAG.read_bytes()).hexdigest(), "rows": rows}, indent=2) + "\n")
