import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURRENT = json.loads((ROOT / "research_pass45" / "outputs" / "corpus_results.json").read_text())
P26 = json.loads((ROOT / "research_pass26" / "baseline.json").read_text())
current = {(row["image"], row["preset"]): row for row in CURRENT["rows"]}
checks = []
for image in ("cat.jpg", "flower.jpg", "googlewiki.png", "mario64.png"):
    for preset in ("quality", "high_quality"):
        path = ROOT / "research_pass26" / "artifacts" / "architecture" / f"{image}.{preset}.pbc"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = current[(image, preset)]
        if digest != row["stream_sha256"]:
            raise SystemExit(f"Pass26 stream mismatch: {image}/{preset}")
        checks.append({"image": image, "preset": preset, "bytes": path.stat().st_size, "stream_sha256": digest, "matches_current_release_python": True})

if P26["zip_sha256"] != "edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede":
    raise SystemExit("Pass26 archive identity mismatch")
print(json.dumps({"pass26_zip_sha256": P26["zip_sha256"], "matched_cases": len(checks), "checks": checks}, indent=2))
