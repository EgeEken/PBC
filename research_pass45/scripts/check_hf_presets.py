import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://egeeken-pbc.hf.space/api/presets"
expected = json.loads((ROOT / "research_pass10" / "baseline_manifest.json").read_text())["presets"]
with urllib.request.urlopen(URL, timeout=20) as response:
    actual = json.load(response)

if set(actual) != set(expected):
    raise SystemExit(f"Preset names differ: API={sorted(actual)} expected={sorted(expected)}")
for preset, reference in expected.items():
    resolved = dict(actual[preset])
    model = resolved.get("learned_filler_model_path")
    if model:
        resolved["learned_filler_model_path"] = "patch_policy.npz"
    if resolved != reference:
        keys = sorted(set(resolved) | set(reference))
        differences = {key: {"api": resolved.get(key), "expected": reference.get(key)} for key in keys if resolved.get(key) != reference.get(key)}
        raise SystemExit(f"{preset} mismatch: {json.dumps(differences, sort_keys=True)}")
print("HF preset definitions match the pinned reference")
