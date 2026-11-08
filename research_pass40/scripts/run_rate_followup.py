import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / "research_pass40"
OUT = PASS / "outputs" / "ablation"
RESULT_PATH = OUT / "results.json"
sys.path.insert(0, str(ROOT / "research_pass39" / "scripts"))
import benchmark as p39

result = json.loads(RESULT_PATH.read_text())
p39.OUT = OUT
p39.CONFIGS = result["configurations"]
points = {
    "googlewiki.png": [(18, "C", 18)],
    "cat.jpg": [(27, "A", 33), (45, "A", 51), (45, "C", 51)],
    "coast.jpg": [(71, "A", 82), (84, "A", 91), (72, "C", 82), (85, "C", 92)],
}

for image, trials in points.items():
    source = ROOT / "hpt_data" / image
    for target_q, config, q in trials:
        row, _, _ = p39.one(source, image, q, config, 6, 0, "calwarm2")
        row["rate_target_config"] = "D" if config == "C" else "B" if config == "A" else None
        row["rate_target_qcolor"] = target_q
        row["source_sha256"] = p39.digest(source)
        result["rows"].append(row)
        order = [config]
        for repeat in range(1, 4):
            for current in order:
                row, _, _ = p39.one(source, image, q, current, 6, repeat, "followup")
                row["rate_target_config"] = "D" if current == "C" else "B" if current == "A" else None
                row["rate_target_qcolor"] = target_q
                row["source_sha256"] = p39.digest(source)
                result["rows"].append(row)
                print(f"{image:14} target {row['rate_target_config']} q{target_q:2} {current} q{q:2} r{repeat} {row['bytes']:8}B {row['bpp']:.4f}bpp MSE {row['rgb_mse']:.4f}", flush=True)
        RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

result["rate_followup"] = {
    "googlewiki.png": [(4, "C", 6), (18, "C", 18)],
    "cat.jpg": [(27, "A", 33), (45, "A", 51), (45, "C", 51)],
    "coast.jpg": [(71, "A", 82), (84, "A", 91), (72, "C", 82), (85, "C", 92)],
}
RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
