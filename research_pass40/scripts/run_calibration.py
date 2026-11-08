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
choices = {
    "googlewiki.png": [(5, {"B": 3, "D": 4}), (20, {"B": 18, "D": 18})],
    "cat.jpg": [(35, {"B": 27, "D": 28}), (50, {"B": 45, "D": 45})],
    "coast.jpg": [(80, {"B": 71, "D": 72}), (90, {"B": 84, "D": 85})],
}

for image, targets in choices.items():
    source = ROOT / "hpt_data" / image
    for target, quality_by_config in targets:
        order = ["B", "D"]
        random.Random(f"p40-cal-{image}-{target}").shuffle(order)
        for config in order:
            q = quality_by_config[config]
            row, _, _ = p39.one(source, image, q, config, 6, 0, "calwarm")
            row["rate_target_qcolor"] = target
            row["source_sha256"] = p39.digest(source)
            result["rows"].append(row)
        for repeat in range(1, 4):
            rotated = order[repeat % 2:] + order[:repeat % 2]
            for config in rotated:
                q = quality_by_config[config]
                row, _, _ = p39.one(source, image, q, config, 6, repeat, "calibration")
                row["rate_target_qcolor"] = target
                row["source_sha256"] = p39.digest(source)
                result["rows"].append(row)
                print(f"{image:14} target q{target:2} {config} q{q:2} r{repeat} {row['bytes']:8}B {row['bpp']:.4f}bpp MSE {row['rgb_mse']:.4f} enc {row['encode_seconds']:.3f}s", flush=True)
        result["rate_calibration"] = {name: [(target, values) for target, values in targets] for name, targets in choices.items()}
        RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
        print(f"completed calibration {image} target q{target}", flush=True)
