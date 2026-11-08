import hashlib
import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / "research_pass39"
RESULTS = json.loads((PASS / "outputs" / "benchmark" / "results.json").read_text())
AGG = json.loads((PASS / "outputs" / "benchmark" / "summary.json").read_text())
SPATIAL = json.loads((PASS / "outputs" / "benchmark" / "spatial_analysis.json").read_text())
QMAP_DIR = PASS / "outputs" / "qmap"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


qmap_parity = []
for config in ("REF", "A", "B", "C", "D"):
    diagnostic = QMAP_DIR / f"cat_q35_{config}.avif"
    reference = PASS / "outputs" / "benchmark" / f"s6_cat_q35_{config}_timed3.avif"
    qmap_parity.append({"config": config, "bytes_identical": diagnostic.read_bytes() == reference.read_bytes(),
                        "diagnostic_sha256": sha(diagnostic), "reference_sha256": sha(reference)})

memory = [row for row in RESULTS["rows"] if row["phase"] == "memory"]
phases = Counter(f"{row['phase']}_speed{row['speed']}" for row in RESULTS["rows"])
summary = {
    "pass": 39,
    "objective": "Measure existing AV1 AQ and tune=iq behavior through the pinned libavif AV1 still-image path; no production changes.",
    "repository": {"branch": "research/experiments", "starting_head": "3723dc14de88c6c3dcbd98dd80a097e019a92a9f"},
    "sources": AGG["build"],
    "settings": RESULTS["settings"],
    "configurations": RESULTS["configurations"],
    "ledger": {"rows": len(RESULTS["rows"]), "phase_counts": dict(phases), "diagnostic_qmap_encodes": 5},
    "metrics": {"rate": "8 * completed AVIF bytes / source pixels", "rgb_mse": "RGB source and decoded arrays promoted to float64 before subtraction and squaring", "perceptual": AGG["build"]["ssimulacra2_input"], "timings": "fresh-process encode and decode; one warmup plus three timed repeats at main points"},
    "speed6_points": AGG["speed6_points"],
    "speed8_points": AGG["speed8_points"],
    "matched_rate_pairs": AGG["matched_rate_pairs"],
    "spatial_analysis": SPATIAL,
    "memory_samples": memory,
    "qmap_byte_parity": qmap_parity,
    "gates": {"A_source_and_option_path": "passed; variance AQ accepted but inactive in the selected no-recode path", "B_completed_RD_timing_decode": "passed", "C_effective_qindex_and_regions": "passed; separate diagnostic build preserves bytes", "D_meaningful_near_rate_improvement": "failed; IQ results are mixed and regress RGB MSE in 7/8 matched pairs", "E_holdout": "not run; no candidate selected"},
    "decision": "No production candidate. Keep Pass 39 as research only.",
    "artifacts": {"raw_rows": "outputs/benchmark/results.json", "aggregates": "outputs/benchmark/summary.json", "region_details": "outputs/benchmark/spatial_analysis.json", "rate_distortion_figure": "visuals/rate_distortion.png", "spatial_figure": "visuals/spatial_allocation.png"},
}
(PASS / "PASS39_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"wrote PASS39_SUMMARY.json with {len(summary['matched_rate_pairs'])} matched-rate records and {len(qmap_parity)} qmap parity checks")
