import json

from benchmark import OUT, ROOT, one, verify_assets

QCAL = {"flower.jpg": [65, 82], "coast.jpg": [64, 84]}


def main():
    verify_assets()
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text())
    rows = []
    for image, qvalues in QCAL.items():
        src = ROOT / "hpt_data" / image
        for q in qvalues:
            row, _, _ = one(src, image, q, "C", 6, 0, "calwarm2")
            rows.append(row)
            for rep in range(1, 4):
                row, _, _ = one(src, image, q, "C", 6, rep, "calibration")
                row["source_sha256"] = next(a["sha256"] for a in json.loads((ROOT / "research_pass35" / "SOURCE_MANIFEST.json").read_text())["assets"] if a["name"] == image)
                rows.append(row)
                print(f"{image:14} IQ q{q:2} {row['bytes']:8}B {row['bpp']:.4f}bpp MSE {row['rgb_mse']:.4f} SSIM2 {row['ssimulacra2']:.3f} enc {row['encode_seconds']:.3f}s", flush=True)
    results["rows"].extend(rows)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    (OUT / "iq_rate_calibration2.json").write_text(json.dumps({"selection": "rate-only second values for targets whose first calibration missed by more than 5%", "qvalues": QCAL, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
