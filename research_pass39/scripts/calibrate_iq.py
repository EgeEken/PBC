import json

from benchmark import OUT, ROOT, one, verify_assets

QCAL = {
    "googlewiki.png": [12, 30],
    "mario64.png": [48, 62],
    "flower.jpg": [72, 81],
    "cat.jpg": [],
    "coast.jpg": [67, 87],
}


def main():
    verify_assets()
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text())
    rows = []
    for image, qvalues in QCAL.items():
        src = ROOT / "hpt_data" / image
        for q in qvalues:
            row, _, _ = one(src, image, q, "C", 6, 0, "calwarm")
            rows.append(row)
            for rep in range(1, 4):
                row, _, _ = one(src, image, q, "C", 6, rep, "calibration")
                row["source_sha256"] = next(a["sha256"] for a in json.loads((ROOT / "research_pass35" / "SOURCE_MANIFEST.json").read_text())["assets"] if a["name"] == image)
                rows.append(row)
                print(f"{image:14} IQ q{q:2} {row['bytes']:8}B {row['bpp']:.4f}bpp MSE {row['rgb_mse']:.4f} SSIM2 {row['ssimulacra2']:.3f} enc {row['encode_seconds']:.3f}s", flush=True)
    results["rows"].extend(rows)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    (OUT / "iq_rate_calibration.json").write_text(json.dumps({"selection": "rate-only; qcolor fitted by linear interpolation in log(bpp) between the original C endpoints", "qvalues": QCAL, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
