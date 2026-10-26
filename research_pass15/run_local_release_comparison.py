import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
WORKER = HERE / "benchmark_worker.py"
IMAGE_NAMES = (
    "cat.jpg", "eiffel.png", "flower.jpg", "googlewiki.png",
    "mario64.png", "monet.jpg", "ratedistortion.png", "seagull.jpg",
)
PRESETS = ("compression", "balanced", "quality", "high_quality")
THREADS = {key: "1" for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)}


def mean(values):
    return statistics.mean(values)


def cv(values):
    average = mean(values)
    return 100 * statistics.pstdev(values) / average if average else 0.0


def send(process, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


def start_worker(name, source, python, image_paths, cpu):
    env = os.environ.copy()
    env.update(THREADS)
    env.update({
        "PBC_CPU": str(cpu),
        "PBC_IMAGES": json.dumps([str(path) for path in image_paths]),
        "PBC_SOURCE": str(source.resolve()),
        "PYTHONPATH": str(source.resolve()),
        "PYTHONUNBUFFERED": "1",
    })
    process = subprocess.Popen(
        [str(python), str(WORKER)], cwd=source, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    process._pbc_name = name
    process._pbc_source = str(source.resolve())
    return process


def stop_worker(process):
    if process.poll() is None:
        process.stdin.write('{"op":"stop"}\n')
        process.stdin.flush()
        process.stdin.close()
        process.wait(timeout=30)
    if process.returncode:
        raise RuntimeError(f"{process._pbc_name} worker exited {process.returncode}: {process.stderr.read()[-3000:]}")


def case_stats(rows, variant, field):
    result = {}
    for row in rows:
        if row["variant"] == variant:
            result.setdefault((row["image"], row["preset"]), []).append(row[field])
    return result


def comparison(rows, variant_a, variant_b, field):
    a = case_stats(rows, variant_a, field)
    b = case_stats(rows, variant_b, field)
    keys = sorted(a)
    return {
        "a_mean_case_median": mean([statistics.median(a[k]) for k in keys]),
        "b_mean_case_median": mean([statistics.median(b[k]) for k in keys]),
        "a_cv_mean_percent": mean([cv(a[k]) for k in keys]),
        "b_cv_mean_percent": mean([cv(b[k]) for k in keys]),
        "case_reductions_percent": {f"{i}/{p}": 100 * (1 - statistics.median(b[(i, p)]) / statistics.median(a[(i, p)])) for i, p in keys},
    }


def write_report(output, args, metadata):
    rows = output["rows"]
    variants = output["variants"]
    old, new = variants["PBC 3.0"], variants["PBC 3.1 development"]
    lines = [
        "# PBC 3.0 vs PBC 3.1 development — local release comparison",
        "",
        "## Method",
        "",
        f"PBC 3.0 is the immutable release at `{output['pbc30_commit']}`. PBC 3.1 development candidate `{output['development_candidate_id']}` derives from frozen source `{output['pbc31_baseline_commit']}` (`pbc31_release_rng_baseline`) and changes only `pbc3_ops.integral()` to fused Numba accumulation. Workers load one source tree each in separate persistent Python 3.11.16 processes. They each warm all four presets before measurement, preload the same eight EXIF-transposed RGB images, then run three interleaved repetitions per image and preset. Timers wrap the complete `compress` and `decompress` calls only. Hashing, patch parsing, external RGB MSE, IPC, and image loading are outside timers.",
        "The design records 192 variant observations (each observation times one encode plus one decode), forming 96 matched old/new case-repetition pairs.",
        "",
        f"Machine: {metadata['cpu_model']} ({metadata['core_summary']}); both workers are pinned to logical CPU {args.p_core}, mapped by Windows processor topology to a P-core. All numerical thread limits are 1. PBC 3.0 and PBC 3.1 use the same Python/dependency runtime, image data, model hash, default seed 2003, release RNG, and internal final-MSE setting.",
        "",
        f"Runtime: Python {output['python']}; NumPy {output['numpy']}; Numba {output['numba']}; Pillow {output['pillow']}. Source-isolated process CWDs: `{old['source']}` and `{new['source']}`. Windows reports a 2.5 GHz current/max package clock; per-core boost telemetry is unavailable through psutil on this host, so frequency was not normalized or inferred from that report.",
        "",
        "Input file SHA-256 values:",
        "",
        "| Image | Input SHA-256 |",
        "|---|---|",
    ]
    for image in IMAGE_NAMES:
        lines.append(f"| {image} | `{output['input_sha256'][image]}` |")
    lines.extend([
        "",
        "## Correctness gate",
        "",
        f"All {output['matched_stream_pairs']}/{output['measured_stream_pairs']} paired measured streams were byte-identical. All {output['matched_pixel_pairs']}/{output['measured_stream_pairs']} decoded RGB outputs, patch counts, final bpp, and independently scored original-resolution RGB MSE values matched exactly. Repeated outputs were stable within each source. Internal MSE remained enabled in both implementations. The frozen model SHA-256 was `{output['model_sha256']}`.",
        "",
        "## Preset timing results",
        "",
        "Values are averages of the 8 per-image medians (three repetitions per case). Reduction is `1 - PBC 3.1 / PBC 3.0`.",
        "",
        "| Preset | PBC 3.0 encode (s) | PBC 3.1 encode (s) | Encode reduction | PBC 3.0 decode (s) | PBC 3.1 decode (s) | Decode reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for preset in (*PRESETS, "overall"):
        aenc, benc = output["summary"][preset]["encode"]["a_mean_case_median"], output["summary"][preset]["encode"]["b_mean_case_median"]
        adec, bdec = output["summary"][preset]["decode"]["a_mean_case_median"], output["summary"][preset]["decode"]["b_mean_case_median"]
        label = preset.replace("_", " ").title() if preset != "overall" else "Overall"
        lines.append(f"| {label} | {aenc:.4f} | {benc:.4f} | {100*(1-benc/aenc):.2f}% | {adec:.4f} | {bdec:.4f} | {100*(1-bdec/adec):.2f}% |")
    lines.extend([
        "",
        "## Per-image medians",
        "",
        "Each cell averages the four per-preset medians for that image.",
        "",
        "| Image | PBC 3.0 encode (s) | PBC 3.1 encode (s) | Encode reduction | PBC 3.0 decode (s) | PBC 3.1 decode (s) | Decode reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for image in IMAGE_NAMES:
        old_enc = mean([statistics.median(case_stats(rows, "PBC 3.0", "encode_seconds")[(image, p)]) for p in PRESETS])
        new_enc = mean([statistics.median(case_stats(rows, "PBC 3.1 development", "encode_seconds")[(image, p)]) for p in PRESETS])
        old_dec = mean([statistics.median(case_stats(rows, "PBC 3.0", "decode_seconds")[(image, p)]) for p in PRESETS])
        new_dec = mean([statistics.median(case_stats(rows, "PBC 3.1 development", "decode_seconds")[(image, p)]) for p in PRESETS])
        lines.append(f"| {Path(image).stem} | {old_enc:.4f} | {new_enc:.4f} | {100*(1-new_enc/old_enc):.2f}% | {old_dec:.4f} | {new_dec:.4f} | {100*(1-new_dec/old_dec):.2f}% |")
    lines.extend([
        "",
        "## Variability and outliers",
        "",
        f"Mean within-case CV across three repetitions: encode {output['summary']['overall']['encode']['a_cv_mean_percent']:.2f}% for PBC 3.0 and {output['summary']['overall']['encode']['b_cv_mean_percent']:.2f}% for PBC 3.1; decode {output['summary']['overall']['decode']['a_cv_mean_percent']:.2f}% and {output['summary']['overall']['decode']['b_cv_mean_percent']:.2f}%. Whole-call reductions across the 32 cases were {output['summary']['overall']['encode']['regressions']} encode regressions and {output['summary']['overall']['decode']['regressions']} decode regressions (negative values).",
        "",
        "Slowest PBC 3.1 encoder case by median: " + output["slowest"]["encode"] + ". Slowest PBC 3.1 decoder case by median: " + output["slowest"]["decode"] + ". Lowest encode reduction: " + output["worst_regression"]["encode"] + ". Lowest decode reduction: " + output["worst_regression"]["decode"] + ".",
        "",
        "The complete per-repetition timing rows and SHA/MSE/bpp/patch checks are preserved locally in `local_comparison_integral_fused.json` (ignored by Git).",
        "",
        "## Relation to HF results",
        "",
        "This local test uses a different machine, dataset and timing protocol from the controlled HF cpu-basic comparison. The HF result was 28.83% lower encoding time and 64.38% lower decoding time with byte-identical files on its matched cases. The two percentages are reported separately and should not be used as direct cross-machine comparisons.",
        "",
    ])
    (HERE / "PBC31_LOCAL_RELEASE_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--p-core", type=int, default=12)
    parser.add_argument("--cpu-model", default="12th Gen Intel Core i9-12900H")
    parser.add_argument("--reported-clock-mhz", type=int, default=2500)
    parser.add_argument("--output-json", type=Path, default=HERE / "local_comparison.json")
    args = parser.parse_args()

    image_paths = [args.data_dir / name for name in IMAGE_NAMES]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    old_source = ROOT / "research_pass1" / "release"
    new_source = ROOT
    baseline = json.loads((ROOT / "research_pass10" / "baseline_manifest.json").read_text(encoding="utf-8"))
    if baseline["baseline_id"] != "pbc31_release_rng_baseline":
        raise AssertionError("unexpected PBC 3.1 baseline")
    import hashlib
    for name, expected in baseline["source_sha256"].items():
        if name == "pbc3_ops.py":
            continue
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"PBC 3.1 source mismatch: {name}")
    if hashlib.sha256((ROOT / "patch_policy.npz").read_bytes()).hexdigest() != baseline["model_sha256"]:
        raise AssertionError("PBC 3.1 model hash mismatch")
    workers = {
        "PBC 3.0": start_worker("PBC 3.0", old_source, args.python, image_paths, args.p_core),
        "PBC 3.1 development": start_worker("PBC 3.1 development", new_source, args.python, image_paths, args.p_core),
    }
    rows = []
    try:
        for variant, process in workers.items():
            ready = send(process, {"op": "warm"})
            if not ready.get("ready"):
                raise RuntimeError(f"{variant} warmup failed: {ready}")
        for rep in range(3):
            for image_index, image in enumerate(IMAGE_NAMES):
                for preset in PRESETS:
                    variants = ("PBC 3.0", "PBC 3.1 development") if rep % 2 == 0 else ("PBC 3.1 development", "PBC 3.0")
                    for variant in variants:
                        row = send(workers[variant], {"op": "measure", "image_index": image_index, "preset": preset})
                        row.update({"variant": variant, "rep": rep})
                        rows.append(row)
                print(f"completed repetition {rep + 1}/3 for {image} ({len(rows)}/192 variant observations)", flush=True)
    finally:
        for process in workers.values():
            stop_worker(process)

    by_key = {}
    for row in rows:
        by_key.setdefault((row["image"], row["preset"], row["rep"]), {})[row["variant"]] = row
    keys = sorted(by_key)
    equality = {
        "matched_stream_pairs": sum(by_key[key]["PBC 3.0"]["stream_sha256"] == by_key[key]["PBC 3.1 development"]["stream_sha256"] for key in keys),
        "matched_pixel_pairs": sum(by_key[key]["PBC 3.0"]["decoded_rgb_sha256"] == by_key[key]["PBC 3.1 development"]["decoded_rgb_sha256"] for key in keys),
        "matched_bpp_pairs": sum(by_key[key]["PBC 3.0"]["bpp"] == by_key[key]["PBC 3.1 development"]["bpp"] for key in keys),
        "matched_mse_pairs": sum(by_key[key]["PBC 3.0"]["rgb_mse"] == by_key[key]["PBC 3.1 development"]["rgb_mse"] for key in keys),
        "matched_patch_pairs": sum(by_key[key]["PBC 3.0"]["patch_count"] == by_key[key]["PBC 3.1 development"]["patch_count"] for key in keys),
    }
    assert len(keys) == 96
    assert all(v == len(keys) for v in equality.values()), equality
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "schema": "pbc-pass15-local-release-comparison-raw-v1",
        "pbc30_commit": "2e59f1b06980aa467da0cde94fa33c3f9a0d4d51",
        "pbc31_commit": "8006f30debfc4a57289df06df21636aeb52baffd",
        "pbc31_baseline_commit": "8006f30debfc4a57289df06df21636aeb52baffd",
        "development_candidate_id": "pbc31_integral_fused",
        "candidate_pbc3_ops_sha256": hashlib.sha256((ROOT / "pbc3_ops.py").read_bytes()).hexdigest(),
        "baseline_id": "pbc31_integral_fused",
        "python": sys.version.split()[0],
        "thread_env": THREADS,
        "image_names": list(IMAGE_NAMES),
        "presets": list(PRESETS),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    for variant in workers:
        streams = {}
        for row in rows:
            if row["variant"] == variant:
                streams.setdefault((row["image"], row["preset"]), set()).add(row["stream_sha256"])
        assert all(len(values) == 1 for values in streams.values()), f"non-deterministic {variant} stream"

    summary = {}
    for preset in (*PRESETS, "overall"):
        selected = rows if preset == "overall" else [row for row in rows if row["preset"] == preset]
        summary[preset] = {}
        for field in ("encode_seconds", "decode_seconds"):
            arows = [row for row in selected if row["variant"] == "PBC 3.0"]
            brows = [row for row in selected if row["variant"] == "PBC 3.1 development"]
            a = {}
            b = {}
            for row in arows:
                a.setdefault(row["image"], []).append(row[field])
            for row in brows:
                b.setdefault(row["image"], []).append(row[field])
            if preset != "overall":
                a = {key: value for key, value in case_stats(rows, "PBC 3.0", field).items() if key[1] == preset}
                b = {key: value for key, value in case_stats(rows, "PBC 3.1 development", field).items() if key[1] == preset}
            else:
                a = {key: value for key, value in case_stats(rows, "PBC 3.0", field).items()}
                b = {key: value for key, value in case_stats(rows, "PBC 3.1 development", field).items()}
            if preset == "overall":
                a_values = [statistics.median(v) for v in a.values()]
                b_values = [statistics.median(v) for v in b.values()]
            else:
                a_values = [statistics.median(v) for v in a.values()]
                b_values = [statistics.median(v) for v in b.values()]
            reductions = [100 * (1 - statistics.median(b[key]) / statistics.median(a[key])) for key in a]
            summary[preset]["encode" if field == "encode_seconds" else "decode"] = {
                "a_mean_case_median": mean(a_values),
                "b_mean_case_median": mean(b_values),
                "a_cv_mean_percent": mean([cv(v) for v in a.values()]),
                "b_cv_mean_percent": mean([cv(v) for v in b.values()]),
                "regressions": sum(value < 0 for value in reductions),
            }
    summary["overall"]["encode"]["regressions"] = sum(1 for image in IMAGE_NAMES for preset in PRESETS if statistics.median(case_stats(rows, "PBC 3.1 development", "encode_seconds")[(image,preset)]) > statistics.median(case_stats(rows, "PBC 3.0", "encode_seconds")[(image,preset)]))
    summary["overall"]["decode"]["regressions"] = sum(1 for image in IMAGE_NAMES for preset in PRESETS if statistics.median(case_stats(rows, "PBC 3.1 development", "decode_seconds")[(image,preset)]) > statistics.median(case_stats(rows, "PBC 3.0", "decode_seconds")[(image,preset)]))
    for preset in PRESETS:
        for metric, field in (("encode", "encode_seconds"), ("decode", "decode_seconds")):
            detail = comparison(rows, "PBC 3.0", "PBC 3.1 development", field)
            if preset != "overall":
                keys_preset = [key for key in detail["case_reductions_percent"] if key.endswith("/" + preset)]
                base_a = case_stats(rows, "PBC 3.0", field)
                base_b = case_stats(rows, "PBC 3.1 development", field)
                detail["a_mean_case_median"] = mean([statistics.median(base_a[tuple(k.split("/"))]) for k in keys_preset])
                detail["b_mean_case_median"] = mean([statistics.median(base_b[tuple(k.split("/"))]) for k in keys_preset])
                detail["a_cv_mean_percent"] = mean([cv(base_a[tuple(k.split("/"))]) for k in keys_preset])
                detail["b_cv_mean_percent"] = mean([cv(base_b[tuple(k.split("/"))]) for k in keys_preset])
                detail["regressions"] = sum(detail["case_reductions_percent"][key] < 0 for key in keys_preset)
            summary[preset][metric] = detail
    worst = {}
    for metric, field in (("encode", "encode_seconds"), ("decode", "decode_seconds")):
        details = comparison(rows, "PBC 3.0", "PBC 3.1 development", field)
        speed = {key: 100 * (1 - statistics.median(case_stats(rows, "PBC 3.1 development", field)[tuple(key.split("/"))]) / statistics.median(case_stats(rows, "PBC 3.0", field)[tuple(key.split("/"))])) for key in details["case_reductions_percent"]}
        slowest_key = max(speed, key=lambda key: statistics.median(case_stats(rows, "PBC 3.1 development", field)[tuple(key.split("/"))]))
        worst[metric] = f"{slowest_key} ({statistics.median(case_stats(rows, 'PBC 3.1 development', field)[tuple(slowest_key.split('/'))]):.4f}s median)"
        worst_key = min(speed, key=speed.get)
        direction = "slower" if speed[worst_key] < 0 else "smallest reduction"
        worst[metric + "_regression"] = f"{worst_key} ({abs(speed[worst_key]):.2f}% {direction})"

    import numpy, numba
    from PIL import __version__ as pillow_version
    output = {
        "schema": "pbc-pass15-local-release-comparison-v1",
        "pbc30_commit": "2e59f1b06980aa467da0cde94fa33c3f9a0d4d51",
        "pbc31_commit": "8006f30debfc4a57289df06df21636aeb52baffd",
        "pbc31_baseline_commit": "8006f30debfc4a57289df06df21636aeb52baffd",
        "development_candidate_id": "pbc31_integral_fused",
        "candidate_pbc3_ops_sha256": hashlib.sha256((ROOT / "pbc3_ops.py").read_bytes()).hexdigest(),
        "baseline_id": "pbc31_integral_fused",
        "python": sys.version.split()[0], "numpy": numpy.__version__, "numba": numba.__version__, "pillow": pillow_version,
        "cpu_model": args.cpu_model, "p_core_logical_cpu": args.p_core,
        "thread_env": THREADS, "warmups_per_process": 4, "measured_repetitions": 3,
        "interleaving": "For each image/preset/repetition, alternate old/new process start order.",
        "source_isolation": {name: process._pbc_source for name, process in workers.items()},
        "model_sha256": baseline["model_sha256"], "equal_stream_seed": 2003,
        "image_names": list(IMAGE_NAMES), "presets": list(PRESETS), "rows": rows,
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in image_paths},
        "preset_configs": baseline["presets"],
        "measured_stream_pairs": len(keys), "matched_stream_pairs": equality["matched_stream_pairs"],
        "matched_pixel_pairs": equality["matched_pixel_pairs"], "matched_bpp_pairs": equality["matched_bpp_pairs"],
        "matched_mse_pairs": equality["matched_mse_pairs"], "matched_patch_pairs": equality["matched_patch_pairs"],
        "summary": summary,
        "slowest": {"encode": worst["encode"], "decode": worst["decode"]},
        "worst_regression": {"encode": worst["encode_regression"], "decode": worst["decode_regression"]},
        "reported_cpu_clock_mhz": args.reported_clock_mhz,
    }
    output["variants"] = {
        "PBC 3.0": {"source": str(old_source.resolve())},
        "PBC 3.1 development": {"source": str(new_source.resolve())},
    }
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_report(output, args, {"cpu_model": args.cpu_model, "core_summary": "14 cores: 6 E-cores (efficiency class 1) and 8 P-cores (class 0); 20 logical processors"})
    print(f"{args.output_json}: {len(rows)} encode/decode result pairs; identical streams {equality['matched_stream_pairs']}/{len(keys)}")


if __name__ == "__main__":
    main()
