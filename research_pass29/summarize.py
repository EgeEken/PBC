import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
RESULTS = json.loads((OUT / "results.json").read_text())
CLASSES = {
    "googlewiki.png": "text",
    "mario64.png": "synthetic graphics",
    "flower.jpg": "smooth photography",
    "cat.jpg": "detailed photography",
}
NAMES = (
    "original_raster_global4", "original_raster_global16", "original_raster_group4",
    "grouped_virtual_original_order", "best_seeded_virtual_256", "random_seed0",
    "content_order_signaled", "grouped_template",
)


def main():
    images, means = [], {}
    for row in RESULTS["images"]:
        values = {v["variant"]: v for v in row["representations"]}
        pixels = row["dimensions"][0] * row["dimensions"][1]
        groups = np.load(OUT / "arrays" / f"{row['image']}.npz")
        group_counts = {
            "distribution": np.bincount(groups["distribution_assignments"], minlength=4).tolist(),
            "distribution_structure": np.bincount(groups["structure_assignments"], minlength=4).tolist(),
        }
        reps = []
        for name in NAMES:
            v = values[name]
            reps.append({
                "variant": name, "complete_file": True, "enhancement_bytes": v["bytes"],
                "enhancement_bpp": v["bytes"] * 8 / pixels,
                "base_plus_enhancement_bytes": row["base"]["bytes"] + v["bytes"],
                "rgb_mse": v["mse"], "edge_top10_mse": v["edge_top10_mse"],
                "smooth_bottom50_mse": v["smooth_bottom50_mse"], "parts": v["parts"],
                "encode_timing_3": v.get("encode_timing_3"),
                "decode_timing_3": v.get("decode_timing_3"),
            })
        upper = values["content_order_no_map_upper_bound"]
        reps.append({
            "variant": "content_order_no_map_upper_bound", "complete_file": False,
            "diagnostic_bytes": upper["bytes"], "rgb_mse": upper["mse"],
            "excluded_order_map_bytes": upper["parts"]["order_map"],
        })
        seeds = row["seed_search"]["distribution_structure"]
        best = min(seeds, key=lambda x: x["seeded_bytes"])
        virtual = values["grouped_virtual_original_order"]["bytes"]
        images.append({
            "image": row["image"], "content_class": CLASSES[row["image"]],
            "dimensions": row["dimensions"], "source_sha256": row["base"]["source_sha256"],
            "pbc_base_bytes": row["base"]["bytes"], "pbc_base_sha256": row["base"]["sha256"],
            "pbc_base_rgb_mse": row["base_mse"], "group_counts": group_counts,
            "representations": reps, "seed_search": row["seed_search"],
            "best_seed_budget": best["budget"], "best_seed_index": best["selected_seed_index"],
            "best_seed_signal_bits": best["candidate_bits"], "best_seed_complete_bytes": best["seeded_bytes"],
            "grouping_net_saving_percent": 100 * (values["original_raster_group4"]["bytes"] - virtual) / values["original_raster_group4"]["bytes"],
            "best_seed_penalty_percent": 100 * (best["seeded_bytes"] - virtual) / virtual,
            "template_penalty_percent": 100 * (values["grouped_template"]["bytes"] - virtual) / virtual,
            "content_map_penalty_percent": 100 * (values["content_order_signaled"]["bytes"] - virtual) / virtual,
            "content_unmapped_difference_percent": 100 * (upper["bytes"] - virtual) / virtual,
            "timings": row["timings"], "peak_traced_bytes": row["peak_traced_bytes"],
        })
    for name in NAMES:
        vals = [{v["variant"]: v for v in row["representations"]}[name] for row in RESULTS["images"]]
        means[name] = {
            "mean_enhancement_bytes": float(np.mean([v["bytes"] for v in vals])),
            "mean_rgb_mse": float(np.mean([v["mse"] for v in vals])),
            "mean_enhancement_bpp": float(np.mean([
                v["bytes"] * 8 / (r["dimensions"][0] * r["dimensions"][1])
                for v, r in zip(vals, RESULTS["images"])
            ])),
        }
    summary = {
        "baseline_codec_commit": RESULTS["head"],
        "runtime_source_sha256": RESULTS["runtime_source_sha256"],
        "runtime": RESULTS["runtime"],
        "configuration": {
            "images": list(CLASSES), "cell_size": RESULTS["cell_size"],
            "groups": RESULTS["groups"], "seed_budgets": list(RESULTS["seed_budgets"]),
            "proxy_sample_cells_per_group": RESULTS["proxy_sample_cells_per_group"],
            "entropy_coder": "XZ/LZMA preset 6", "palette_training_sample_max": 65536,
            "quality_preset": "PBC3Config.quality(), release RNG, no config overrides",
        },
        "overall_means": means, "images": images,
    }
    (ROOT / "PBC_V4_NONLOCAL_GROUPING_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    report = [
        "# PBC V4: nonlocal grouping and seeded permutations",
        "",
        "Pass 29 is a bounded component study. It does not change the PBC codec or claim a V4 codec improvement.",
        "",
        "## Frozen setup",
        "",
        f"- Codec source commit: {RESULTS['head']}. Runtime source and model hashes are in PBC_V4_NONLOCAL_GROUPING_SUMMARY.json.",
        "- Each base uses PBC 3.1 Quality, release RNG and default preset settings. It is encoded once, independently decoded, and held fixed. Original-resolution signed RGB residuals use int16 subtraction. The exact stream is retained locally with its hash; base-encode duration was not retained, and no base timing comparison is claimed.",
        f"- Runtime measurements are local Windows 10, Python {RESULTS['runtime']['python'].split()[0]}, {RESULTS['runtime']['cpu']}, 20 logical CPUs, with Numba/BLAS/OpenMP constrained to one thread. No HF timing was run.",
        "- Inputs: googlewiki.png (text), mario64.png (synthetic graphics), flower.jpg (smooth photography) and cat.jpg (detailed photography). Source hashes, base stream sizes and base SHA-256 values are recorded per image.",
        "- Grid: nonoverlapping 16x16 cells. Partial cells are zero-padded in packed indices and cropped on restoration. Four groups use deterministic Lloyd clustering, farthest-first initialization and canonical center ordering.",
        "- Distribution descriptors are mean RGB residual, per-channel variance and an eight-bin signed luminance histogram. Structure descriptors add mean absolute horizontal and vertical residual gradients and their magnitude. Group IDs are transmitted at two bits per cell.",
        "- Palette fitting uses deterministic stride sampling capped at 65,536 residual pixels per palette, farthest-first initialization, Lloyd updates, signed int16 RGB centers and nearest squared-RGB indices. Controls use global palettes of 4 and 16 entries; grouped coding uses four entries per group.",
        "- Distortion is original-resolution RGB MSE after adding the quantized correction to the decoded base and clipping to 0..255. Edge and smooth diagnostics use the strongest 10% and lowest-gradient 50% of source pixels.",
        "- Index streams use XZ/LZMA preset 6. Complete size includes header, group map, palette, packed indices, XZ stream, seed signal and ordering map.",
        "- Seed PRNG is xorshift64* with 64-bit wraparound and multiplier 0x2545F4914F6CDD1D. Initial per-group state is ((seed_index+1)*0x9E3779B97F4A7C15 + (group+1)*0xD1B54A32D192ED03) mod 2^64, with a fixed nonzero replacement for zero. Descending Fisher-Yates uses j = output mod (i+1). Enumeration is seed index 0..N-1; fixed test vectors are in test_experiment.py.",
        "- The positioning idea follows PBC V2.3's get_stroke_params in PBC2_3.py at commit 9a7f2551ccf0e4f66a943de5666aeaa6509b84c1, which used RandomState(seed) for stroke row and column. This experiment defines its own PRNG and cell shuffle.",
        "",
        "## Complete-file results",
        "",
        "Seed and content reorderings use frozen grouped indices, so their reconstruction distortion is identical within each image. Base plus grouped-virtual bytes include the unchanged PBC stream.",
        "",
        "| Image / class | Base B | Global-4 B / MSE | Global-16 B / MSE | Group palette raster B / MSE | Grouped virtual B / MSE | Virtual bpp | Best searched seed B | Content map sent B | Template B | Base + virtual B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in images:
        v = {x["variant"]: x for x in row["representations"]}
        report.append(
            f"| {row['image']} / {row['content_class']} | {row['pbc_base_bytes']:,} "
            f"| {v['original_raster_global4']['enhancement_bytes']:,} / {v['original_raster_global4']['rgb_mse']:.2f} "
            f"| {v['original_raster_global16']['enhancement_bytes']:,} / {v['original_raster_global16']['rgb_mse']:.2f} "
            f"| {v['original_raster_group4']['enhancement_bytes']:,} / {v['original_raster_group4']['rgb_mse']:.2f} "
            f"| {v['grouped_virtual_original_order']['enhancement_bytes']:,} / {v['grouped_virtual_original_order']['rgb_mse']:.2f} "
            f"| {v['grouped_virtual_original_order']['enhancement_bpp']:.4f} "
            f"| {row['best_seed_complete_bytes']:,} "
            f"| {v['content_order_signaled']['enhancement_bytes']:,} "
            f"| {v['grouped_template']['enhancement_bytes']:,} "
            f"| {row['pbc_base_bytes'] + v['grouped_virtual_original_order']['enhancement_bytes']:,} |"
        )
    report += [
        "",
        f"Equal-weight means: global-4 {means['original_raster_global4']['mean_enhancement_bytes']:,.0f} B / MSE {means['original_raster_global4']['mean_rgb_mse']:.2f}; grouped palette raster {means['original_raster_group4']['mean_enhancement_bytes']:,.0f} B / MSE {means['original_raster_group4']['mean_rgb_mse']:.2f}; grouped virtual {means['grouped_virtual_original_order']['mean_enhancement_bytes']:,.0f} B / MSE {means['grouped_virtual_original_order']['mean_rgb_mse']:.2f}; global-16 {means['original_raster_global16']['mean_enhancement_bytes']:,.0f} B / MSE {means['original_raster_global16']['mean_rgb_mse']:.2f}. This small content screen is not a population estimate.",
        "",
        "### Ordering and seed search",
        "",
        "| Image | Group-palette raster B | Grouped virtual B | Net reduction | Best budget / seed | Best seeded B / index bits | Seed penalty | Content upper bound without map | Content complete B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in images:
        v = {x["variant"]: x for x in row["representations"]}
        report.append(
            f"| {row['image']} | {v['original_raster_group4']['enhancement_bytes']:,} "
            f"| {v['grouped_virtual_original_order']['enhancement_bytes']:,} "
            f"| {row['grouping_net_saving_percent']:.2f}% "
            f"| {row['best_seed_budget']} / {row['best_seed_index']} "
            f"| {row['best_seed_complete_bytes']:,} / {row['best_seed_signal_bits']} "
            f"| +{row['best_seed_penalty_percent']:.2f}% "
            f"| {v['content_order_no_map_upper_bound']['diagnostic_bytes']:,} "
            f"| {v['content_order_signaled']['enhancement_bytes']:,} |"
        )
    report += [
        "",
        "The content upper bound omits the cell-position map and is not a valid file. Complete content ordering sends four raw bytes per cell. Candidate budgets identify an index with 0, 4, 6 or 8 bits; nonzero seed fields occupy one byte in the container, with padding counted.",
        "",
        "The seed table records bytes and selected seed for both distribution and distribution-plus-structure grouping. Proxy scoring covers the first N seeds at each budget; actual XZ coding checks only the three proxy leaders plus up to two fixed enumeration probes. Rank correlations use this small sample.",
        "",
        "| Image | Grouping | 1 seed B/index | 16 seeds B/index | 64 seeds B/index | 256 seeds B/index | Proxy at 256 | Coded probes at 256 | Rank correlation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in images:
        for mode in ("distribution", "distribution_structure"):
            seeds = row["seed_search"][mode]
            q = seeds[-1]
            vals = " | ".join(f"{x['seeded_bytes']:,}/{x['selected_seed_index']}" for x in seeds)
            proxy_scale = "/".join(f"{x['proxy_seconds_budget']:.3f}" for x in seeds)
            rho = q["sampled_spearman"]
            report.append(
                f"| {row['image']} | {mode.replace('_', '+')} | {vals} "
                f"| {proxy_scale}s | {q['coded_candidates_seconds']:.3f}s "
                f"| {'n/a' if rho is None else f'{rho:.2f}'} |"
            )
    report += [
        "",
        "Structured-group cell counts (group IDs 0 through 3) show imbalance: "
        + "; ".join(f"{r['image']}: {','.join(map(str, r['group_counts']['distribution_structure']))}" for r in images) + ".",
        "",
        "### Runtime",
        "",
        "| Image | Features | Clustering | Palette fit | 256-seed proxy | Full research pass | Virtual encode median | Virtual decode median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in images:
        v = next(x for x in row["representations"] if x["variant"] == "grouped_virtual_original_order")
        t = row["timings"]
        p = row["seed_search"]["distribution_structure"][-1]["proxy_seconds_256"]
        report.append(
            f"| {row['image']} | {t['feature_extraction_seconds']:.3f}s | {t['grouping_seconds']:.3f}s "
            f"| {t['palette_quantize_setup_seconds']:.3f}s | {p:.3f}s "
            f"| {t['total_enhancement_seconds']:.2f}s | {v['encode_timing_3']['median']:.3f}s "
            f"| {v['decode_timing_3']['median']:.3f}s |"
        )
    group_savings = [r["grouping_net_saving_percent"] for r in images]
    seed_penalties = [r["best_seed_penalty_percent"] for r in images]
    template_penalties = [r["template_penalty_percent"] for r in images]
    content_penalties = [r["content_map_penalty_percent"] for r in images]
    seed_upper_gaps, palette_rate_changes, palette_mse_gains = [], [], []
    class_savings = "; ".join(f"{r['content_class']} {r['grouping_net_saving_percent']:.2f}%" for r in images)
    for r in images:
        v = {x["variant"]: x for x in r["representations"]}
        upper = v["content_order_no_map_upper_bound"]["diagnostic_bytes"]
        seed_upper_gaps.append(100 * (r["best_seed_complete_bytes"] - upper) / upper)
        palette_rate_changes.append(100 * (v["original_raster_group4"]["enhancement_bytes"] - v["original_raster_global4"]["enhancement_bytes"]) / v["original_raster_global4"]["enhancement_bytes"])
        palette_mse_gains.append(100 * (v["original_raster_global4"]["rgb_mse"] - v["original_raster_group4"]["rgb_mse"]) / v["original_raster_global4"]["rgb_mse"])
    report += [
        "",
        "Representation timings use one warm-up plus three single-threaded runs and report medians. Full research-pass time includes decoding the saved base, search, palette fitting, all controls and validation; it is not the time to encode one selected enhancement. Component byte breakdowns, timing samples, edge/smooth distortion and peak traced allocations are in the JSON summary. Peak traced allocation is a process high-water measure.",
        "",
        "Group4 palettes cost 96 bytes total; global-4 costs 24 bytes and global-16 costs 96 bytes. The 2-bit assignment map is ceil(cell_count*2/8) bytes. Content ordering adds 4*cell_count raw mapping bytes. The template stores 256 representative index bytes plus XZ-coded two-bit modular differences. Each summary row also gives base-plus-enhancement bytes and bpp.",
        "",
        "## Failure analysis",
        "",
        "The group map itself is a small, counted cost; the larger losses come from weak and sometimes highly unbalanced clusters, increased index entropy after group-specific palettes, and the raw map required by content-informed order. Grouping only marginally improves XZ size when indices are frozen. Seeded permutations change local adjacency but do not give LZMA a better global context. The shared-template deltas do not create enough repeated zero structure to repay the representatives and map. Quantization remains the dominant rate-distortion choice: global-16 reaches much lower error at substantially higher rate, while four shared entries keep size down with higher error.",
        "",
        "## Conclusions",
        "",
        f"A. Group-contiguous cells reduce complete lossless size by {min(group_savings):.2f}-{max(group_savings):.2f}% versus original raster order with the same assignments, palettes and indices. This misses the 3% screen on all images. The current clusters are highly unbalanced.",
        "",
        f"B. Seeded permutations do not produce useful savings after signaling. The best sampled files are {min(seed_penalties):.2f}-{max(seed_penalties):.2f}% larger than deterministic grouped order for every image. A 256-candidate budget does not rescue the rate.",
        "",
        f"C. Best sampled seeds are within {min(seed_upper_gaps):.2f}-{max(seed_upper_gaps):.2f}% of the content-order no-map upper bound. That arrangement is not decodable without its map. Complete content-order files are {min(content_penalties):.1f}-{max(content_penalties):.1f}% larger than grouped order.",
        "",
        f"D. Shared-template coding costs {min(template_penalties):.2f}-{max(template_penalties):.2f}% more than virtual coding. Cat is effectively tied, with added representative/difference reconstruction work.",
        "",
        f"E. Group-specific palettes lower MSE by {min(palette_mse_gains):.1f}-{max(palette_mse_gains):.1f}% versus global-4 while increasing rate by {min(palette_rate_changes):.1f}-{max(palette_rate_changes):.1f}%. Compared with global-16, grouped files are smaller but have much higher MSE. This is a rate-distortion trade.",
        "",
        f"F. Virtual-order savings by class are {class_savings}. Four images do not establish class-wide generalization.",
        "",
        "G. Stop expanding seed search. The only mechanism that merits a narrow second pass is balanced deterministic grouping: virtual order reduced bytes on all four inputs, but stayed below the screen and current groups are highly unbalanced. Keep the four-image scope and stop if complete files do not improve. No architecture change is justified yet.",
        "",
        "H. A learned encoder action remains a hypothesis. This pass computes descriptors and groups but does not train or evaluate a learned policy.",
        "",
        "The findings above use complete files, counted side information, exact index/position restoration, full RGB reconstruction and fresh-process decoding. The content-order upper bound and learned-action idea remain diagnostics or hypotheses.",
        "",
        "## Correctness and reproduction",
        "",
        "The five unittest groups cover PRNG vectors, 2/4-bit packing, signed residuals, partial cells, all-zero/constant values, three virtual orderings, shared templates, malformed/truncated/trailing data and fresh-process decoding of both containers. Actual file decodes reproduce the reported original-resolution MSE.",
        "",
        "From repository root with the pinned research environment: run research_pass29/experiment.py, then research_pass29/experiment.py --refresh-decode-timings, then research_pass29/summarize.py. Generated bases, arrays and raw output stay in ignored research_pass29/outputs/.",
        "",
        "![Cell grouping layouts](visuals/googlewiki.png.layout.png)",
        "",
        "![Rate-distortion comparison](visuals/rate_distortion.png)",
        "",
    ]
    (ROOT / "PBC_V4_NONLOCAL_GROUPING_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print("wrote summary and results report")


if __name__ == "__main__":
    main()
