import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def main():
    profile=json.loads((HERE/'profile.json').read_text())
    screen=json.loads((HERE/'screen.json').read_text())
    full=json.loads((HERE/'full.json').read_text())
    edges=json.loads((HERE/'integrated_edges.json').read_text())
    micro=json.loads((HERE/'micro.json').read_text())
    stage=json.loads((HERE/'staging.json').read_text())
    checks=json.loads((HERE/'feature_checks.json').read_text())
    raw_names=('profile.json','screen.json','full.json','edges.json','micro.json','feature_checks.json','staging.json','staging_validation.json','screen_overlap_discarded.json','micro_overlap_discarded.json','integrated_checks.json','integrated_edges.json')
    provenance={k:screen[k] for k in ('baseline_commit','baseline_id','candidate_id','python','numpy','numba','pillow','platform','cpu','thread_env','warmups','repetitions','source_sha256','configs')}
    provenance.update({'cpu_model':'12th Gen Intel Core i9-12900H (14 cores, 20 logical processors)','psutil':'7.2.2','llvmlite':'0.50.0','image_sha256':full['image_sha256'],'frozen_screen_candidate_sha256':screen['candidate_sha256'],'validated_runtime_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in screen['source_sha256']},'raw_evidence_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in raw_names},'staging':stage,'discarded_screen_reason':'A CPU-12 diagnostic microbenchmark overlapped this run. Entire run excluded; the prescribed three-repeat screen was repeated without competing diagnostic work.'})
    (HERE/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    lines=['# Pass 16 exact residual order statistics','',
        'Candidate `pbc31_feature_histogram` adds one optimization to the Pass 15 control: integer residual percentiles and quantized unique counting share a bounded histogram. All other feature expressions retain their NumPy definitions and reduction order. The histogram selects the same lower/upper order statistics and uses the same linear interpolation branch as NumPy; `rint(value / 2)` retains ties-to-even rounding. The float32 vector order, normalization, model, presets, search, RNG, patch geometry and v0 framing are unchanged.', '',
        'Only integer target/canvas arrays with residual range at most 65,535 and magnitudes at most 2**52 take the histogram path. Other inputs retain the original percentile/unique expressions. This keeps fractional feature inputs and unusually large residual ranges exact while bounding allocation. Production target and canvas arrays are int32. The helper uses the existing Numba dependency, no fastmath, no shared mutable state, and no new configuration field. It allocates at most 512 KiB of histogram storage plus small fixed arrays. First-use Numba compilation is outside the warmed speed claim.', '',
        '## Correctness', '',
        f'{checks["cases"]} direct feature cases passed byte equality for all 32 float32 elements and exact logits, value outputs and selected actions. They cover 3/4 channels, integer and floating dtypes, constants, high contrast, negative residuals, clipped/overshooting canvases, fractional values, one-pixel boundaries, full and partial rectangles, and noncontiguous reversed/strided views. Another {micro["order_stat_cases"]} tests compare float64 percentiles byte-for-byte, including small counts, interpolation boundaries, negative values and large integer offsets. Three wide-range tests exercise the original NumPy path.', '',
        'The repeated screen passed all 48 paired complete compressed byte strings, SHA-256 values, unpacked ordered-body hashes, patch counts, decoded RGB hashes, final post-LZMA bytes/bpp, internal MSE and independently scored original-resolution RGB MSE. Four additional diagnostic pairs compared every feature vector and policy output plus each sampled proposal and NumPy generator state.', '',
        'The frozen candidate then passed all 100 main/holdout image-preset pairs in both separate diagnostic and uninstrumented evaluations. Diagnostics compare feature bytes to the original extractor inside each worker and compare ordered feature/policy/proposal/RNG hashes across workers. Complete base64-encoded compressed bytes are compared directly in the parent before discarding that field from saved rows. Identical uncompressed stream bodies establish ordered patch-payload equality. The decoder source is unchanged.', '',
        'After integration, the feature function and histogram-kernel ASTs matched the frozen experimental implementation, and all 810 feature/logit/action checks passed again. Sixteen additional synthetic RGB/RGBA image-preset cases matched byte strings, decoded pixels, image mode/size and metrics across controls. Serial and two-thread concurrent execution matched in each isolated worker. Each kernel call owns its histogram. Worker RSS/peak working-set readings are saved in integrated_edges.json; they include the whole Python/Numba process and are not claimed as isolated kernel allocation measurements.', '',
        '## Full encoder screen', '',
        'Cat, Eiffel, Monet and Seagull; all four untouched presets; three warmed interleaved repetitions. Each variant runs in its own persistent process pinned to logical P-core 12, one numerical thread, Python 3.11.16 / NumPy 2.4.6 / Numba 0.68.0 / Pillow 11.3.0. Workers preload EXIF-transposed RGB images and warm all four presets. Timers wrap the entire compress call, including preparation, internal MSE and LZMA. Loading, IPC, hashing, external scoring and diagnostics are outside the timed call. Values below are means of per-image medians, including the overall row.', '',
        '| Preset | Integral-fused control (s) | Feature candidate (s) | Encode reduction |', '| --- | ---: | ---: | ---: |']
    for name,r in screen['summary'].items():
        lines.append(f'| {name} | {r["baseline"]:.6f} | {r["candidate"]:.6f} | {r["reduction_percent"]:.2f}% |')
    cvs={n:[r['cv_percent'][n] for r in screen['cases']] for n in ('baseline','candidate')}
    lines+=['',f'All 16 case medians improved. Mean within-case CV: control {statistics.mean(cvs["baseline"]):.2f}%, candidate {statistics.mean(cvs["candidate"]):.2f}%. Case reductions range {min(r["reduction_percent"] for r in screen["cases"]):.2f}% to {max(r["reduction_percent"] for r in screen["cases"]):.2f}%. Raw per-repetition times remain in screen.json. An earlier screen overlapped a diagnostic microbenchmark; the entire screen_overlap_discarded.json run is excluded. The table uses its uncontended replacement only.', '', '| Image/preset | Control median (s) | Candidate median (s) | Reduction | Control CV | Candidate CV |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in screen['cases']:
        lines.append(f'| {r["image"]}/{r["preset"]} | {r["baseline"]:.6f} | {r["candidate"]:.6f} | {r["reduction_percent"]:.2f}% | {r["cv_percent"]["baseline"]:.2f}% | {r["cv_percent"]["candidate"]:.2f}% |')
    lines+=['','## Conditional full-set confirmation','','The 20 main and five holdout images each ran all four presets. These one-pass timings are diagnostic confirmation, not an additional repeated speed estimate. The repeated screen above is the performance claim.','', '| Image | Control mean (s) | Candidate mean (s) | Reduction |','| --- | ---: | ---: | ---: |']
    for image in sorted({r['image'] for r in full['cases']}):
        cases=[r for r in full['cases'] if r['image']==image]
        a,b=[statistics.mean(r[n] for r in cases) for n in ('baseline','candidate')]
        lines.append(f'| {image} | {a:.6f} | {b:.6f} | {100*(1-b/a):.2f}% |')
    regressions=[r for r in full['cases'] if r['reduction_percent']<0]
    lines+=['',f'Full-set case timing wins: {100-len(regressions)}/100. Individual slower observations: '+('; '.join(f'{r["image"]}/{r["preset"]} ({r["reduction_percent"]:.2f}%)' for r in regressions) or 'none')+'.', '',
        '## Feature-only timings', '',
        'These medians of 15 interleaved extractor calls are isolated microbenchmarks; they are not full-encode reductions. They were rerun after all full-codec measurements, with no other diagnostic or benchmark worker running. The original overlapping micro run is retained as micro_overlap_discarded.json and excluded.', '', '| Square region | Original extractor (ms) | Candidate extractor (ms) |','| --- | ---: | ---: |']
    for size,r in micro['microbench_seconds'].items(): lines.append(f'| {size} | {1000*r["baseline"]:.3f} | {1000*r["candidate"]:.3f} |')
    lines+=['',f'Integrated worker warm RSS / peak working set: control {edges["baseline"]["warm_memory"]["rss"]/2**20:.2f} / {edges["baseline"]["after_memory"].get("peak_wset",0)/2**20:.2f} MiB; candidate {edges["candidate"]["warm_memory"]["rss"]/2**20:.2f} / {edges["candidate"]["after_memory"].get("peak_wset",0)/2**20:.2f} MiB. These small process-level differences are observational; the histogram allocation itself is bounded at 512 KiB.','','## Decision','','Promote as the next PBC 3.1 development candidate, retaining fused integral construction. The incremental whole-encode reduction is 3.88% against pbc31_integral_fused, with every preset and screen-case median improved and all 100 full-set outputs identical. No sampler experiment was needed. No HF deployment, publication, tag, protected-branch modification, model training or RD change occurred. The previous frozen baseline and Pass 15 source/evidence remain available.', '',
        f'Local staging was updated only after the full correctness, timing and edge gates passed. The complete previous artifact is retained at `{stage["retained"]}`. The replacement passed the existing isolated extraction/install, flat-module and package re-export, four-preset, deterministic RGB/RGBA, frozen Cat SHA, and immutable-PBC-3.0 v0 decode validation. New ZIP SHA-256: `{stage["zip_sha256"]}`.', '',
        'Exact source/model/data/runtime configurations and ignored raw-evidence SHA-256 values are in provenance.json. Restore the declared datasets through research/prepare_assets.py. Prepare the immutable control before reproduction:', '',
        '```', 'research_pass16/.venv/Scripts/python.exe research_pass16/run_benchmark.py --prepare-only', 'research_pass16/.venv/Scripts/python.exe research_pass16/check_features.py', 'research_pass16/.venv/Scripts/python.exe research_pass16/check_order_stats.py', 'research_pass16/.venv/Scripts/python.exe research_pass16/run_benchmark.py', 'research_pass16/.venv/Scripts/python.exe research_pass16/run_benchmark.py --full', 'research_pass16/.venv/Scripts/python.exe research_pass16/check_edges.py --integrated', '```', '',
        'Do not overlap diagnostic microbenchmarks with timed workers. build_release_staging.py requires passed local results and an untouched known Pass 15 staging artifact; it refuses existing candidate/archive destinations. It is an explicit local staging action, not part of benchmark execution.']
    (HERE/'PBC_FEATURE_OPTIMIZATION.md').write_text('\n'.join(lines)+'\n')
    summary=f'''# PBC Pass 16 summary

Promoted development candidate: `pbc31_feature_histogram` (Pass 15 fused integral images plus exact integer residual order statistics). Relative to `pbc31_integral_fused` at c0d0339a509ceb4027574fc388cef18a643b14e5, full encoding time fell **3.88%** in the uncontended four-image, four-preset, three-repeat local screen. All 16 case medians improved. Compression/Balanced/Quality/High Quality reductions were **3.57% / 3.10% / 3.96% / 4.62%**.

The deployed legacy policy uses `cheap=False` and all 32 full features. Percentiles and quantized unique counting are the largest avoidable feature costs. They now share a bounded histogram for integer codec regions; other statistics and normalization retain their definitions. The isolated extractor is faster, but promotion rests on complete compress-call timing with internal MSE and LZMA enabled.

Direct feature tests: 810/810 vectors, logits, values and actions match; 78/78 float64 order-statistic tests and three wide-range cases match. Repeated screen: 48/48 measured stream pairs match. Full main/holdout gate: 100/100 image-preset pairs match complete streams, ordered patch bodies/counts, decoded RGB pixels, final bpp, independent RGB MSE, feature vectors, policy outputs and NumPy proposal/RNG diagnostics. Sixteen RGB/RGBA cases also match across serial and concurrent execution. No sampler alternative was attempted because the feature candidate passed.

The resulting best PBC 3.1 development configuration keeps the same model, presets, seed 2003, release-compatible NumPy RNG, internal MSE, LZMA and v0 stream. Only pbc3_features.py changes relative to the Pass 15 codec. No RD or decode-speed change is claimed. The new warmed incremental gain must not be added to the previous 34.00% local PBC 3.0 encode reduction or the separate pre-fused HF measurements. This candidate has not been timed on HF.

Local staging was rebuilt and passed the existing isolated release validation, including top-level/package APIs and immutable PBC 3.0 v0 decoding. Prior Pass 15 ZIP/source remain in `{stage['retained']}`. New ZIP SHA-256: `{stage['zip_sha256']}`. No publication, release tag, HF deployment, main/rust modification or force push occurred.

Reports: PBC_FEATURE_EXTRACTION_PROFILE.md and PBC_FEATURE_OPTIMIZATION.md. Reproducible scripts, exact configurations, input/source hashes and raw evidence digests accompany them in this directory. Raw JSON, model/data copies, binaries and caches remain ignored. The next bounded opportunity is proposal sampling: hoist invariant bounds/logarithms and reduce Python bookkeeping while preserving each conditional NumPy RNG call; no sampler implementation is claimed here.
'''
    (HERE/'PBC_PASS16_SUMMARY.md').write_text(summary)
    print('Reports and compact provenance written')

if __name__=='__main__': main()
