import collections
import json
from pathlib import Path
import statistics
import subprocess

from benchmark import OUT, ROOT, PRESETS


def mean(values):
    return statistics.mean(values)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def load(name):
    return json.loads((OUT / name).read_text())


def baseline_report():
    rows = [json.loads(line) for line in (OUT / 'baseline.jsonl').read_text().splitlines()]
    assert len(rows) == 900
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['image'], row['preset'], row['source']].append(row)
    cases = {}
    for key, trials in groups.items():
        assert len(trials) == 3
        assert len({r['stream_sha256'] for r in trials}) == 1
        assert len({r['pixels_sha256'] for r in trials}) == 1
        cases[key] = {**trials[0], **{metric: statistics.median(r[metric] for r in trials) for metric in ('encode_seconds', 'decode_seconds')}}
    comparisons = []
    for key, base in cases.items():
        if key[-1] != 'release':
            continue
        for source in ('old', 'current'):
            other = cases[(*key[:3], source)]
            comparisons.append({'dataset': key[0], 'image': key[1], 'preset': key[2], 'source': source,
                                'bytes_equal': base['stream_sha256'] == other['stream_sha256'],
                                'pixels_equal': base['pixels_sha256'] == other['pixels_sha256'],
                                **{metric + '_pct': (other[metric] / base[metric] - 1) * 100 for metric in ('mse', 'bpp', 'encode_seconds', 'decode_seconds')}})
    (OUT / 'paired_comparisons.json').write_text(json.dumps(comparisons, indent=2))
    summary = load('baseline_summary.json')
    values = []
    for dataset in ('hpt_data', 'hpt_data_val'):
        for preset in PRESETS:
            for source in ('release', 'old', 'current'):
                s = next(r for r in summary if (r['dataset'], r['preset'], r['source']) == (dataset, preset, source))
                values.append((dataset, preset, source, f"{s['mse']:.6f}", f"{s['bpp']:.6f}",
                               f"{s['encode_seconds']:.6f}", f"{s['encode_seconds_mean_image_median']:.6f}",
                               f"{s['decode_seconds']:.6f}", f"{s['decode_seconds_mean_image_median']:.6f}"))
    wins = []
    aggregates = []
    for dataset in ('hpt_data', 'hpt_data_val'):
        for preset in (*PRESETS, 'all'):
            selected = [r for r in comparisons if r['source'] == 'current' and r['dataset'] == dataset and (preset == 'all' or r['preset'] == preset)]
            counts = []
            for metric in ('mse', 'bpp', 'encode_seconds', 'decode_seconds'):
                v = [r[metric + '_pct'] for r in selected]
                counts.append(f'{sum(x < -1e-9 for x in v)}/{sum(abs(x) <= 1e-9 for x in v)}/{sum(x > 1e-9 for x in v)}')
            wins.append((dataset, preset, *counts))
            bases = [r for k, r in cases.items() if k[0] == dataset and k[-1] == 'release' and (preset == 'all' or k[2] == preset)]
            others = [cases[(r['dataset'], r['image'], r['preset'], 'current')] for r in bases]
            delta = {metric: 100 * (mean(r[metric] for r in others) / mean(r[metric] for r in bases) - 1) for metric in ('mse', 'bpp', 'raw_bytes', 'final_bytes', 'encode_seconds', 'decode_seconds')}
            aggregates.append({'dataset': dataset, 'preset': preset, **delta})
    (OUT / 'aggregate_deltas.json').write_text(json.dumps(aggregates, indent=2))
    delta_table = table(['Set', 'Preset', 'MSE %', 'bpp %', 'raw bytes %', 'final bytes %', 'encode median %', 'decode median %'],
                        [(r['dataset'], r['preset'], *(f'{r[k]:+.3f}' for k in ('mse', 'bpp', 'raw_bytes', 'final_bytes', 'encode_seconds', 'decode_seconds'))) for r in aggregates])
    old_equal = sum(r['bytes_equal'] and r['pixels_equal'] for r in comparisons if r['source'] == 'old')
    current_equal = sum(r['bytes_equal'] for r in comparisons if r['source'] == 'current')
    cv_rows = [(s['dataset'], s['preset'], s['source'], f"{100*s['encode_seconds_mean_image_cv']:.2f}", f"{100*s['decode_seconds_mean_image_cv']:.2f}") for s in summary]
    outliers = sorted((r for r in comparisons if r['source'] == 'current'), key=lambda r: r['mse_pct'], reverse=True)[:8]
    outlier_table = table(['Set', 'Image', 'Preset', 'MSE %', 'bpp %', 'encode median %'],
                          [(r['dataset'], r['image'], r['preset'], f"{r['mse_pct']:+.3f}", f"{r['bpp_pct']:+.3f}", f"{r['encode_seconds_pct']:+.3f}") for r in outliers])
    patch_counts = []
    for source in ('release', 'old', 'current'):
        for preset in PRESETS:
            counts = [r['patch_count'] - 3 for k, r in cases.items() if k[0] == 'hpt_data' and k[2] == preset and k[3] == source]
            patch_counts.append((source, preset, min(counts), f'{mean(counts):.2f}', max(counts)))
    deployment_note = 'Remote image-upload verification was not performed without explicit authorization; public metadata/source/status GETs succeeded.'
    if (OUT / 'deployment_comparison.json').exists():
        deployment = load('deployment_comparison.json')
        deployment_note = f"After explicit user authorization, exactly four original googlewiki.png uploads were sent to the HF /api/compress endpoint, one per preset. {sum(r['remote_bytes_equal_local'] for r in deployment)}/4 files and decoded pixel arrays match local encodes under the same RGBA policy. All four remote previews, displayed sizes and displayed RGBA MSE values match actual decoded artifacts. Quality differs: RGB MSE {deployment[2]['decoded_rgb_mse']:.6f} remotely versus {deployment[2]['local_decoded_rgb_mse']:.6f} locally; bytes {deployment[2]['final_bytes']} versus {deployment[2]['local_final_bytes']}. The first differing decoded patch is zero-based index 14 (15th stream patch, 11th iterative patch after four initialization patches), with a different selected box. The earlier decoded patches match. The source/model checks match the reported immutable HF revision, but the reason for this cross-runtime trajectory difference is unresolved; it is not dismissed as numerical noise. No additional compression requests were made to investigate it. See deployment_comparison.json and the retained local/remote RGBA streams."
        deployment_note += '\n\n' + table(['Preset', 'Bytes equal', 'Decoded pixels equal', 'Raw bytes', 'Final bytes', 'bpp', 'RGB MSE', 'RGBA MSE'],
                                          [(r['preset'], r['remote_bytes_equal_local'], r['remote_decoded_pixels_equal_local'], r['raw_bytes'], r['final_bytes'], f"{r['bpp']:.6f}", f"{r['decoded_rgb_mse']:.6f}", f"{r['decoded_rgba_mse']:.6f}") for r in deployment])
    text = f'''# PBC benchmark validation

## Verdict

The new isolated harness is suitable for controlled local default-preset comparisons. Existing Quick RD/sweep data is not sufficient for release acceptance. The broader benchmark's core PBC metric function is correctly designed, but its stored database rows do not establish current source provenance.

There are 900 measured rows: 25 original images × four presets × three implementations × three repeats, preceded by a 48-row smoke test (two images, all presets, two repeats). Release and old have identical serialized hashes and decoded pixel hashes in **{old_equal}/100 distinct cases**. Current is byte-identical to release in **{current_equal}/100**. All 900 encoder previews equal actual decoded output. Every source/image/preset is deterministic across the three repeats.

## Audit findings and local corrections

| Component | Finding | Correction/coverage |
|---|---|---|
| Quick RD, `pbc3_quick_rd.py:86`, `:104` | Current and old evaluators score `res.image`, do not call decompress, use internal encode duration, and omit decode time/raw bytes/source hashes. Runs group old before current; old is restarted per preset, with different warmup. | Replaced for this audit by isolated `research_pass1/benchmark.py`; website runner left unchanged. |
| Old subprocess | Codec imports are isolated, but model default is relative to root cwd in existing Quick RD, so it can load root weights. These weights happen to match here. Old learned_filler alters sys.path. | New workers have implementation-specific cwd and absolute model paths. Loaded helper/module inventories were inspected after learned warmup. |
| Sweep, `pbc3_sweep.py:166` | Uses encoder preview/internal duration; database results can reflect older code or params. | Not used as a release comparison. No database modified. |
| Broader benchmark, `pbc3_benchmark.py:134` | Calls compress and decompress, times complete calls with perf_counter, scores decoded RGB using safe float conversions, counts complete final file. | Locally exercised by `validate_backend.py`; raw artifacts and exact backend coverage are recorded in backend_validation.json. |
| Result properties, `pbc3_types.py:228` | `decode_time` and `decode_seconds` alias the result's single operation duration. On an encoder result this is encode time. | Harness measures calls explicitly and never infers decode timing from an encoder result. Server decoder uses a decompress result, which is valid. |
| Website, `server.py:329` | Auto mode routes to production presets; UI accepts overrides for learned enable/q and forces one learned candidate/top-k. MSE comes from encoder result, displayed size/time rounded. The website preserves an opaque alpha channel; its MSE then averages RGBA, while benchmark loaders discard alpha and measure RGB. | All benchmark previews are empirically equal to decode. Four RGB-normalized local API tests match the harness files, decoded pixels/MSE, and size; an original-RGBA test confirms the different metric. This alpha-policy discrepancy remains in the existing website. Remote GETs alone cannot verify returned compressed files. |
| Provenance and stale output | Quick RD JSON has a creation time but no commit/model/config/input hashes. Deployed `/api/quick_rd/results` reports `new_exists=false`; available DB rows are historical. | New per-image rows include exact input, config, model and stream identifiers. Remote observations stored separately from local timing. |
| Warmup/threads/order | Existing preload helpers differ; neither existing benchmark enforces all numerical thread settings or balanced per-case trial order. | Same complete learned encode/decode warmup per preset, one CPU affinity, six thread variables=1, persistent separate workers, source order rotated by trial/image/preset. |

No codec, model, production default, existing database, or deployed file was repaired or synchronized. Corrections are confined to the new reproducible audit harness.

For the original opaque-RGBA googlewiki.png, the local balanced API reports MSE {load('backend_alpha_observation.json')['displayed_mse']:.2f}; the decoded RGB MSE is {load('backend_alpha_observation.json')['decoded_rgb_mse']:.6f}, and RGBA MSE is {load('backend_alpha_observation.json')['decoded_rgba_mse']:.6f}. A perfect alpha channel lowers the four-channel average to 75% of its RGB value. The original RGBA stream is retained in backend_original_rgba.pbc3. The RGB benchmark does not make a native-RGBA quality claim.

Local TestClient initially stalled in Windows asyncio's fallback socketpair inside the sandbox. The same in-process tests succeeded outside that restriction; those local tests contacted no external host. Startup lifespan/provider probes were excluded, while real form handling, compression, response serialization and decode verification were exercised.

{deployment_note}

## Measurement method

Both sets are exactly the supplied hpt_data (20) and hpt_data_val (5), including uppercase `.JPG` holdout filenames. Input paths and SHA-256 are in baseline_metadata.json. Every source opens the same file, applies EXIF transpose, converts to RGB, and materializes it before the timed call. Alpha is discarded consistently for this RGB benchmark. Native RGBA quality is outside its scope.

Each worker pins itself to logical CPU 0 before warmup and measured encodes. NumPy/BLAS/Numba thread environment settings are established before worker imports. Only one worker executes a requested operation at a time. All sources have the same Python/dependencies/hardware. OS/background scheduling and frequency variation remain possible. No HF timing is merged with these local rows.

Full encode wall time surrounds PBC3.compress including preprocessing, model-object construction/cache lookup, search, serialization, LZMA, final conversion/resize and codec-internal MSE. File loading and the harness's external validation are outside that interval. Decode wall time surrounds PBC3.decompress on the generated bytes, including entropy decoding, bit parsing, reconstruction and original-size conversion. JSON/process IPC, disk artifact writes and independent MSE/hash computation are outside both codec timers. A warmed model-weight cache is retained equally; no prepared image arrays or canvas/search state are reused across encodes.

Decoded image dimensions must equal EXIF-normalized source dimensions; no benchmark-side resize hides mismatches. MSE is mean RGB squared error with float64 subtraction/reduction in baseline rows. bpp is `8 * len(final_data) / (original_width * original_height)`. Raw bytes are the uncompressed bitstream body plus the same six-byte outer header; raw_body_bytes is also retained. Patch count comes from the emitted stream header and includes initialization patches. No patch-bit proxy is substituted for final size.

One full learned encode/decode on googlewiki.png per preset precedes timing. First calls in fresh workers are retained in metadata warmup_rows, but Numba disk caches may already exist; these are not pristine install/cold-boot timings. Final timing claims concern warmed calls only. Three interleaved measured repetitions are retained individually. Tables give both arithmetic means of all calls and averages of each image's median duration. Percent timing comparisons use the latter. CV is population standard deviation / mean over three repeats, then averaged across images; three repeats give only a limited noise estimate.

## Controlled results

{table(['Set', 'Preset', 'Source', 'MSE', 'bpp', 'encode mean s', 'encode median s', 'decode mean s', 'decode median s'], values)}

Current relative to immutable release (percent changes of equal-weight image/preset means; lower is better):

{delta_table}

Raw/final byte percentage means weight images by their bytes; mean bpp weights image/preset cases equally. They need not have the same percentage change. Both are reported rather than conflated.

Wins / ties / losses against release, with each image/preset counted once and latency based on within-image medians:

{table(['Set', 'Preset', 'MSE W/T/L', 'bpp W/T/L', 'encode W/T/L', 'decode W/T/L'], wins)}

Largest MSE regressions:

{outlier_table}

All per-image changes, including rate outliers, remain in paired_comparisons.json. The current root does not establish a release improvement across quality, final rate, encode and decode jointly. Mean wins must not hide the image/preset regressions above.

Main-set emitted iterative patch counts (three initialization patches excluded):

{table(['Source', 'Preset', 'Minimum', 'Mean', 'Maximum'], patch_counts)}

The filler can terminate before its requested budget. Different RNG trajectories change both patch contents and stopping points, so the combined root speed difference must not be attributed solely to output-preserving implementation changes.

## Timing variation

{table(['Set', 'Preset', 'Source', 'encode mean CV %', 'decode mean CV %'], cv_rows)}

No independent HF old-versus-new timing run was launched. Single remote compression checks and HTTP/encode durations are retained separately; they are not controlled, warmed HF latency comparisons and are never pooled with local timings.

## Reproduce

From repository root, run `python research_pass1/benchmark.py --smoke --repeats 2`, inspect results, then `python research_pass1/benchmark.py --repeats 3`. These commands overwrite only their own research output files. The immutable release snapshot is verified against GitHub tree identifiers in github_provenance.json; do not replace it with a current-code ablation.

Raw rows: research_pass1/baseline.jsonl and baseline.csv. Expanded metadata/configs: baseline_metadata.json. Summaries: baseline_summary.json, aggregate_deltas.json, paired_comparisons.json. First-repeat compressed files: streams/{{release,old,current}}/{{hpt_data,hpt_data_val}}/. Exact original user status and final artifact manifest are retained separately.
'''
    (ROOT / 'PBC_BENCHMARK_VALIDATION.md').write_text(text, encoding='utf-8')


def results_report():
    floor = load('resolution_floor.json')
    profiles = load('profile_summary.json')
    screen = load('screen_results.json')
    singles = load('pruning_individual_tests.json')
    comparison = load('stream_comparison.json')
    rng = load('rng_attribution.json')
    types = {
        'cat.jpg': 'animal photograph', 'cathedral.png': 'architecture photograph', 'coast.jpg': 'landscape photograph',
        'ducklings.jpg': 'animal photograph', 'eiffel.png': 'architecture photograph', 'field.jpg': 'landscape photograph',
        'flag.jpg': 'fabric photograph', 'flower.jpg': 'flower/bokeh photograph', 'flowers.jpg': 'flower/bokeh photograph',
        'googlewiki.png': 'text/synthetic', 'mario64.png': 'game/synthetic', 'monet.jpg': 'textured artwork',
        'monkey.png': 'puppet/meme photograph', 'panorama.png': 'landscape photograph', 'park.jpg': 'landscape photograph',
        'perlman.png': 'portrait photograph', 'ratedistortion.png': 'text/synthetic', 'seagull.jpg': 'animal photograph',
        'stockpoint.jpg': 'portrait photograph', 'sunset.jpg': 'landscape photograph', '15k_kills.png': 'text/synthetic',
        'ducks.jpg': 'animal photograph', 'old_man_standing.jpg': 'portrait photograph',
        'pigeon.JPG': 'snow/urban photograph', 'pigeons.JPG': 'animal photograph'}
    by_type = []
    for image_type in sorted(set(types.values())):
        for limit in (250000, 500000, 1000000, 2000000):
            subset = [r for r in floor if types[r['image']] == image_type and r['limit'] == limit and r['preset'] == 'balanced']
            by_type.append({'image_type': image_type, 'limit': limit, 'images': len(subset),
                            'source_pixels_mean': mean(r['width'] * r['height'] for r in subset),
                            'roundtrip_mse': mean(r['roundtrip_mse'] for r in subset),
                            'balanced_release_mse': mean(r['baseline_mse'] for r in subset)})
    (OUT / 'resolution_floor_by_type.json').write_text(json.dumps(by_type, indent=2))
    floors = []
    for dataset in ('hpt_data', 'hpt_data_val'):
        for preset in PRESETS:
            for limit in (250000, 500000, 1000000, 2000000):
                subset = [r for r in floor if (r['dataset'], r['preset'], r['limit']) == (dataset, preset, limit)]
                floors.append((dataset, preset, limit, f"{mean(r['roundtrip_mse'] for r in subset):.4f}",
                               f"{mean(r['baseline_mse'] for r in subset):.4f}",
                               f"{100*mean(r['roundtrip_mse'] for r in subset)/mean(r['baseline_mse'] for r in subset):.2f}"))
    dominant = []
    for operation in ('encode', 'decode'):
        times = collections.defaultdict(list)
        for r in profiles:
            if r['operation'] == operation:
                times[r['phase']].append(r['exclusive_seconds'])
        total = mean(next(r['total_seconds'] for r in profiles if r['image'] == name and r['repeat'] == repeat and r['operation'] == operation) for name in ('flower.jpg', 'monet.jpg', 'googlewiki.png') for repeat in (0, 1))
        for phase, values in sorted(times.items(), key=lambda item: mean(item[1]), reverse=True)[:12]:
            dominant.append((operation, phase, f'{mean(values):.6f}', f'{100*mean(values)/total:.2f}'))
    variants = []
    variant_metrics = []
    for variant in sorted({r['experiment'] for r in screen if r['experiment'].startswith(('C_', 'D_'))}):
        subset = [r for r in screen if r['experiment'] == variant]
        delta = {key: 100*(mean(r[key] for r in subset)/mean(r['baseline_' + key] for r in subset)-1)
                 for key in ('mse', 'bpp', 'encode_seconds', 'decode_seconds')}
        delta['raw_bytes'] = 100*(mean(r['raw_bytes'] for r in subset)/mean(r['baseline_raw_bytes'] for r in subset)-1)
        delta['final_bytes'] = 100*(mean(r['final_bytes'] for r in subset)/mean(r['baseline_final_bytes'] for r in subset)-1)
        wins = sum(r['mse'] < r['baseline_mse'] and r['bpp'] <= r['baseline_bpp'] for r in subset)
        outcome = 'Latency tradeoff' if delta['mse'] < -1e-6 and delta['bpp'] <= 1e-6 else ('RD tradeoff' if delta['bpp'] < -1e-6 and delta['mse'] > 1e-6 else 'Regression / no RD gain')
        variants.append((variant, *(f'{delta[k]:+.3f}' for k in ('mse', 'bpp', 'raw_bytes', 'final_bytes', 'encode_seconds', 'decode_seconds')), f'{wins}/{len(subset)}', outcome))
        variant_metrics.append({'variant': variant, 'rows': len(subset), **delta, 'outcome': outcome})
    (OUT / 'screen_summary.json').write_text(json.dumps(variant_metrics, indent=2))
    e_pairs = []
    for fresh in [r for r in screen if r['experiment'] == 'E_normal_budget']:
        prefix = next(r for r in screen if r['experiment'] == 'E_prefix' and (r['image'], r['preset'], r['budget']) == (fresh['image'], fresh['preset'], fresh['budget']))
        e_pairs.append({'image': fresh['image'], 'preset': fresh['preset'], 'budget': fresh['budget'],
                        'bytes_equal': fresh['stream_sha256'] == prefix['stream_sha256'],
                        'fresh_mse': fresh['mse'], 'prefix_mse': prefix['mse'], 'fresh_bpp': fresh['bpp'], 'prefix_bpp': prefix['bpp']})
    (OUT / 'prefix_budget_comparison.json').write_text(json.dumps(e_pairs, indent=2))
    dominance = []
    for candidate in [r for r in screen if r['experiment'].startswith(('C_', 'D_'))]:
        controls = [r for r in screen if r['experiment'].startswith('E_') and r['image'] == candidate['image'] and r['preset'] == candidate['preset']]
        dominated = any(r['mse'] <= candidate['mse'] and r['bpp'] <= candidate['bpp'] and (r['mse'] < candidate['mse'] or r['bpp'] < candidate['bpp']) for r in controls)
        dominance.append({'image': candidate['image'], 'variant': candidate['experiment'], 'dominated_by_observed_prefix_or_fresh_budget': dominated})
    (OUT / 'candidate_control_dominance.json').write_text(json.dumps(dominance, indent=2))
    per_image = []
    for name in ('flower.jpg', 'monet.jpg', 'googlewiki.png'):
        for variant in ('C_single_strict', 'C_single_tolerance_0.1pct', 'C_greedy', 'D_refit_5', 'D_refit_15', 'D_refit_all'):
            r = next(r for r in screen if r['image'] == name and r['experiment'] == variant)
            per_image.append((name, variant, f"{r['mse']:.4f}", f"{r['bpp']:.6f}", r['raw_bytes'], r['final_bytes'], f"{r['extra_seconds']:.3f}", r.get('accepted', int(r.get('removed_patch', -1) >= 0))))
    greedy_metrics = next(r for r in variant_metrics if r['variant'] == 'C_greedy')
    refit_metrics = next(r for r in variant_metrics if r['variant'] == 'D_refit_all')
    text = f'''# PBC first-pass results

## Decision

Use the verified immutable public release as the research baseline. The historical old runtime is equivalent on the tested default-preset sets but is not source-identical. Current root is a combined trajectory/performance variant, not clean release PBC 3.0. No PBC 3.1 release success is established by this pass.

Phases 1–2 passed the default-preset comparison gate. All five requested experiment categories ran locally. A uses the full main and holdout datasets; B profiles three representative images; C/D screen the balanced preset on flower.jpg (smooth photographic/bokeh), monet.jpg (textured art), and googlewiki.png (text/synthetic); E screens these three images across all four presets. No new algorithm was enabled in production. C/D are pilot proposals, not exhaustive optimization of their candidate spaces.

## Attribution diagnostic

Stream inspection checks all 100 first-repeat release cases against old and current. Old equal headers/init/payloads: {sum(r['byte_equal'] for r in comparison if r['source']=='old')}/100. Current matching initial three channel patches: {sum(r['init_patches_equal'] for r in comparison if r['source']=='current')}/100. Earliest differing patch indices are retained in stream_comparison.json. This localizes differences beyond initialization without calling them numerical noise.

Replacing only current root's RNG constructor with NumPy default_rng in an isolated process produced release-identical files on {sum(r['current_with_numpy_rng_equals_release'] for r in rng)}/{len(rng)} representative image/preset cases. This is a causal ablation for attribution, not a release comparison or proposed improvement. Default current root comparisons still use its real custom RNG. No production RNG was changed.

## A — Working-resolution round trip

Actual release preprocessing was used: EXIF-normalized RGB -> YCbCr -> PIL bicubic working-size image -> RGB -> PIL bicubic source-size output, with no quantization or residual patches. Four offline pixel limits were measured without changing defaults.

This is a preprocessing round-trip diagnostic, not a strict mathematical lower bound on every possible working-grid representation: precompensated samples could differ from the codec's direct downsample, and YCbCr integer conversion adds error even without resizing. Ratios below compare means, not a causal partition of the final codec error.

{table(['Set', 'Preset', 'Pixel limit', 'Roundtrip RGB MSE', 'Final release MSE', 'Ratio %'], floors)}

Per-image source/working dimensions, all preset comparisons and round-trip MSE are in resolution_floor.csv/json. Image-type grouping and size diagnostics are in resolution_floor_by_type.json. These controls do not measure full-encode RD or latency at larger limits, and do not justify raising production resolution.

At the production 250k limit, grouped by descriptive image type (main and holdout combined; group counts are small):

{table(['Image type', 'Images', 'Mean source MP', 'Roundtrip MSE', 'Balanced release MSE'], [(r['image_type'], r['images'], f"{r['source_pixels_mean']/1e6:.3f}", f"{r['roundtrip_mse']:.4f}", f"{r['balanced_release_mse']:.4f}") for r in by_type if r['limit']==250000])}

## B — Production profiling

The actual current root compress/decompress pipeline was warmed and traced twice on each of the three screening images, balanced preset. Existing nested TimingTrace spans separate preprocessing, initialization, residual/integral/anchor work, proposals, features, MLP, quantization/resampling, application, channel errors, serialization, LZMA and output conversion. Exclusive durations below avoid summing nested inclusive spans. Tracing overhead remains; release speed conclusions use the separate uninstrumented benchmark.

{table(['Operation', 'Phase', 'Mean exclusive s', 'Share of traced operation %'], dominant)}

Complete inclusive/exclusive phase tables and raw events/counters are retained in profile_summary.csv and profile_*.json. This identifies where to investigate, not a measured optimization.

Surgical leads: compile the Python proposal sampling/prescore loop together while preserving its chosen RNG sequence exactly; investigate reusing visible-error state after a local patch instead of repeatedly rebuilding full-channel arrays; inspect full feature extraction (the production model uses percentiles and unique) before changing MLP code; specialize bit-grid read/write if serialization/parsing is material. PIL bicubic and NumPy sort/cumsum/BLAS calls already execute native code. A C/Rust wrapper alone does not remove their work. The workspace already contains rejected/reverted integral-fusion experiments; this profile does not justify repeating them without a different mechanism and HF evidence. Any Numba replacement needs byte/pixel equivalence tests and independent HF timing. LZMA stays in place; post-wrapper bytes remain the rate metric. No native rewrite was attempted.

## C — Post-encode pruning

Every available iterative single-patch removal was tested for each of the three balanced trajectories ({len(singles)} candidates in total). A requested budget of 50 does not imply 50 accepted patches: the production filler can stop earlier when its selected proposal has no positive reduction. Actual emitted patch counts are retained in all benchmark rows. Every candidate had its patch count/header rewritten, was entropy-packed, and was fully decoded before RGB MSE evaluation. Initialization patches were retained.

Strict single removal chooses the smallest final file among deletions with MSE no greater than the full trajectory. Tolerance permits at most +0.1% relative RGB MSE. Greedy is one reverse pass, accepting each deletion only when final MSE does not rise and post-LZMA bytes strictly decrease. These variants start independently from the same complete trajectory. Selection scans and replay/serialization/metric computation are all included in extra encode time; candidate decode measurement is part of this optimization cost.

## D — Frozen-geometry backward refitting

Independent last-5, last-15 and complete backward passes keep patch geometry, order, cell size and maximum palette bitcount fixed. A last-15 pass revisits all available patches when fewer than 15 were emitted. Original palette-mask length and positive-bias semantics are retained; palette bounds, generated mask and indices may be regenerated within that representation.

For each revisited patch, the decoded internal canvas minus that patch's deterministic delta estimates the other fixed contributions. The ordinary bicubic residual-grid/palette/quantization constructor proposes replacement values against the final working target. Every proposal is then serialized and fully decoded, and accepted only for strictly lower original-size RGB MSE with no increase in actual post-LZMA bytes. This is a conservative single refit proposal per patch, not a solved global coefficient optimizer. Subtraction is used only to propose values; final acceptance always uses proper replay. Preset warmup is disabled and the script rejects multistage/warmup streams.

## C/D screening results

Percent changes compare equal-weight means across the three balanced images. Raw/final byte means and mean bpp have different weighting. Decode measurements here are single screened calls and can be noisy; they are not release timing evidence. Baseline encode cost is the median of the three uninstrumented release calls plus all measured optimization overhead.

{table(['Variant', 'MSE %', 'bpp %', 'raw bytes %', 'final bytes %', 'encode %', 'decode %', 'strict RD image wins', 'Classification'], variants)}

{table(['Image', 'Variant', 'MSE', 'bpp', 'raw bytes', 'final bytes', 'extra encode s', 'accepted'], per_image)}

Per-image candidates are in screen_results.csv/json; all individual deletion results are in pruning_individual_tests.csv/json. candidate_control_dominance.json compares each retained C/D operating point against the observed prefix/fresh-budget controls. Nondominance against this small observed set is not proof of an improved continuous RD frontier.

{table(['Variant', 'Dominated by an observed same-image prefix/fresh-budget RD point', 'Cases'], [(variant, sum(r['dominated_by_observed_prefix_or_fresh_budget'] for r in dominance if r['variant']==variant), sum(r['variant']==variant for r in dominance)) for variant in sorted({r['variant'] for r in dominance})])}

## E — Prefix and patch-budget analysis

Existing completed release trajectories were serialized/decoded at initialization-only and selected iterative budgets through the full trajectory (including late budgets). No search reran for a prefix. Fresh normal encodes were also run at selected identical smaller budgets; changing patch_count changes search interpolation and feature step fraction.

Of {len(e_pairs)} same-budget fresh/prefix pairs, {sum(r['bytes_equal'] for r in e_pairs)} are byte-identical. Individual RD points, including poor or counterproductive late prefixes, are in screen_results.csv; exact fresh/prefix comparisons are in prefix_budget_comparison.json. A prefix extracted after a complete encode still pays for the complete search plus extraction. Fresh-budget encode time is reported separately, without inheriting that full trajectory cost. This diagnostic alone is not a new stopping policy.

## Scope, reproducibility and next iteration

No C/D candidate is accepted as a release improvement from this screen. No C/D variant was promoted to full main/holdout evaluation: strict greedy pruning changes mean MSE by only {greedy_metrics['mse']:+.3f}% and mean bpp by {greedy_metrics['bpp']:+.3f}% while increasing total encode cost by about {greedy_metrics['encode_seconds']:.0f}%; complete backward refitting changes MSE by {refit_metrics['mse']:+.3f}% and bpp by {refit_metrics['bpp']:+.3f}% while increasing encode cost by about {refit_metrics['encode_seconds']:.0f}%. These are small RD leads with substantial latency penalties. A three-image balanced pilot cannot establish four-preset success. Broader holdout confirmation is still required before any claimed generalization. No combined prune/refit/stopping variant or large combinatorial sweep was run.

Next work should preserve the immutable baseline and frozen stream artifacts, then use the profile to select one output-preserving optimization with a credible single-core/HF cost reduction. For an RD lead, improve or bound its incremental cost before promoting it to a full dataset run. Do not raise resolution, change RNG, or alter patch budget and describe that alone as a release win.

Commands from repository root, after the baseline: `python research_pass1/experiments.py compare`, `python research_pass1/rng_diagnostic.py`, `python research_pass1/experiments.py floor`, `python research_pass1/experiments.py profile`, `python research_pass1/experiments.py screen`, `python research_pass1/validate_backend.py`, then `python research_pass1/generate_reports.py`. Run sequentially, with no concurrent CPU benchmark. Profiling, controls and search reuse remain isolated opt-in scripts.

Exact commit, expanded configs, model hashes, input lists, repeat/warmup/thread environment and original per-case baselines are in baseline_metadata.json/baseline.jsonl. experiment_provenance.json records actual before/after source/model/dataset/config/environment snapshots for the final diagnostic runs; independent C/D/E variants share a frozen screening-job snapshot window. Profiling uses two traced repeats after one ordinary warmup per image. Screening uses one pass per variant; prefix/fresh-budget evaluations precede pruning/refitting and warm their common runtime paths. Script/source hashes and before/after repository status are included in the artifact manifest. Release source and model remain unchanged. No commit, push or deployment was performed.
'''
    (ROOT / 'PBC_FIRST_PASS_RESULTS.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    baseline_report()
    results_report()
