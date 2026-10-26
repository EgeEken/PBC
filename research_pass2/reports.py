import json
from pathlib import Path
import statistics

from common import ROOT, OUT, PASS1, PRESETS
from analyze import collapse, read


def table(rows, columns):
    def cell(value):
        if value is None:
            return '—'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value)
    return '\n'.join(['| ' + ' | '.join(title for title, key in columns) + ' |',
        '| ' + ' | '.join('---' for _ in columns) + ' |'] +
        ['| ' + ' | '.join(cell(row.get(key)) for title, key in columns) + ' |' for row in rows])


def aggregate_pairs(rows):
    result = {}
    for metric in ('mse', 'bpp', 'final_bytes', 'raw_bytes', 'encode_seconds', 'decode_seconds'):
        if rows and metric + '_baseline' in rows[0]:
            baseline = statistics.mean(r[metric + '_baseline'] for r in rows)
            candidate = statistics.mean(r[metric + '_candidate'] for r in rows)
            result[metric + '_baseline'] = baseline
            result[metric + '_candidate'] = candidate
            result[metric + '_pct'] = 100 * (candidate / baseline - 1)
            result[metric + '_wins'] = sum(r[metric + '_candidate'] < r[metric + '_baseline'] for r in rows)
    return result


def rng_report():
    paired = read('rng_paired')
    rows = []
    for dataset in ('hpt_data', 'hpt_data_val'):
        for preset in PRESETS + ('all',):
            selected = [r for r in paired if r['dataset'] == dataset and (preset == 'all' or r['preset'] == preset)]
            rows.append(dict(dataset=dataset, preset=preset, cases=len(selected), **aggregate_pairs(selected)))
    overall = aggregate_pairs(paired)
    discrepancy = json.loads((OUT / 'hf_discrepancy.json').read_text()) if (OUT / 'hf_discrepancy.json').exists() else {}
    cpu = json.loads((OUT / 'hf_cpu_variant.json').read_text()) if (OUT / 'hf_cpu_variant.json').exists() else {}
    model_hash = json.loads((OUT / 'initial_snapshot.json').read_text())['hashes']['patch_policy.npz']
    observation = discrepancy.get('local_proposal_observation', {})
    tie_analysis = read('hf_tie_analysis')
    profile_rows = [r for r in read('profile_component_summary') if r['experiment'] == 'rng_profile' and r['function'] in ('sample_box', 'read_grid', 'write_grid', 'channel_sum_error', 'final_mse')]
    text = f'''# RNG-controlled speed attribution

The current-root NumPy-RNG variant matches the immutable release on all 100 image/preset cases and all three repeats: complete files, decoded RGB pixels, initialization/iterative patch payloads, counts and order. Every run also matches its frozen Pass 1 release file. This establishes output preservation for these default configurations and this local dependency/CPU environment; it does not extend to arbitrary warmup, overflow or custom settings.

The only experimental algorithm substitution is `pbc3_ops.PBC3Rng = numpy.random.default_rng` in a separate current-root worker. Production RNG/defaults are unchanged. Controls are release `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51` and root `3e9e8c94b7497ecefc5e482a66fde697dfad6e19`. Model SHA-256 is `{model_hash}`.

Across all 25 images/four presets, mean per-case median encode changes {overall.get('encode_seconds_pct', 0):+.2f}% and decode {overall.get('decode_seconds_pct', 0):+.2f}% with identical output. These are combined engineering benefits. They do not individually attribute canvas storage, delta reuse, bit I/O, model caching or tracing overhead, and cannot be subtracted directly from Pass 1's differently timed aggregate to assign an exact RNG share.

{table(rows, [('Set','dataset'),('Preset','preset'),('Cases','cases'),('Release encode s','encode_seconds_baseline'),('Root NumPy encode s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Release decode s','decode_seconds_baseline'),('Root NumPy decode s','decode_seconds_candidate'),('Decode %','decode_seconds_pct')])}

## Timing and component evidence

Three interleaved warmed repeats, persistent source-isolated workers, CPU affinity 0 and numerical thread settings 1. Input loading and independent RGB scoring/parsing are outside timers; complete compress/decompress calls are inside. Warmup rows retain first invocation costs, including any disk-cache loading/compilation. This is not a pristine cold-start test. Per-case mean/median, extrema and CV are retained by the analysis script; source/model/config/dataset hashes and actual module paths are in rng_control_metadata.json and rng_control_provenance.json.

Separate encoder cProfile runs on flower, Monet and googlewiki, all presets, two repeats, compare individual function self/cumulative times and call counts in rng_profile.json. Profiled full timings include profiler overhead and are not primary latency estimates. Nested cumulative times must not be summed. Independent candidate ablations are in PBC_PERFORMANCE_RESULTS.md.

{table(profile_rows, [('Source','variant'),('Function','function'),('Self s/case','self_seconds_per_case'),('Cumulative s/case','cumulative_seconds_per_case'),('Calls/case','calls_per_case')])}

## Bounded HF discrepancy investigation

Retained original-RGBA Quality files were inspected locally. First quantized payload divergence: patch {discrepancy.get('first_differing_patch')}, zero-based, after four initialization patches. Earlier payloads reconstruct identically and the earlier prefix reserializes identically. Channel errors immediately before divergence: {discrepancy.get('channel_errors_before_difference')}. The local maximum anchor block score is shared by {observation.get('top_anchor_score_tie_count')} blocks. Full candidate geometry, RNG state and ranked proposals are retained in hf_discrepancy.json.

{table(discrepancy.get('variants', []), [('Local variant','variant'),('Remote bytes equal','remote_equal'),('Saved local bytes equal','local_equal'),('First divergence from remote','first_difference_from_remote')])}

CPU-dispatch control: `{json.dumps(cpu, sort_keys=True)}`.

Retained-prefix replay: `{json.dumps(tie_analysis, sort_keys=True)}`. The remote rectangle is absent from the locally sampled 200-box pool, although its width/height occur. Local top-20 anchor scores contain ties, including a two-way cutoff tie; the best local proposal score is not tied. With one learned candidate, geometry is chosen before model action inference. Anchor tie ordering/RNG/platform differences are therefore a plausible proposal-stage source. Local CPU dispatch demonstrably changes output, but did not reproduce the remote file; a live causal attribution is not established.

These checks use no uploads, remote compression calls, dependency installation or running-process introspection. Local float64 MLP inference and the release RNG are bounded diagnostic substitutions, not codec candidates. Equal earlier quantized patches constrain the divergence to later selection; they do not prove that every transient proposal/RNG state matched the remote process. The remote process's installed NumPy/BLAS versions, enabled CPU dispatch and loaded module/model state are not observable through the retained artifacts.

The next diagnostic would need the running process's exact dependency/CPU-dispatch and loaded input/model hashes, plus channel, RNG state, top-20 anchors and sampled proposals at iterative step 11 for this saved input. That requires explicitly authorized server introspection or an opt-in diagnostic run; it is not performed here. The subtask stops at this evidence boundary.

Reproduce: `python research_pass2/runner.py rng`; `python research_pass2/analyze_rng.py equivalence`; `python research_pass2/analyze_rng.py profile`; `python research_pass2/analyze_rng.py discrepancy`. For CPU dispatch, start a fresh process with `NPY_DISABLE_CPU_FEATURES=AVX2,FMA3` and run `python research_pass2/analyze_rng.py cpu_variant`. Raw evidence: rng_control.jsonl, rng_equivalence.json, rng_paired.json, rng_profile.json, hf_discrepancy.json and hf_cpu_variant.json.
'''
    (OUT / 'RNG_SPEED_ATTRIBUTION.md').write_text(text, encoding='utf-8')


def performance_report():
    name = 'performance_confirmation' if read('performance_confirmation_paired') else 'performance_screen'
    paired = read(name + '_paired')
    rows = []
    screen_rows = []
    for variant in ('channel_sum', 'grid_numpy', 'skip_final_mse'):
        selected = [r for r in read('performance_screen_paired') if r['candidate'] == variant]
        screen_rows.append(dict(candidate=variant, cases=len(selected), **aggregate_pairs(selected)))
    decoder_rows = [r for r in read('decoder_component_summary') if r['phase'] in ('decode.patch.read', 'decode.reconstruct.final_resize', 'decode.patch.resample_apply')]
    hot_rows = [r for r in read('profile_component_summary') if r['experiment'] == 'performance_components' and r['function'] in ('write_grid', 'channel_sum_error', 'channel_sum')]
    variability = [{**r, 'encode_cv_pct': 100 * r['encode_seconds_mean_cv'], 'decode_cv_pct': 100 * r['decode_seconds_mean_cv']} for r in read('performance_confirmation_summary') if r['preset'] == 'all']
    dimensions = {(r['dataset'], r['image']): r['width'] * r['height'] for r in read('performance_confirmation')}
    size_rows = []
    for candidate in sorted({r['candidate'] for r in paired}):
        for size in ('<1 MP', '>=1 MP'):
            group = [r for r in paired if r['candidate'] == candidate and (dimensions.get((r['dataset'], r['image']), 0) < 1000000) == (size == '<1 MP')]
            if group:
                size_rows.append(dict(candidate=candidate, original_size=size, cases=len(group), **aggregate_pairs(group)))
    for candidate in sorted({r['candidate'] for r in paired}):
        for dataset in ('hpt_data', 'hpt_data_val'):
            group = [r for r in paired if r['candidate'] == candidate and r['dataset'] == dataset]
            if group:
                rows.append(dict(candidate=candidate, dataset=dataset, cases=len(group),
                    equal=sum(r['bytes_equal'] and r['pixels_equal'] and r['patches_equal'] for r in group), **aggregate_pairs(group)))
    text = f'''# Bit-exact performance experiments

All candidates are isolated in research_pass2; production source/config defaults are unchanged. Each candidate is compared with the same current-root RNG and implementation, disabled versus enabled. No representation/search/model changes are combined with these timings.

- `channel_sum`: narrow Numba integer reduction for the existing full-channel visible absolute error. This avoids temporary clip/difference/absolute arrays and preserves exact int64 sums. It does not implement an incrementally maintained residual map or repeat the rejected integral-image fusion. Local clipped-region updates were deferred because hooking mutation/warmup state would be a broader and riskier change than this allocation-free reduction.
- `grid_numpy`: MSB-first vectorized NumPy bit packing/unpacking, preserving every bit alignment and reader/writer suffix state. It introduces temporary bit arrays proportional to grid cells × bitcount, rather than a native rewrite. Production version-0 syntax and generated palette semantics are unchanged.
- `skip_final_mse`: existing `compute_final_mse=False` configuration. The preview is still constructed and resized, bytes/pixels stay identical, and the result's internal MSE is None. The harness independently decodes and scores. Default True and website behavior remain intact. This is an API/runtime tradeoff, not an RD improvement or newly implemented feature.

## Complete-codec results

Six-image screen, all four presets, three repeats:

{table(screen_rows, [('Variant','candidate'),('Cases','cases'),('Encode %','encode_seconds_pct'),('Encode wins','encode_seconds_wins'),('Decode %','decode_seconds_pct'),('Decode wins','decode_seconds_wins')])}

Primary confirmation source: {name}. Values average per-image/preset medians over three warmed interleaved single-core repeats. Equality checks include complete bytes, decoded RGB pixels, patch order/count, bpp and independently scored MSE. No production randomness is replaced for D.

{table(rows, [('Variant','candidate'),('Set','dataset'),('Cases','cases'),('Exact','equal'),('Baseline enc s','encode_seconds_baseline'),('Variant enc s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Baseline dec s','decode_seconds_baseline'),('Variant dec s','decode_seconds_candidate'),('Decode %','decode_seconds_pct')])}

All screening results, including candidates not promoted, remain in performance_screen.jsonl/_summary.json/_paired.json. Fixed promotion decisions predate confirmation and holdout in performance_promotion.json. Per-image wins, outliers, repeated-call CV, means and extrema are retained in machine-readable tables, rather than hiding small-image regressions in the aggregate. Small percent changes near observed repeat noise are inconclusive.

Actual-call means and mean per-case repeat CV:

{table(variability, [('Set','dataset'),('Variant','label'),('Encode mean s','encode_seconds_mean_of_calls'),('Encode CV %','encode_cv_pct'),('Decode mean s','decode_seconds_mean_of_calls'),('Decode CV %','decode_cv_pct')])}

By original image size, pooling main/holdout cases only for this descriptive size comparison:

{table(size_rows, [('Variant','candidate'),('Original size','original_size'),('Cases','cases'),('Encode %','encode_seconds_pct'),('Decode %','decode_seconds_pct')])}

The MSE option does not change the decoder; its small measured decode differences are timing variation, not an attributed decode improvement. Grid-I/O decoder gains are independently measured on identical files and all 100 cases. Size grouping is descriptive and confounds content type; it is not a controlled causal image-size experiment.

Separate component profiles use three images, balanced, two repeats after warmup: performance_components.json. Timers from those profiled calls include instrumentation overhead; micro/function improvements alone are not a full-codec speed claim. RSS-after fields reflect cumulative process allocators and rotation, not isolated peak-memory deltas. Grid temporaries are O(grid cells × bitcount) bytes (plus the uint16 output and packing temporaries); channel reduction removes O(working pixels) temporary arrays. No unsupported peak-memory claim is made.

{table(hot_rows, [('Variant','variant'),('Function','function'),('Profile self s/case','self_seconds_per_case'),('Profile cumulative s/case','cumulative_seconds_per_case'),('Calls/case','calls_per_case')])}

Decoder phase traces, three images/four presets/two repeats, separate from untraced latency:

{table(decoder_rows, [('Variant','method'),('Phase','phase'),('Mean inclusive s','inclusive_seconds'),('Mean exclusive s','exclusive_seconds')])}

## Correctness and limitations

optimization_unit_checks.json tests grid widths 0–9 at all eight initial alignments, empty/short/long grids and suffix-state preservation. The channel kernel additionally tests signed-int16 extrema, clipping, odd dimensions and RGB/RGBA. Production encodes provide the independent meaningful correctness gate across real image trajectories. Kernels are warmed before timings; first calls/compilation are retained separately in metadata. No native parser, percentile replacement, approximate channel accounting, combined speed variant or HF deployment is tested.

Local improvements need a later explicitly authorized cpu-basic experiment before deployment claims: run disabled/enabled versions in the same warmed single-thread process on fixed already-approved server-side inputs, interleave repeats, and check byte/pixel hashes and module/dependency identity. No upload or deployment is authorized by this report.

Reproduce: `python research_pass2/performance.py correctness`, `screen`, `confirm`, and `components`, sequentially. Confirmation reads the frozen promotion decision. Source: optimizations.py; raw rows and before/after snapshots are stored beside each experiment.
'''
    (OUT / 'PBC_PERFORMANCE_RESULTS.md').write_text(text, encoding='utf-8')


def quality_report():
    screen = [r for r in read('resolution_screen_summary') if r['preset'] != 'all']
    allocation = [r for r in read('allocation_screen_summary') if r['preset'] != 'all']
    confirmed = [r for r in read('quality_confirmation_summary') if r['preset'] != 'all']
    matches = [r for r in read('rd_matched_summary') if r['baseline_source'] == 'release' and r['experiment'] == 'quality_confirmation' and r['label'].startswith('q500')]
    matched_outliers = sorted([r for r in read('rd_matched_comparisons') if r['experiment'] == 'quality_confirmation' and r['baseline_source'] == 'release' and r['label'].startswith('q500') and r['matched_rate_mse_pct'] is not None], key=lambda r: r['matched_rate_mse_pct'], reverse=True)[:8]
    at_rates = [r for r in read('curves_at_control_rates') if r['limit'] == 500000 and r['control_source'] == 'release' and r['control_preset'] == 'high_quality' and r['image'] in ('cat.jpg', 'cathedral.png', 'flower.jpg', 'googlewiki.png', 'monet.jpg', 'stockpoint.jpg')]
    ledger = read('quality_patch_ledgers')
    quality_pairs = read('quality_confirmation_paired')
    outliers = sorted([r for r in quality_pairs if r['control_source'] == 'release' and r['control_preset'] == 'high_quality'], key=lambda r: r['mse_pct'], reverse=True)[:8]
    trajectory_rows = read('quality_trajectories')
    trajectory_summary = []
    for source in ('release', 'current'):
        for preset in ('quality', 'high_quality'):
            increments = [r for r in trajectory_rows if r['source'] == source and r['preset'] == preset and 'marginal_sse_reduction' in r]
            if increments:
                trajectory_summary.append(dict(source=source, preset=preset, increments=len(increments),
                    rgb_mse_increases=sum(r['marginal_sse_reduction'] < 0 for r in increments),
                    nonpositive_added_file_bits=sum(r['marginal_file_bits'] <= 0 for r in increments)))
    representation = []
    for source in ('release', 'current'):
        for preset in ('quality', 'high_quality'):
            patches = [r for r in ledger if r['source'] == source and r['preset'] == preset and not r['initialization']]
            if patches:
                representation.append(dict(source=source, preset=preset, patches=len(patches),
                    cells=statistics.mean(r['grid_cells'] for r in patches), bits=statistics.mean(r['bitcount'] for r in patches),
                    cell=statistics.mean(r['cell_size'] for r in patches), raw=statistics.mean(r['total_bits'] / 8 for r in patches),
                    overlap=statistics.mean(r['previous_iterative_coverage_fraction'] for r in patches)))
    text = f'''# Single-stage quality scaling

The experiment changes only opt-in single-stage configurations under research_pass2. The codec's v0 representation, decoder, source inputs, seed, RGB benchmark convention and model stay fixed. No serialized multistage, coarse initialization, pruning, refitting, alternate search architecture or training is reintroduced.

## Fixed screen and measured points

The six images were selected before measurement in PLAN.md: cat (large photograph/fine fur), Monet (texture/art), flower (large smooth/bokeh photograph), cathedral (architecture/edges), googlewiki (text/synthetic), stockpoint (relatively low 250k round-trip loss). Both release and current root were measured at 250k/500k/1M, Quality/High Quality, two interleaved warmed repeats. Times are means of per-image medians, not HF timings.

{table(screen, [('Source/limit','label'),('Preset','preset'),('RGB MSE','mse'),('bpp','bpp'),('Raw bytes','raw_bytes'),('Final bytes','final_bytes'),('Iterative patches','iterative_patches'),('Encode s','encode_seconds'),('Decode s','decode_seconds')])}

These six-image means are descriptive operating points, not matched-rate evidence. Source dimensions, actual working dimensions, expanded configs and per-image results are retained in resolution_screen.jsonl. Image limits are caps: a small source can stop being resized before the 1M limit.

## Preset diagnosis

Source inspection confirms only three preset differences between Quality and High Quality: requested iterative patches 50 versus 20, search_q_start 0.5 versus 0.7, learned_filler_q 0.8 versus 0.95. Both have 250k default working pixels, identical automatic initialization, cell-size/action support, allowed palette limits, mask size and decoder syntax. The learned model chooses actual cell size and bitcount; the heuristic patch_palette_bitcount is not a cap on that learned action path. Raising learned q changes features/actions, not merely a quantizer knob. Early stopping occurs when the selected learned proposal has no positive exact reduction; it is not an exhaustive search saturation proof.

On the 20-image immutable-release controls, Quality to HQ changes mean RGB MSE 160.3306 -> 143.1334 (-10.73%) while bpp rises 0.2996 -> 0.5425 (about +81%). Pass 1's 250k round-trip mean is 128.3761, close to the HQ MSE but not a mathematical floor or an additive error decomposition.

Per-iterative-patch representation on the six default-resolution trajectories:

{table(representation, [('Source','source'),('Preset','preset'),('Patches','patches'),('Mean cells','cells'),('Mean bitcount','bits'),('Mean cell size','cell'),('Raw bytes/patch','raw'),('Previous iterative overlap','overlap')])}

These are patch-weighted descriptive means, not independently causal allocations. Initialization is excluded here and kept in all final file sizes. Palette masks/bounds, grid cost, area, coverage and per-prefix final bytes are in quality_patch_ledgers.json and quality_trajectories.json. Prefixes are replayed from frozen completed streams; increasing a fresh budget changes search interpolation and the model's step fraction, so a prefix is not a smaller-budget encoding policy. Marginal RGB SSE reduction is measured after actual serialization/LZMA/decode. Added final bits are the difference between complete prefix files; they can be zero/negative and are context-dependent, unlike per-patch raw bits.

{table(trajectory_summary, [('Source','source'),('Preset','preset'),('Measured increments','increments'),('RGB MSE increases','rgb_mse_increases'),('Nonpositive added final bits','nonpositive_added_file_bits')])}

Three isolated targeted hypotheses were screened: High Quality budget 50; Quality learned q 0.95; High Quality search_q_start 0.5. These are the complete allocation ablations, rather than a Cartesian parameter sweep.

{table(allocation, [('Variant','label'),('MSE','mse'),('bpp','bpp'),('Patches','iterative_patches'),('Encode s','encode_seconds'),('Decode s','decode_seconds')])}

Four strategically selected rate configurations followed: 500k Quality budgets 20/35, 1M Quality budget 20, and 250k Quality learned q 0.9, with fresh root Q/HQ controls. Two repeats and all six images. Their measured points and expanded configurations are in rate_screen_plan.json and rate_screen.jsonl. No interpolation is performed outside the observed range.

## Matched-rate comparisons and confirmation

The two fixed 500k Quality budgets (20 and 35) were selected before new holdout candidate runs. They test a class-dependent lead: p35 improves both rate and quality versus release HQ on flower/googlewiki in the screen, while textured Monet and low-round-trip-loss stockpoint lose efficiency. q1000_p20 and the preset-only ablations were not promoted. These choices are documented in quality_confirmation_plan.json; they are research confirmation, not production acceptance.

Full main/holdout absolute results, three repeats, same fresh release/root Q/HQ controls:

{table(confirmed, [('Set','dataset'),('Variant','label'),('Preset','preset'),('Cases','cases'),('MSE','mse'),('bpp','bpp'),('Patches','iterative_patches'),('Encode s','encode_seconds'),('Decode s','decode_seconds')])}

Candidate-point interpolation against each image's nondominated release default-preset controls:

{table(matches, [('Set','dataset'),('Variant','label'),('Cases','cases'),('Rate overlap','rate_overlap_cases'),('Mean matched-rate MSE %','mean_matched_rate_mse_pct'),('Rate wins','matched_rate_wins'),('MSE overlap','mse_overlap_cases'),('Mean matched-MSE bpp %','mean_matched_mse_bpp_pct'),('Observed dominated','observed_control_dominated')])}

The mean percentage is over valid overlapping images, not over the entire dataset. A missing overlap is not a win or loss. Piecewise interpolation uses actual neighboring measured nondominated points in arithmetic bpp/MSE coordinates; it is an estimate, not another achievable single-file operating point. Sparse default controls can span wide rate intervals. Nearest actually measured rate distances, exact brackets/weights and per-image outliers are retained in rd_matched_comparisons.json. A close-rate flag means within 5%, not exact equality. No BD-rate/extrapolation or continuous dominance claim is made.

Restricted to intervals whose measured neighbors are specifically release Quality and High Quality (rather than Compression/Balanced):

{table(matches, [('Set','dataset'),('Variant','label'),('Q–HQ overlapping images','q_hq_overlap_cases'),('Matched-rate MSE %','q_hq_mean_matched_rate_mse_pct'),('Wins','q_hq_matched_rate_wins')])}

Largest matched-rate increases versus the observed release frontier:

{table(matched_outliers, [('Set','dataset'),('Image','image'),('Variant','label'),('MSE %','matched_rate_mse_pct'),('Nearest actual rate distance %','nearest_rate_distance_pct')])}

Matched-rate and matched-MSE tables can use different image subsets/ranges; their aggregate signs are not interchangeable. No joint dominance is inferred by combining those two means.

Largest pointwise MSE increases versus fresh release HQ (these are operating-point comparisons, not matched-rate regressions):

{table(outliers, [('Set','dataset'),('Image','image'),('Variant','label'),('MSE %','mse_pct'),('bpp %','bpp_pct'),('Encode %','encode_seconds_pct'),('Decode %','decode_seconds_pct')])}

quality_confirmation_paired.json retains every pointwise comparison, strict RD image wins and actual runtime changes. budget_trajectory_comparisons.json explicitly checks initialization and the first changed iterative patch between fresh p20/p35 encodes, rather than treating the two budgets as nested trajectories.

For the six screening images, the 500k family interpolated at release HQ rate:

{table(at_rates, [('Image','image'),('Release HQ bpp','control_bpp'),('Release HQ MSE','control_mse'),('MSE at HQ rate %','mse_at_control_rate_pct'),('Rate at HQ MSE %','rate_at_control_mse_pct')])}

observed_rd_frontiers.json retains the best measured operating points across available configurations, per image. This adaptive observed envelope is not a learned selector or a fixed general-purpose preset. curves_at_control_rates.json compares both release and root Q/HQ targets with measured 500k/1M families. quality_rd_screen.png/.svg and plot_rd.py show raw observed points and connecting guides; the numerical interpolation/brackets in JSON are authoritative. Main and holdout remain separately summarized.

## Interpretation limits

The round-trip diagnostics and actual larger-resolution MSE gains establish that the 250k canvas is a material spatial-detail limitation on several large/detail-rich images. They do not establish a universal distortion bound. Extra pixels alone do not guarantee a better RD frontier: texture and already smooth/low-loss inputs can consume more grid bits and search time without a matched-rate benefit. At 1M, the fixed-size local boxes also cover a smaller fraction of the working canvas, and requested budgets can fill completely. This is consistent with coverage/representation pressure; it does not by itself isolate the model, geometry, or search as the sole cause.

High Quality's inefficiency is a measured combination of higher cost per chosen representation, fewer requested corrections, and spatial-detail limitations. The one-proposal learned stop rule and greedy trajectory remain unresolved bottlenecks. More patches or a higher learned q alone did not establish a general curve improvement in the screen. Positive 500k image-specific RD results show that the additive single-stage architecture is not proven fundamentally saturated. A distinct next architecture test would need cost-aware spatial support/action selection or an explicit high-frequency residual representation, with matched-rate controls, rather than another fixed coarse-to-fine stream schedule or more RL.

All raw points, trajectories, per-image comparisons and provenance are retained. Reproduce resolution via runner.py resolution; allocation via quality.py ablations; rate/confirmation via quality.py planned rate_screen / quality_confirmation; ledgers via quality.py trajectories; then analyze.py and plot_rd.py. These commands overwrite only their own Pass 2 output names; they never alter Pass 1.
'''
    (OUT / 'PBC_QUALITY_SCALING.md').write_text(text, encoding='utf-8')


def entropy_report():
    accounting = read('bitstream_accounting')
    components = []
    if accounting:
        total_raw = sum(r['raw_bytes'] for r in accounting) * 8
        total_final = sum(r['final_bytes'] for r in accounting)
        for key in accounting[0]['syntax_bits']:
            bits = sum(r['syntax_bits'][key] for r in accounting)
            effect = sum(r['zeroing_compressed_byte_effect_nonadditive'][key] for r in accounting)
            components.append(dict(component=key, bits=bits, raw_share_pct=100 * bits / total_raw,
                zeroing_effect_pct=100 * effect / total_final))
        components += [dict(component='wrapper', bits=48 * len(accounting), raw_share_pct=100 * 48 * len(accounting) / total_raw),
            dict(component='final_padding', bits=sum(r['padding_bits'] for r in accounting), raw_share_pct=100 * sum(r['padding_bits'] for r in accounting) / total_raw)]
    summaries = read('entropy_summary')
    reps = [r for r in summaries if r['source'] == 'current' and r['kind'] == 'representation' and r['phase'] == 'screen']
    lzmas = [r for r in summaries if r['source'] == 'current' and r['kind'] == 'lzma' and r['phase'] == 'screen']
    memory = []
    for path in sorted(OUT.glob('memory_*.json')):
        r = json.loads(path.read_text())
        if 'dictionary_bytes' not in r:
            continue
        memory.append(dict(method=path.stem.removeprefix('memory_'), dictionary=r['dictionary_bytes'],
            constructed_delta=r['rss_constructed'] - r['rss_before'], pack_delta=r['rss_after_pack'] - r['rss_before']))
    text = f'''# Frozen-trajectory entropy research

No screened lossless representation or LZMA setting establishes a general final-size improvement. Both new grid transforms increase every source/preset aggregate after LZMA. They are rejected as general coding candidates; no full-main or holdout entropy candidate was promoted. This is a bounded six-image/four-preset screen for both immutable release and current root, not proof that every entropy representation is exhausted.

## Bitstream accounting

All 48 input streams are frozen Pass 1 files; searches are not rerun. Exact accounting includes the six-byte wrapper, image header, patch coordinates/channel, generated-palette flag, mask length and mask, signed bounds, maximum bitcount, 16-bit cell size, quantized indices, and final 0–7 padding bits. Grid dimensions are implied by geometry/cell size, not separately serialized. Parsing asserts the sum equals the complete raw bitstream length.

{table(components, [('Component','component'),('Total raw bits','bits'),('Raw share %','raw_share_pct'),('Zeroing effect on final bytes %','zeroing_effect_pct')])}

The compressed column is a nonadditive diagnostic: replace only that component's bits with zeros at their original positions and recompress. These deliberately invalid streams are never decoded. Context/match changes make these effects nonadditive and occasionally negative; they are not a unique allocation of compressed bytes to fields. A large raw component alone is not evidence that a replacement will save final bytes. Exact raw and post-LZMA sizes, per-patch symbol counts by channel/preset, zero indices versus zero palette values, horizontal/vertical correlation and repeat fractions are retained in bitstream_accounting.json, symbol_distributions.json and index_statistics.json.

Unweighted patch means in C1 were approximately 55.0% zero indices, 58.1% horizontal repeats, 59.0% vertical repeats, and correlations 0.390/0.402. These statistics motivated two independent experiments; they did not predict a guaranteed compressed-size gain.

## Exact lossless representations

- `horizontal_delta`: leave the first index of each row unchanged; subsequent symbols are current minus left, modulo 2^bitcount. Decode with row-wise cumulative sums modulo the same alphabet.
- `column_order`: serialize a column-major traversal in the same fixed-width symbol slots; reverse that permutation before palette lookup.

Each changes only grid symbol representation, keeps header/palette/geometry semantics intact and has no side information beyond existing dimensions/bitcount. Research-only magic `P2D1` / `P2T1` identifies the alternate format (layout tag 0, existing entropy byte and body layout); the six-byte outer framing is included in all sizes. AlternateCodec in entropy.py provides the matching decoder. Production v0 decoding remains intact and explicitly rejects both research magics. Every trial recovers exact quantized indices and exact decoded RGB pixels. No combined predictor/traversal/RLE was tested.

Current-root stream results (release results also retained, with the same aggregate direction):

{table(reps, [('Preset','preset'),('Method','method'),('Bytes','final_bytes'),('bpp','bpp'),('Bytes %','byte_change_pct'),('Image wins / 6','byte_wins'),('Repack s','representation_pack_seconds'),('Decode s','decode_seconds')])}

Each point retains unchanged original decoded MSE in entropy_screen.json. Repacking time includes serialization, optional transform and entropy packing of frozen patches; it excludes original search. Full decode is independently timed with the matching decoder. This is not a measured complete new encoder latency. Original full encode/decode and expanded configuration are linked in each row. No full-codec speed claim is inferred by adding differently timed historical search to repack time. The new inverse adds cumsum/permutation work and temporary arrays; it has no learned dependency.

## Bounded LZMA2 sweep

Production is raw LZMA2, `preset=PRESET_EXTREME`, meaning preset 0 plus extreme, with a 256 KiB dictionary. Independent alternatives are preset 1 (1 MiB), preset 6 (8 MiB), and production settings with explicit 64 KiB dictionary. Dictionary sizes are verified from the installed liblzma filter-property encoding in the memory probes. Preset 1 reduces entropy-pack latency but costs rate. Preset 6 is near neutral/slightly larger in aggregate and can emit references beyond the production decoder's dictionary. The 64 KiB option is a memory/rate tradeoff with negligible or adverse size changes.

{table(lzmas, [('Preset','preset'),('LZMA method','method'),('Bytes','final_bytes'),('bpp','bpp'),('Bytes %','byte_change_pct'),('Pack s','entropy_pack_seconds'),('Unpack s','entropy_unpack_seconds'),('Full decode s','decode_seconds'),('Existing decoder compatible','production_decoder_compatible')])}

All entropy round trips reproduce the exact raw body, and matching-filter complete decodes reproduce exact pixels. Existing-decoder compatibility is tested separately on every case. An incompatible row's six-byte wrapper size is a diagnostic configuration comparison, not a deployable v0 file: no dictionary metadata extension was designed or accepted, and an actual new self-describing format would need its full overhead counted. Incompatible variants are rejected for existing-wrapper use. The exact production control reproduces original compressed bytes. LZMA STORE fallback remains active when compression is not smaller.

Separate process memory probes, 4 MiB repeated frozen raw body for allocation visibility:

{table(memory, [('Filter','method'),('Dictionary bytes','dictionary'),('RSS construct delta','constructed_delta'),('RSS after pack delta','pack_delta')])}

RSS is observed allocation behavior for this synthetic memory probe, not an exact liblzma memory bound or an image RD result. Smaller encoder dictionaries do not automatically reduce RAM in the unchanged production decoder, which still uses its configured dictionary. All settings are warmed and rotated, three repeats per frozen case. Entropy-pack and entropy-unpack timings remain separate from whole decode and historical search.

The memory fixture artificially repeats a complete raw body across 4 MiB; large dictionaries can exploit those artificial long-distance repeats. Its packed size is not evidence of a single-image bitrate improvement.

## Extreme-compression decision

Neither transform retains savings at Compression or Balanced; both raise final bitrate at unchanged MSE. Obvious symbol repetition is already exploited effectively by the existing byte stream plus LZMA, and these two transforms disrupt useful patterns enough to lose overall. This last mechanism is a plausible explanation of the measurements, not a proof of optimal entropy coding. The bounded LZMA sweep gives rate-for-speed/memory tradeoffs, not the hoped-for fixed-quality size gain. Header tightening was already a small historical lead and was not repeated here.

No encoder rate reduction from B is compared with raw entropy savings: every rate comparison uses complete final bytes divided by original pixels. No C technique is combined with a D optimization or enabled in production. Reproduce `python research_pass2/entropy.py accounting`, then `screen`. Decision and all reasons are frozen in entropy_promotion.json. The optional `main`/`holdout` commands require a nonempty deliberate promotion; neither was run because no size candidate passed. Memory probes use `memory --filter NAME` in separate processes.
'''
    (OUT / 'PBC_ENTROPY_RESEARCH.md').write_text(text, encoding='utf-8')


def summary_report():
    rng = aggregate_pairs(read('rng_paired'))
    engineering = []
    for dataset in ('hpt_data', 'hpt_data_val'):
        for candidate in ('grid_numpy', 'skip_final_mse'):
            selected = [r for r in read('performance_confirmation_paired') if r['dataset'] == dataset and r['candidate'] == candidate]
            engineering.append(dict(dataset=dataset, candidate=candidate, cases=len(selected), **aggregate_pairs(selected)))
    quality = []
    for dataset in ('hpt_data', 'hpt_data_val'):
        for candidate in ('q500_p20', 'q500_p35'):
            selected = [r for r in read('quality_confirmation_paired') if r['dataset'] == dataset and r['label'] == candidate and r['control_source'] == 'release' and r['control_preset'] == 'high_quality']
            quality.append(dict(dataset=dataset, candidate=candidate, cases=len(selected), strict_rd_wins=sum(r['strict_rd_win'] for r in selected), **aggregate_pairs(selected)))
    overlap = [r for r in read('rd_matched_summary') if r['experiment'] == 'quality_confirmation' and r['baseline_source'] == 'release' and r['label'].startswith('q500')]
    root_entropy = [r for r in read('entropy_summary') if r['source'] == 'current' and r['kind'] == 'representation' and r['method'] != 'control']
    initial = json.loads((OUT / 'initial_snapshot.json').read_text())
    dataset_hashes = {k: v for k, v in initial['hashes'].items() if k.startswith(('hpt_data\\', 'hpt_data_val\\'))}
    counts = {name: len(read(name)) for name in ('rng_control', 'resolution_screen', 'allocation_screen', 'rate_screen', 'quality_confirmation', 'performance_screen', 'performance_confirmation', 'rng_profile', 'performance_components', 'entropy_screen')}
    text = f'''# PBC Pass 2 research summary

All four workstreams completed locally. The pass establishes matched-work engineering speedups, screens the single-stage quality/resolution frontier, rejects two tested lossless entropy transformations, and confirms fixed runtime candidates on main/holdout. It does not authorize or enable a production PBC 3.1 configuration.

## What ran

- A: 25 images × four presets × release/current-with-NumPy-RNG × three repeats (600 rows), plus separate component profiles and a bounded saved-HF-file investigation.
- B: six fixed main images, release/root, Quality/HQ, 250k/500k/1M, two repeats (144 rows); three targeted allocation ablations (36); four targeted rate settings plus fresh Q/HQ controls (72). Two fixed 500k Quality budget variants and fresh release/root Q/HQ controls were then confirmed on all 20 main and five holdout images, three repeats (450).
- C: 48 frozen release/root streams (six images × four presets × two sources), exact syntax/symbol diagnostics, two independent alternate representations, four LZMA controls/settings, three repeats (1,008 rows). Four separate dictionary-memory probes. No entropy candidate passed the screen, so none was promoted to main/holdout.
- D: three candidates on six images/four presets/three repeats (288); two fixed candidates plus current-root control on all 25 images/four presets/three repeats (900). Separate encoder/decoder component measurements, 640 bit-layout checks and integer clipping/RGBA tests. No combined variant.

Actual row counts: `{json.dumps(counts, sort_keys=True)}`. Dataset SHA-256 inventory contains {len(dataset_hashes)} inputs. All expanded configs, source/model/input hashes, module paths, dimensions, actual patch counts, raw/final bytes, decoded RGB MSE and complete compress/decompress wall times are retained. See PLAN.md, README.md, initial_snapshot.json, each *_metadata.json and *_provenance.json.

All primary timings are local, warmed, single-core and single numerical thread; source/variant order rotates. Three-repeat timings are aggregated by per-case medians, with actual-call means/CV/extrema available. First-invocation warmup calls retain any cache/compilation cost; process-import startup and pristine cold starts are not measured. No HF timings are pooled with local results. `bpp = 8 × complete file bytes / original pixels`; RGB scoring uses actual decoded original-size files. The website's RGBA convention is not used for research distortion.

## Measured engineering results

With release-compatible RNG and exactly identical outputs, current root retains encode {rng.get('encode_seconds_pct', 0):+.2f}% and decode {rng.get('decode_seconds_pct', 0):+.2f}% overall. All 100 distinct cases match on all three repeats. This is the combined existing implementation benefit, not attribution to one change.

Independent D confirmation against unchanged current root:

{table(engineering, [('Set','dataset'),('Variant','candidate'),('Cases','cases'),('MSE','mse_candidate'),('bpp','bpp_candidate'),('Encode base s','encode_seconds_baseline'),('Encode variant s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Decode base s','decode_seconds_baseline'),('Decode variant s','decode_seconds_candidate'),('Decode %','decode_seconds_pct')])}

All compared files, pixels, patches, MSE and bpp are identical. `grid_numpy` is a new isolated implementation; `skip_final_mse` exercises an existing API option and returns internal MSE=None while independent artifact scoring remains active. The channel-reduction screen was exact but its small/noisy latency gain did not justify full promotion. See PBC_PERFORMANCE_RESULTS.md for per-image wins, repeat noise, targeted profiles and memory limits.

## Quality and RD evidence

Fixed candidate points versus fresh release HQ (different configurations, so this table alone is not a matched-rate claim):

{table(quality, [('Set','dataset'),('Candidate','candidate'),('Cases','cases'),('MSE','mse_candidate'),('bpp','bpp_candidate'),('MSE %','mse_pct'),('bpp %','bpp_pct'),('Encode %','encode_seconds_pct'),('Decode %','decode_seconds_pct'),('Strict RD image wins','strict_rd_wins')])}

Candidate-point comparisons inside the measured release frontier's rate range:

{table(overlap, [('Set','dataset'),('Candidate','label'),('Rate-overlap images','rate_overlap_cases'),('Mean interpolated MSE %','mean_matched_rate_mse_pct'),('Wins within overlap','matched_rate_wins'),('MSE-overlap images','mse_overlap_cases'),('Mean interpolated bpp %','mean_matched_mse_bpp_pct'),('Dominated measured points','observed_control_dominated')])}

Overlap-only means are not dataset-wide guarantees. rd_matched_comparisons.json retains every bracket/weight, nearest measured rate and outlier; no extrapolation occurs. The six screening images have denser 500k curves than the other confirmation images (which have two fixed points), so any curve-envelope aggregation must retain its coverage. PBC_QUALITY_SCALING.md gives absolute controls, preset/action costs, per-prefix marginal rates, catastrophic pointwise outcomes and class-specific interpretation. quality_rd_screen.png/.svg is backed by rd_raw_points.json/.csv and plot_rd.py.

Specifically inside measured Quality-to-High-Quality neighbor brackets:

{table(overlap, [('Set','dataset'),('Candidate','label'),('Q–HQ overlap','q_hq_overlap_cases'),('Interpolated MSE %','q_hq_mean_matched_rate_mse_pct'),('Wins','q_hq_matched_rate_wins')])}

For p35, the main-set signal is stronger than the holdout Q–HQ signal; pigeon.JPG remains a +35.47% matched-rate MSE outlier. p20 has a +84.31% matched-rate MSE outlier on mario64.png and +61.80% on pigeon.JPG. These are not hidden by mean bitrate savings. Interpolation against all four default presets and interpolation restricted to Q–HQ cover different image/rate subsets. Matched-MSE means also use a different valid subset, so their signs must not be combined into a joint-dominance claim.

## Entropy outcome

Current-root post-LZMA changes at exactly fixed decoded quality:

{table(root_entropy, [('Preset','preset'),('Transform','method'),('Bytes %','byte_change_pct'),('bpp','bpp'),('Image wins / 6','byte_wins')])}

Neither transform improves any preset aggregate. Preset 1 trades bitrate for entropy-pack speed; preset 6 is near neutral/slightly larger and can exceed the existing decoder dictionary; 64 KiB is a memory tradeoff. No negative entropy result is hidden by promoting an isolated image win. See PBC_ENTROPY_RESEARCH.md for exact raw syntax, nonadditive compressed-component diagnostics, complete decode checks, overhead and dictionary-memory observations.

## Answers to the research questions

1. Larger working resolution reduces actual decoded MSE, but does not universally improve RD. Some large/detail-rich and text/synthetic images gain; texture and already-low-round-trip-loss inputs can spend more bits/time without a matched-rate gain. The full per-image measured controls above determine the extent, rather than preprocessing error alone.
2. High Quality changes the model's quality input, starting proposal weighting and budget. It requests 20 corrections instead of 50 and chooses far costlier grids/precision: on the six release trajectories, about 3.5× grid cells and 4× raw bytes per iterative patch. The allowed syntax/actions are the same. The 20-image release Q->HQ increase is about +81% bpp for -10.73% MSE.
3. The evidence supports a combination of working detail, costly representation choices and allocation/stop behavior. One-proposal early termination is not proof that no useful patch exists. None of the three isolated preset-only ablations establishes a general frontier improvement.
4. Tested 500k single-stage points improve the observed RD frontier on some images, including strict pointwise wins. Matched-rate means and coverage above show where this generalizes and where it fails. No universal continuous frontier or release improvement is claimed from sparse interpolation.
5. The answer is class-dependent: cat/flower/cathedral/googlewiki expose detail-related gains, while Monet and stockpoint show efficiency failures in the screen. A fixed content-independent resolution increase is not justified; confirmation outliers and holdout results remain explicit.
6. Quantized grids dominate raw size and the component-zeroing compressed sensitivity. Metadata is small/repetitive, and compressed byte ownership is not uniquely additive. Full accounting includes framing/alignment rather than estimating patch cost from grid cells alone.
7. Neither tested lossless transform produces a meaningful general post-LZMA saving. Both preserve exact semantics but lose final size. The bounded filter sweep also finds no general fixed-quality size win.
8. The negative result includes Compression and Balanced; presumed zero/repeat redundancy does not turn into retained low-bpp savings for these methods.
9. About {abs(rng.get('encode_seconds_pct', 0)):.1f}% encode and {abs(rng.get('decode_seconds_pct', 0)):.1f}% decode benefit remains when RNG forces exactly matched work. Individual existing optimization shares were not separately ablated.
10. Vectorized grid I/O preserves streams and reduces full decoding latency; optional internal MSE skipping preserves artifacts and reduces encoding work for callers that accept no internal metric. Exact channel reduction is a smaller, inconclusive screen. Individual/full measurements are separate.
11. Bit parsing/packing and deterministic palette/index reconstruction are viable standalone/native building blocks. Remaining PIL bicubic resizing is already native and needs exact semantics preserved in a port. A generic language rewrite, repeated integral fusion or expensive sorting/feature replacement without full-codec evidence is a poor immediate candidate.
12. Ranked follow-up: first validate the isolated grid-I/O improvement and metric-free API path on cpu-basic in a later authorized controlled experiment (low implementation risk; deployment benefit unverified). Next pursue quality representation/allocation that can exploit spatial detail at a fixed final rate, using the fixed 500k image-specific leads and explicit texture/smooth controls (higher RD potential and model/search interaction risk). Then investigate proposal-state reuse/search efficiency with deterministic tie handling and byte-equivalence controls (moderate implementation risk). More generic entropy permutations and a broad native rewrite are lower priority; a fundamentally new PBC 4.0 representation is not established as necessary by this pass.

## Unresolved and repository state

The saved HF Quality discrepancy remains unresolved. Earlier payloads agree; the remote rectangle is absent from the locally replayed proposal pool. There are anchor ties and disabling local NumPy CPU features changes the trajectory, but no tested local variant reproduces the remote file. The checked one-candidate path chooses geometry before model action inference. Further resolution requires authorized running-process/version/CPU/RNG/anchor introspection or another explicitly authorized diagnostic; none was requested or performed.

No strict all-metric production release is accepted. Improvements are classified independently: fixed-output runtime improvements, class-dependent RD leads, rate/quality or rate/runtime trades, regressions and inconclusive low-overlap cases. No conventional-codec superiority claim is made and no AVIF/JXL comparison was added.

All new files are under research_pass2/. Production codec/defaults/model/datasets, existing user work and Pass 1 reports/artifacts are preserved. No training, remote image/compression request, commit, push or deployment occurred. Source/model/input and Pass 1 integrity, saved stream hashes, repeat determinism, output equivalence and syntax accounting are checked by final_checks.py. git_status_before.txt, git_status_after.txt, changed_files.txt, artifact_manifest.json and final_validation.json record the final inventory and verification.
'''
    (OUT / 'PBC_PASS2_SUMMARY.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    rng_report()
    performance_report()
    quality_report()
    entropy_report()
    summary_report()
