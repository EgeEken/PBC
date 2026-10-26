import json
import statistics

from support import ROOT, OUT, PASS2, read
from reports import table, aggregate_pairs
from analyze import collapse


def grouped_pairs(name, candidate_key='candidate', baseline=None):
    rows = read(name)
    result = []
    for candidate in sorted({r[candidate_key] for r in rows}):
        for dataset in ('hpt_data', 'hpt_data_val'):
            group = [r for r in rows if r[candidate_key] == candidate and r['dataset'] == dataset and (baseline is None or r['baseline'] == baseline)]
            if group:
                result.append(dict(candidate=candidate, dataset=dataset, cases=len(group),
                    strict_rd_wins=sum(r.get('strict_rd_win', False) for r in group), **aggregate_pairs(group)))
    return result


def performance():
    result = grouped_pairs('performance_full_paired')
    combination = [r for r in read('combined_effects') if r['experiment'] == 'performance_full']
    correctness = read('performance_correctness')
    integration = read('integration_checks')
    variability = [{**r, 'encode_cv_pct': 100 * r['encode_seconds_mean_cv'], 'decode_cv_pct': 100 * r['decode_seconds_mean_cv']}
        for r in read('performance_full_summary') if r['preset'] == 'all']
    memory = [r for r in read('grid_allocation') if r['count'] == 500000]
    manifest = read('hf_manifest')
    text = f'''# Performance integration and HF validation plan

All measurements are local. Production files, defaults, model, UI and running HF Space are unchanged. The exact proposed integration is copied to integration_candidate/pbc3_stream.py and supplied as proposed_grid_io.patch; it is not applied. `git apply --check research_pass3/proposed_grid_io.patch` passes. Full confirmation exercises the proposed module's dispatch path, not just the earlier standalone NumPy prototype.

## Scope and correctness

The current-root baseline, grid only, existing internal-MSE skip only and both combined were screened on six predetermined images/all presets, then confirmed on all 20 main/five holdout images/all four presets, three interleaved warmed repeats. The combined result is measured directly; individual percentages are never added as evidence.

The bit-layout gate covers {len(correctness['bit_layout'])} cases: widths 0–16, all eight initial offsets, empty/short/long and strided integer inputs. Signed int64 and uint64 extremes preserve low-bit masking. Independent image checks cover 1×1 through 97×131, RGB, opaque and varying-alpha RGBA, original/4096-pixel working limits, file equality, full-channel pixels and internal metric availability. Manual generated-palette fixtures exercise actual precision 1 and 9. Contiguous and strided byte-buffer cases pass. The proposed scalar/NumPy dispatch copy adds {len(integration['bit_layout'])} bit-layout checks and eight end-to-end mode/metric combinations.

Truncated STORE/LZMA streams, bad version/entropy tags, explicit-palette payload flags, missing declared patches, zero dimensions, zero cell size, invalid RGB channel and out-of-range geometry preserve the baseline's tested accept/reject and exception classes. Valid zero-patch streams remain valid. Full malformed observations are in performance_correctness.json. The helper rejects unsupported negative counts/widths and widths above uint16, and checks required payload bytes before allocating a positive-width output. Noncanonical private helper inputs now have explicit type/range errors; error messages are not promised identical. Valid production patch widths are at most nine.

This is not a decoder security rewrite. Existing header canvas allocation, aggregate dimensions/channel/patch counts, coordinate validation and some NumPy/PIL exception behavior remain independent limitations. Arbitrarily huge headers are not safely stress-tested by attempting huge allocations. Raw zero-bit grids can still imply allocation without consuming payload bits. These limits must not be described as resolved by faster grid I/O.

## Complete-codec results

Times are means of per-image/preset medians. Independent original-size decoded RGB MSE/bpp remain exactly unchanged for every current variant; all emitted bytes, pixels, counts and patch payloads match current baseline and the frozen current streams. Detailed per-image wins/regressions remain in performance_full_paired.json.

{table(result, [('Set','dataset'),('Variant','candidate'),('Cases','cases'),('MSE','mse_candidate'),('bpp','bpp_candidate'),('Encode base s','encode_seconds_baseline'),('Encode s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Decode base s','decode_seconds_baseline'),('Decode s','decode_seconds_candidate'),('Decode %','decode_seconds_pct')])}

Measured combined effect versus the sum of separately measured absolute savings:

{table(combination, [('Set','dataset'),('Separate savings sum s','sum_of_separate_time_savings'),('Measured combined saving s','measured_combined_saving'),('Offset s','combined_offset_seconds')])}

A positive offset means combined saving is smaller than the separate-savings sum; a negative offset means larger. Independent work reductions do not guarantee additive latency because allocations/cache/order and timing noise interact. Metric skipping does not change decoder code; its small decode differences are noise. Grid/both decode gains are the relevant causal comparison.

Actual-call means and repeat variability:

{table(variability, [('Set','dataset'),('Variant','label'),('Encode mean s','encode_seconds_mean_of_calls'),('Encode CV %','encode_cv_pct'),('Decode mean s','decode_seconds_mean_of_calls'),('Decode CV %','decode_cv_pct')])}

## Allocation, dtype and startup implications

MSB bit order, padding and reader state are unchanged for valid inputs. The writer requires a flat integer array; production supplies raveled integer indices. It normalizes to uint16 and masks via selected low bits. The reader uses uint16 arithmetic with supported widths, handles non-byte alignment and normalizes noncontiguous byte buffers only when needed. Ordinary LZMA/STORE byte bodies avoid that copy.

Vectorization allocates O(grid cells × bit width) temporary arrays. Large-grid tracked allocation probes (Python/NumPy tracking, not total native memory):

{table(memory, [('Implementation','implementation'),('Operation','operation'),('Width','width'),('Tracked peak bytes','python_tracked_peak_bytes')])}

Small/medium probes also remain in grid_allocation.json. NumPy is already a codec dependency; operator is standard library. There is no new model, GPU or compiled extension dependency. Fresh-process import measurements are in startup.json; OS file caches are warm and these are not pristine cold-machine tests. Primary encoder timers include the complete compress call; interpreter import and worker-mode installation are separate. Warmup rows retain first-invocation/cache-loading effects.

## API hazard and valid caller paths

`compute_final_mse=False` already exists. It leaves the reconstructed preview available and returns internal `result.mse=None`; the artifact and independent decoded metric are identical. The current website executes `float(result.mse)` in server.py, so globally disabling this flag would break the response. Keep public UI configs at True. A file-only caller can opt out. A benchmark requiring a quality value must independently decode/score and report that metric's cost outside encode latency; a UI that still computes MSE elsewhere has not eliminated that work. Do not substitute zero or silently display None. No server.py change is proposed in this patch.

## Concrete unapplied deployment plan

1. Review proposed_grid_io.patch. Its only production file is pbc3_stream.py: retain scalar functions, add tested NumPy functions and a `GRID_IO_MODE` dispatch defaulting to scalar. No PBC3Config, decoder syntax, model, preset or public API changes. The isolated candidate copy has passed exactness checks. Rollout of a NumPy default would be a separate approval after HF measurements.
2. For the first authorized cpu-basic test, production code need not change at all. Stage only hf_cpu_benchmark.py, hf_manifest.json and integration_candidate/pbc3_stream.py, plus the six immutable release Python modules under a private reference directory. Use the existing production model and existing hpt_data/hpt_data_val files after hash checks. Model SHA-256: `{manifest['model_sha256']}`. The manifest validates normalized-LF source hashes, candidate hash, all 25 raw input hashes and model hash before timing.
3. Run the private harness from the HF checkout, after explicit authorization, with `python research_pass3/hf_cpu_benchmark.py --reference PATH_TO_IMMUTABLE_RELEASE --output research_pass3/hf_cpu_basic_run --label HF_CPU_BASIC --repeats 3`. It does not use HTTP, upload images, restart services or alter production source. A local packaging smoke run is retained under hf_plan_smoke and is explicitly labeled LOCAL_PACKAGING_SMOKE; it is not HF evidence.
4. Reference and current source trees require separate persistent subprocesses to prevent module leakage. Both pin to the same first allowed logical CPU (or explicit --cpu N), use numerical thread settings 1 and process requests serially. All four current modes execute on one warmed current worker; no concurrent mode switching in the public FastAPI process is permitted. Model/config/module paths, actual versions, cold worker startup, warmups and raw rows are retained. The authoritative release is the primary old control. If actual old_pbc is additionally audited, use an isolated worker and identify that source honestly; do not silently call it source-identical release.
5. The existing broader benchmark's _eval_pbc3 correctly calls compress/decompress and scores decoded RGB, but lacks these paired modes/provenance. Quick RD lacks independent decode timing and is unsuitable for promotion. The supplied private harness reuses the validated methodology with standalone packaging, rather than changing the public benchmark APIs. It records full encode/decode separately; data loading, metric computation, inspection and IPC stay outside those timers.
6. No HTTP endpoint change is needed for CPU validation. If an approved service experiment is later required, add a private, serial job runner over existing server-side inputs. Record codec encode/decode, metric, preview PNG/base64, serialization and total server times separately. Measure client RTT independently; RTT minus server time includes transport and queuing and must not be called codec time. Never infer encoder latency from a browser progress timer or subtract heterogeneous historical results.
7. Promotion requires zero baseline-versus-optimized stream/pixel/count/MSE/bpp differences on all same-runtime cases and repeated calls, valid malformed/alpha checks, positive repeatable aggregate encode gains and substantial decode gains without a material latency-tail regression. Rerun noisy outliers rather than accepting a single timing. Keep UI MSE enabled. Compare HF only with HF; local speed percentages are not deployment results. Existing cross-runtime encoder trajectory differences remain a separate issue.
8. Rollback is mode=scalar in the private worker. If an authorized source rollout later occurs, restore only pbc3_stream.py from its recorded pre-change backup/revision and restart only with separate approval. Preserve all unrelated files. No Git reset, synchronization or model rollback is part of this plan.

This is ready for a specifically authorized private HF validation test, subject to the limits above. It is not an approval to deploy and is not a claim of validated cpu-basic performance. No commit, push, deployment, restart or remote compression call occurred.
'''
    (OUT / 'PBC_PERFORMANCE_INTEGRATION.md').write_text(text, encoding='utf-8')


def allocation():
    summaries = [r for r in read('allocation_full_summary') if r['preset'] != 'all']
    matches = [r for r in read('allocation_matched_summary') if r['experiment'] == 'allocation_full']
    main_release = [r for r in matches if r['baseline_source'] == 'release' and 'integrated' in r['label']]
    strongest = [r for r in matches if r['baseline_source'] == 'best_existing_all' and 'integrated' in r['label']]
    static = read('representation_study_summary')
    ledger = read('patch_ledger')
    groups = []
    for label in sorted({r['label'] for r in ledger}):
        rows = [r for r in ledger if r['label'] == label]
        groups.append(dict(label=label, patches=len(rows), raw_bytes_per_patch=statistics.mean(r['raw_bits'] / 8 for r in rows),
            grid_cells=statistics.mean(r['grid_cells'] for r in rows), bitcount=statistics.mean(r['actual_bitcount'] for r in rows),
            sse_gain_per_raw_bit=sum(r['local_sse_gain'] for r in rows) / sum(r['raw_bits'] for r in rows)))
    outcomes = grouped_pairs('allocation_paired', 'label', 'q500p35_none')
    worst = sorted([r for r in read('allocation_matched') if r['experiment'] == 'allocation_full' and r['baseline_source'] == 'best_existing_all' and 'integrated' in r['label'] and r['matched_rate_mse_pct'] is not None], key=lambda r: r['matched_rate_mse_pct'], reverse=True)[:10]
    conclusion = read('conclusions')
    geometry = read('geometry_ablation_summary')
    text = f'''# Rate-aware patch representation and allocation

{conclusion['allocation_verdict']}

## Frozen-trajectory diagnostics

Thirty existing streams cover six predetermined main images (cat, Monet, flower, cathedral, googlewiki and stockpoint): release/current Quality and HQ at 250k, plus current 500k/35. No search is rerun for this study. patch_ledger.json records every iterative geometry, area, grid shape/cells, palette bounds/precision/mask, exact serialized raw bits, exact clipped working-space SSE change and cumulative full-canvas SSE. Summed rectangle deltas are checked against full working-canvas recomputation at sampled positions. All signed quantized grids are retained in 30 compact NPZ files, linked by key/hash from each row; original indices remain in the protected source streams.

{table(groups, [('Control','label'),('Patches','patches'),('Mean raw bytes/patch','raw_bytes_per_patch'),('Mean cells','grid_cells'),('Mean actual bits/index','bitcount'),('Working SSE gain/raw bit','sse_gain_per_raw_bit')])}

The first, one-third, two-thirds, last and largest-raw-cost patches (deduplicated) also have actual original-size RGB prefix changes and marginal complete-file bits. These use actual serialized/LZMA-compressed/decoded prefixes. Negative RGB gain can occur even when working-channel SSE decreases. Prefix bit differences include entropy context and header-count changes; raw patch bytes are not additive post-LZMA ownership. No LZMA is called inside either encoder selector.

## Fixed-state representation study

For at most five patches per stream, hold pre-patch canvas and geometry fixed. Test original, at most twice the cell size, one lower maximum palette bitcount, and an optional half-cell finer-grid diagnostic. These all use v0 patch syntax. Each alternative is actually projected, quantized, resampled and exact-scored; not estimated from unquantized residual energy. An untimed discarded action build warms relevant kernel paths before timings.

{table(static, [('Control','label'),('Alternative','method'),('Patches','patches'),('Mean retained local gain','mean_retained_gain'),('Mean raw-bit fraction','mean_bit_fraction'),('98% gain / 75% bits cases','near_lossless_cheap'),('Evaluation s','mean_evaluation_seconds')])}

No cheaper alternative locally dominates the original in the sampled set, and none meets the strict >=98% gain / <=75% bits shortlist. Thus no fixed-suffix full-file shortlist was manufactured. However, 44 of 262 cheap alternatives retain >=95% local gain at <=85% raw bits, and 184 improve local gain/raw-bit ratio. This is sufficient to test a small rate tradeoff, not evidence of a global codec gain. Precision reductions are generally gentler than density reductions; coarsening often sacrifices about half the local gain. The original selected high-detail corrections are not simply redundant or free to remove.

## Selector and independent ablations

Use three representations maximum: original plus coarser grid and lower precision, deduplicated. Let G0/B0 be the ordinary representation's exact clipped SSE gain/raw bits. Select by `gain - t*(G0/B0)*raw_bits`, t in {{0.1, 0.4, 0.8}}. Original wins exact utility ties. Retain the ordinary policy's positive-gain stop decision in the integrated encoder; no cheaper action rescues an ordinary nonpositive prediction. All tested t<1 preserve a positive original utility. There are no new models, proposals, random draws, geometry beams or entropy evaluations in selection. The rate is an inexpensive raw-bit proxy, not a fitted final-size predictor.

B3a performs one forward reconstruction with original channel/geometry/order/count fixed; representations are fitted against the evolving replay canvas using the original action as reference. It first performs the complete ordinary encode, then parsing/replay and final serialization, so the full duplicate work is charged. Identity replay reproduces all 18 screen controls byte-for-byte. It is a diagnostic, not an efficient shipping design, and is distinct from earlier backward refitting/pruning with repeated full decoding. If a frozen-geometry reference becomes nonpositive, count/order are still retained for this ablation; its nonpositive-reference counters are not a claim of an ordinary encoder stop.

B3b hooks only the representation decision after the ordinary chosen geometry/action, evaluates the same bounded alternatives and lets later channels/residuals/proposals/model actions evolve naturally. Initialization, RNG algorithm/seed, resolution, requested budget, proposal count and model calls per step remain the matched ordinary controls. Actual emitted counts may change through the ordinary stop path. Geometry/payload changes are recorded in allocation_trajectories.json.

Both screens used six images, three existing reference configurations (Q250/HQ250/500k35), three weights and two repeats. Full-stream screening selected only 500k/35 weights 0.4 and 0.8, frozen before new holdout results. Matched fixed-geometry counterparts are retained as full ablations. HQ250 failed the stronger previous-operating-point control despite some release-default wins; Q250 mostly moved below the target Q–HQ rate range. No 1M, classifier, new action model or broad candidate search was added.

## Full main/holdout measurements

All 20 main and five holdout images, three interleaved warmed repeats, nine configurations: release/current Q/HQ, current 500k35, the two frozen integrated candidates and their two fixed-geometry counterparts. Every artifact is fully serialized and independently decoded/scored. All extra representation work, changed stopping counts and fixed replay cost are included in encode wall time.

{table(summaries, [('Set','dataset'),('Configuration','label'),('MSE','mse'),('bpp','bpp'),('Raw bytes','raw_bytes'),('Final bytes','final_bytes'),('Iterative patches','iterative_patches'),('Encode s','encode_seconds'),('Decode s','decode_seconds')])}

Paired against ordinary current 500k35 (same resolution, model input, budget and RNG):

{table(outcomes, [('Set','dataset'),('Candidate','candidate'),('MSE %','mse_pct'),('bpp %','bpp_pct'),('Raw bytes %','raw_bytes_pct'),('Encode %','encode_seconds_pct'),('Decode %','decode_seconds_pct'),('Strict RD image wins','strict_rd_wins')])}

Integrated points versus the measured fixed-geometry family (ordinary 500k35 plus fixed weights 0.4/0.8), at overlapping rates:

{table(geometry, [('Set','dataset'),('Integrated setting','label'),('Overlap','overlap'),('Matched MSE %','mean_mse_pct'),('Wins','wins')])}

geometry_ablation_matched.json retains these exact brackets. This separates removed replay overhead from any RD effect of allowing future geometry to evolve; neither a lower absolute MSE nor the local utility alone proves an improved curve.

Counterfactual original/selected bits/gains in experimental_stats refer to representations on that candidate's own visited states; they are not the baseline trajectory's total cost. Per-image actual final files determine the comparisons.

## Matched-rate and measured-frontier evaluation

Default release frontier comparison:

{table(main_release, [('Set','dataset'),('Integrated candidate','label'),('All-rate overlap','overlap'),('Mean MSE %','mean_matched_rate_mse_pct'),('Wins','matched_rate_wins'),('Q–HQ overlap','q_hq_overlap'),('Q–HQ MSE %','q_hq_mse_pct'),('Q–HQ wins','q_hq_wins')])}

Stronger comparison with the union of all previously measured Pass 2 release/current operating points, including available resolution/budget controls:

{table(strongest, [('Set','dataset'),('Candidate','label'),('Rate overlap','overlap'),('Mean MSE %','mean_matched_rate_mse_pct'),('Median MSE %','median_matched_rate_mse_pct'),('Wins','matched_rate_wins'),('Within 5% rate','near_5pct_rate_cases'),('Near-rate MSE wins','near_5pct_mse_wins'),('MSE overlap','mse_overlap'),('Matched-MSE bpp %','matched_mse_bpp_pct'),('Measured dominated','observed_dominated')])}

All matching is per image, with exact measured neighboring nondominated points, arithmetic bpp/MSE interpolation and no extrapolation. Raw brackets/weights are in allocation_matched.json. Q–HQ denotes brackets whose measured neighbors are specifically Quality/HQ defaults; broader controls may include lower-rate regions. Missing overlaps are neither wins nor losses. Matched-rate and matched-distortion populations differ, so aggregate signs must not be combined into joint dominance. Previous screening images have denser prior curves than other images; every comparison retains that coverage. Sparse interpolation is an estimate rather than an achieved extra file or continuous frontier proof.

Near-rate comparisons retain each actual nearest control, its bpp distance and measured MSE difference. A 5% rate tolerance is not exact matching, and most holdout comparisons still require interpolation. The interval between measured points can hide a nonlinear baseline curve; denser matched-rate controls remain a necessary follow-up before a release claim.

Largest matched-rate regressions against the stronger existing envelope:

{table(worst, [('Set','dataset'),('Image','image'),('Candidate','label'),('MSE %','matched_rate_mse_pct'),('Candidate MSE','mse'),('Candidate bpp','bpp')])}

allocation_paired.json additionally retains strict observed-point wins, actual operating-point tradeoffs and all latency changes against five fresh controls. RD plots (rd_screen, rd_hpt_data, rd_hpt_data_val PNG/SVG) are backed by allocation_points.json/.csv. Connecting lines are guides; JSON brackets are authoritative. Improvements on a subset are content-dependent. No Compression/Balanced default is changed or proposed: selection is confined to Q/HQ-derived opt-in experiments, so no universal low-bpp claim is made. Content routing is skipped; 25 images and the observed heterogeneity do not support tuning another automatic policy on this holdout.

## Interpretation

{conclusion['allocation_interpretation']}

An observed proxy failure is 15k_kills.png: increasing weight from 0.4 to 0.8 reduces raw bytes 128,002 -> 109,669 but raises complete bytes 37,115 -> 46,012 and RGB MSE 62.41 -> 70.38. Actual accepted patches change 33 -> 35. Thus neither raw cost nor a larger rate penalty guarantees a smaller final file after the sequential trajectory and LZMA wrapper. This is a measured reversal; attributing it to any one symbol statistic would require another experiment.

The local proxy identifies real tradeoffs, but its success alone is not evidence of the final encoder. Search geometry, clipping, RGB conversion/resizing, later patch interference, ordinary stop decisions and LZMA jointly determine outcomes. This pass separates those through the fixed/integrated ablations and actual final-file comparisons. No strict production release is accepted and no performance optimization is combined with the allocation candidates.
'''
    (OUT / 'PBC_RATE_AWARE_ALLOCATION.md').write_text(text, encoding='utf-8')


def search():
    results = grouped_pairs('search_full_paired')
    screen = grouped_pairs('search_screen_paired')
    traces = [r for r in read('search_trace_summary') if r['phase'] in ('action.search.visible_error', 'action.search.integral_error', 'encode.patch.score', 'encode.patch.apply')]
    check = read('search_compatibility')
    exact = read('search_exactness')
    verified = sum(r['experimental_stats'].get('verified_proposal_lists_and_rng', 0) for r in exact)
    installs = [r['experimental_stats']['prototype_install_seconds'] for r in read('search_full') if r['optimization'] != 'none']
    setup_screen = grouped_pairs('search_setup_screen_paired')
    text = f'''# Exact search-state reuse

Two local approaches were tested without changing any proposal/RNG/patch semantics. Both retain the original NumPy integral construction; there is no arbitrary-rectangle integral update, integral-fusion kernel, native rewrite or approximate search.

## Valid state and invalidation

Constant: prepared target values, dimensions/config/model weights within a stage. Evolving: canvas, clipped visible errors, channel totals, search integrals/anchors, sampled geometry and all residual-dependent features. Search-q and step-fraction features change each step. Target/canvas owners change on a new encode or warmup resize and invalidate cached maps. A final integral cannot simply survive an accepted patch; it is always rebuilt with the original operation/order.

Approach 1 (`scores`) retains the absolute-error array already computed for each channel score and reuses it in the next search for that unchanged channel. Changed-channel scoring still fully refreshes that map. It removes the duplicate clip/subtract/abs calculation without changing reduction order.

Approach 2 (`rectangles`) stores each channel's int32 absolute-error map and exact integer total. After the real accepted grid/delta is applied to the signed-int16 canvas, recompute only its affected rectangle from the actual clipped post-update canvas and adjust total by exact old/new rectangle sums. Unchanged pixels/channels remain valid. This handles hidden overshoot, clipping and even signed-int16 wrap consistently with existing behavior. It adds two rectangle reductions and bookkeeping, so it is not free. Integrals/anchors are still computed normally, from exactly equal full maps. No target mean/std or percentile result is cached for arbitrary new boxes.

Before timing, both approaches matched all 48 six-image/four-preset checks. This includes {verified} complete proposal-list and RNG-state comparisons, full error maps, exact channel totals, complete streams, pixels and patch trajectories. Synthetic signed-int16 clipping/wrap tests validate 100 rectangle mutations. Additional RGB/RGBA, both selected-delta and apply-grid paths, warmup/new owners and repeated encodes in one context are in search_compatibility.json ({len(check['image_checks'])} cases, two repetitions each). Validation mode's full recomputations and duplicate searches are excluded from speed tests.

## Timing and bookkeeping

The initial screen excluded one-time experimental scope installation and is retained separately where applicable. The primary confirmation charges cache initialization and any per-call prototype installation/cleanup as documented in its plan. All map updates, rectangle sums, array allocations, ordinary integral reconstruction and other codec work are inside full compress latency. Only one strategy is promoted. The source patching mechanism is experimental; a future production implementation would own per-encode state explicitly rather than mutate global functions in an HTTP process.

Screen:

{table(screen, [('Variant','candidate'),('Cases','cases'),('Encode baseline s','encode_seconds_baseline'),('Encode candidate s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Encode wins','encode_seconds_wins')])}

Frozen promoted-strategy reconfirmation after charging prototype installation:

{table(setup_screen, [('Variant','candidate'),('Cases','cases'),('Encode baseline s','encode_seconds_baseline'),('Encode candidate s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Encode wins','encode_seconds_wins')])}

Mean per-call prototype state/namespace installation in the full run is {statistics.mean(installs):.6f} s, included in the encode walls below along with cleanup. Runtime mode selection itself remains benchmark setup, as for other paired modes.

Full fixed-candidate main/holdout confirmation:

{table(results, [('Set','dataset'),('Variant','candidate'),('Cases','cases'),('MSE','mse_candidate'),('bpp','bpp_candidate'),('Encode baseline s','encode_seconds_baseline'),('Encode s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Decode %','decode_seconds_pct')])}

The decoder is untouched by reuse; small decoder timing differences are variation. Every candidate stream/pixel/count matches the exact same current implementation with reuse disabled. Per-image wins and repeat variability are retained in search_full_paired.json and search_full_summary.json. No release-RNG change or speed optimization is combined with this experiment.

Traced target phases, separately from uninstrumented primary timing:

{table(traces, [('Variant','variant'),('Phase','phase'),('Inclusive s','inclusive_seconds'),('Exclusive s','exclusive_seconds'),('Count','count')])}

Read apply+score+visible-error together: bookkeeping shifts work into patch application. Claiming only the reduction in score time would hide that cost. Integral time is retained; unchanged integral inputs do not imply a cached integral. Experimental counters report refreshed maps, updated pixels, full pixels avoided and verification counts. RGB maps occupy roughly 3 × working_pixels × 4 bytes (about 3 MB at 250k, 6 MB at 500k) plus totals/objects; they persist for an encode. Views/copy/transient allocation and dtype assumptions are explicit in search_reuse.py.

## Decision and transfer limits

{read('conclusions')['search_verdict']}

Integer error-map equality and unchanged ordering are compatible with a single-thread CPU runtime, but local timing does not establish HF benefit. Memory bandwidth, rectangle area distribution, NumPy/Numba versions, CPU dispatch, service contention and installation design still need a future authorized same-worker HF comparison. Existing cross-runtime encoder ties are not fixed by caching; the required gate is equality to the same-runtime baseline. No cache code, production default or deployment was changed.
'''
    (OUT / 'PBC_SEARCH_REUSE.md').write_text(text, encoding='utf-8')


def summary():
    conclusion = read('conclusions')
    engineering = [r for r in grouped_pairs('performance_full_paired') if r['candidate'] == 'both']
    allocation_points = [r for r in read('allocation_full_summary') if r['preset'] != 'all' and '_integrated_' in r['label']]
    algorithm = [r for r in read('allocation_matched_summary') if r['experiment'] == 'allocation_full' and r['baseline_source'] == 'best_existing_all' and '_integrated_' in r['label']]
    qhq = [r for r in read('allocation_matched_summary') if r['experiment'] == 'allocation_full' and r['baseline_source'] == 'release' and '_integrated_' in r['label']]
    reuse = grouped_pairs('search_full_paired')
    costs = grouped_pairs('allocation_paired', 'label', 'q500p35_none')
    costs = [r for r in costs if '_integrated_' in r['candidate']]
    model = read('initial_snapshot')['hashes']['patch_policy.npz']
    text = f'''# PBC Pass 3 summary

{conclusion['summary_verdict']}

All three objectives were evaluated independently. No performance flag is combined with the allocation or reuse candidates. The authoritative immutable release is `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51`; current root remains `3e9e8c94b7497ecefc5e482a66fde697dfad6e19`. Model SHA-256: `{model}`. Old_pbc's verified default-output equivalence remains historical evidence; it is not relabeled source-identical release or redundantly fully benchmarked.

## What ran

- A: baseline/grid/skip/both, six main images/all four presets/three repeats; exact edge/API/alpha/malformed tests; then all 25 images/all presets/three repeats (1,200 primary rows). Full confirmation uses the exact unapplied integration copy's dispatch path. Allocation/startup probes and a local portable-HF-harness smoke check are separate.
- B: 30 frozen trajectories (six images, release/current Q250/HQ250, current 500k35), per-patch exact working cost/gain and sampled RGB/final-size deltas; a bounded four-representation static study; fixed-geometry identity replay (18 controls); three weights across three controls, fixed and integrated screens (144 rows each). Two 500k35 integrated settings and their fixed-geometry counterparts, with five fresh release/current controls, were frozen before new holdout results and run over 20 main/five holdout images/three repeats (675 rows).
- C: two reuse mechanisms; 48 initial exactness cases, 1,178 proposal-list/RNG checks plus full maps/totals/files, signed-int16 clipping/wrap and RGB/RGBA/warmup/owner tests. One fixed candidate is confirmed on all 25 images/four presets/three repeats (600 rows), plus separate component traces and explicit prototype-setup accounting.

All primary rows include original-size independently decoded RGB MSE, complete post-LZMA bytes/bpp, raw bytes including the six-byte outer wrapper, encode/decode call walls, actual patches, dimensions, expanded config, source/model/input hashes and module paths. `bpp = 8*complete_file_bytes/original_pixels`. NumPy error differences/squares are safe and reduced in float64. Image loading, independent metrics, inspection and IPC stay outside ordinary compress/decompress timers. Extra representation/replay/reuse work is charged as documented. One CPU core/numerical thread, warmed paths, rotated order and retained three-repeat variation. No local/HF timing mixture or pristine cold-start claim.

## Strongest numerical results

Combined runtime optimization versus unchanged current root, exactly identical output:

{table(engineering, [('Set','dataset'),('MSE','mse_candidate'),('bpp','bpp_candidate'),('Encode baseline s','encode_seconds_baseline'),('Encode combined s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Decode baseline s','decode_seconds_baseline'),('Decode combined s','decode_seconds_candidate'),('Decode %','decode_seconds_pct')])}

Integrated allocation operating points:

{table(allocation_points, [('Set','dataset'),('Candidate','label'),('MSE','mse'),('bpp','bpp'),('Raw bytes','raw_bytes'),('Iterative patches','iterative_patches'),('Encode s','encode_seconds'),('Decode s','decode_seconds')])}

Paired with ordinary current 500k35:

{table(costs, [('Set','dataset'),('Candidate','candidate'),('MSE %','mse_pct'),('bpp %','bpp_pct'),('Encode %','encode_seconds_pct'),('Decode %','decode_seconds_pct'),('Strict pointwise RD wins','strict_rd_wins')])}

Matched-rate comparison with all previously observed release/current controls:

{table(algorithm, [('Set','dataset'),('Candidate','label'),('Overlapping images','overlap'),('MSE %','mean_matched_rate_mse_pct'),('Wins','matched_rate_wins'),('Within 5% actual rate','near_5pct_rate_cases'),('Near-rate MSE wins','near_5pct_mse_wins'),('Matched-MSE overlap','mse_overlap'),('Matched-MSE bpp %','matched_mse_bpp_pct')])}

Specifically inside release Quality-to-HQ neighbor brackets:

{table(qhq, [('Set','dataset'),('Candidate','label'),('Q–HQ overlapping images','q_hq_overlap'),('MSE %','q_hq_mse_pct'),('Wins','q_hq_wins')])}

Exact search reuse versus the identical current root:

{table(reuse, [('Set','dataset'),('Encode baseline s','encode_seconds_baseline'),('Encode reuse s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Encode wins','encode_seconds_wins'),('Decode %','decode_seconds_pct')])}

All interpolated means use only valid per-image overlap; missing cases do not count as wins. Different matched-rate/distortion populations cannot be combined into a dominance claim. The stronger existing envelope is an oracle over previously measured configurations, not a deployable routing policy. Exact brackets/weights, all pointwise outcomes, outliers and source points remain in allocation_matched.json, allocation_paired.json and allocation_points.json. No extrapolation or sparse-curve BD-rate claim.

## Explicit research answers

1. Grid I/O and internal-MSE skipping combine successfully on the 100 current image/preset cases, three repeats, with complete byte/pixel/count/MSE/bpp equality.
2. Direct measurements are approximately additive within observed variation; combined_effects.json records the measured residual rather than adding percentages. Cache/allocation effects are plausible explanations for small offsets, not separately proven causes.
3. All runtime and promoted exact-reuse outputs are identical to their current controls. Allocation intentionally changes payloads and, in integrated mode, later trajectories; it is not bit-exact optimization.
4. Vectorized grids add temporary arrays and stricter private-helper argument checks. Existing malformed-header/resource limitations remain. Internal MSE=None is valid only for callers that opt out; the public server's float(result.mse) requires keeping its flag True. Global mode monkeypatching is unsuitable for concurrent public requests.
5. The private portable harness, hash manifest, isolated candidate module and unapplied one-file patch are ready for a later explicitly authorized cpu-basic test. Reference/current workers are source-isolated on the same CPU; all current modes share a warmed serial worker. It uses repository inputs and no HTTP. Public server/HTTP-overhead validation is separate and requires explicit authorization. See PBC_PERFORMANCE_INTEGRATION.md for exact files, commands, gates and rollback.
6. Expensive index grids dominate raw syntax. High Quality's sampled per-patch raw cost is roughly four times Quality's, with far more cells and higher selected precision. Final entropy-coded byte ownership is nonadditive; raw totals cannot assign a unique compressed cost per patch.
7. Some cheaper representations retain nearly the same local gain: 44/262 retain at least 95% at at most 85% raw bits. None locally dominates or meets the stricter 98%/75% shortlist. Cheap density reduction often loses much more gain than a precision step.
8. {conclusion['sequential_answer']}
9. {conclusion['frontier_answer']}
10. {conclusion['holdout_answer']}
11. The tested allocator is opt-in on Q/HQ-derived paths; Compression/Balanced defaults are not modified. No universal allocator/no-low-bpp-regression claim is made. The independent performance optimization is proven output-preserving across all four presets.
12. {conclusion['outlier_answer']}
13. Full-channel visible error arrays can be shared with search until their channel changes. Rectangle-local error maps/totals can be updated after actual mutation; owner/stage changes require invalidation. Integrals, anchors and residual-dependent features cannot simply be reused stale.
14. Both screened reuse approaches preserve intermediate maps/totals, complete proposal order/RNG state and complete streams in verification. The promoted full-dataset candidate retains complete byte/pixel/count equality.
15. {conclusion['search_verdict']}
16. The arithmetic and avoided allocations are compatible with one CPU core, but transfer is unmeasured. HF requires same-runtime equivalence and interleaved latency/memory checks with actual versions, CPU dispatch and service load. Existing cross-runtime trajectory differences are not explained away or fixed.
17. {conclusion['pbc31_answer']}
18. {conclusion['limitation_answer']}
19. {conclusion['next_rd_experiment']}
20. {conclusion['strategic_answer']}

## Classification and next work

{conclusion['classification']}

Ranked Pass 4 directions:

{conclusion['next_steps']}

Observations are the complete-file/metric/latency tables and exactness checks. Causal evidence comes from paired flag/control comparisons, same-state alternative representations and fixed-versus-evolving geometry ablations. Explanations involving entropy predictability, model distribution shift or future action reuse remain hypotheses unless directly measured. A new representation, retraining, broader beams and a universal content classifier are not justified merely by a proxy improvement.

## Reproduction and repository state

README.md lists exact commands and artifact roles. The four requested reports, raw CSV/JSONL, plots/source points, quantized-grid diagnostics, equality records, provenance, proposed patch and portable benchmark all reside under research_pass3/. changed_files.txt lists every created file; artifact_manifest.json gives byte lengths and SHA-256. Cache files are excluded from the measurement inventory.

Before/after git status and source/model/dataset snapshots are retained. final_checks.py verifies all protected Pass 1/2 artifacts, source/model/input hashes, repeated determinism, stream files, error accounting, fixed-geometry invariants, paired exactness, integration applicability and packaging smoke. No production source/default/model/public behavior was changed. Existing .gitignore/user work is preserved. No commit, push, synchronization, deployment, restart, image upload, remote compression request or training occurred. No PBC 3.1 release is declared.
'''
    (OUT / 'PBC_PASS3_SUMMARY.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    performance()
    allocation()
    search()
    summary()
