import json
import re
import statistics

from support import OUT, PASS3, read
from reports import table, aggregate_pairs
from analyze import collapse


def runtime_report():
    aggregate=read('runtime_full_aggregate')
    summary=read('runtime_full_summary')
    metadata=read('hf_observations')
    checks=read('hf_reported_revision_source_checks')
    space=next(r['body'] for r in metadata if r['url'].startswith('https://huggingface.co/api/'))
    exact=read('runtime_exactness')
    debug=[r for r in exact if r['label']=='all_verify']
    proposal_checks=sum(r['experimental_stats']['verified_proposal_lists_and_rng'] for r in debug)
    endpoints=[dict(method='GET',path='/api/health; /healthz',operation='Health/boot identity',selection='No codec timing; observed /api/health 200'),
        dict(method='GET',path='/openapi.json',operation='Current route/schema inspection',selection='Observed 200; does not expose internal loaded hashes/versions'),
        dict(method='GET',path='/api/benchmark/status',operation='Read active benchmark state',selection='Observed idle; no start performed'),
        dict(method='GET',path='/api/benchmark/dataset',operation='Load/describe all main inputs',selection='No image-selection execution; not invoked'),
        dict(method='POST',path='/api/benchmark/start',operation='Append DB rows for selected codecs/trials',selection='PBC minimum 20 images x 4 presets; no subset/variant selector'),
        dict(method='GET',path='/api/benchmark/results; /download_db',operation='Historical DB results/file',selection='No fresh source/version identity; not treated as current timing'),
        dict(method='POST',path='/api/benchmark/stop; /reset_pbc; /upload_db',operation='Stop/delete/replace benchmark state',selection='Not invoked; resets/uploads outside scope'),
        dict(method='GET',path='/api/quick_rd/status; /results',operation='Status/shared saved points',selection='Status idle; stale output is not timing evidence'),
        dict(method='POST',path='/api/quick_rd/start',operation='Full default old/current comparison',selection='Overwrites shared JSON; lacks decode timing/independent score; not invoked'),
        dict(method='POST',path='/api/compress; /stream_compress; /decode',operation='User-supplied image/stream operation',selection='Requires an upload; not invoked'),
        dict(method='GET',path='/api/sweeps/run/status',operation='Sweep status',selection='Observed idle; no sweep/training started')]
    body=f'''# HF and local runtime validation

All three runtime mechanisms are compatible and output-preserving locally. HF validation of the new implementation is incomplete: it is not deployed and the existing API cannot select it. Remote work was read-only health/schema/status/source inspection. No remote codec timing, image upload, benchmark start, database mutation, restart or deployment occurred.

## Identity and actual remote observations

The existing [Space](https://egeeken-pbc.hf.space/) answered health/schema/status GETs successfully. HF's [Space metadata](https://huggingface.co/api/spaces/EgeEken/PBC) reported stage `{space['runtime']['stage']}`, hardware `{space['runtime']['hardware']['current']}`, one replica, repository/runtime revision `{space['runtime']['sha']}`. Health boot ID is `{metadata[0]['body']['boot_id']}`; it is not a commit. Observations are timestamped in hf_observations.json. The date is 2026-10-26 Europe/Paris (recorded timestamps are UTC). Sandbox DNS failures were followed by authorized read-only network access; the failed records remain in hf_observations_sandbox_failure.json.

Nine successfully fetched source/startup files at the reported immutable revision match local text after CRLF normalization: server, broader/Quick RD backends, PBC3, ops, stream, learned filler, Dockerfile and requirements_docker.txt. One mistaken requirements.txt probe returned 404 and is retained. These are reported-revision source checks, not inspection of in-memory imports. The process does not expose installed NumPy/Numba/CPU dispatch, loaded model digest or actual in-memory codec hashes. Docker specifies Python3.11/Pillow11.3.0 and mostly unpinned other packages; these declarations do not establish installed versions. Prior Pass 1 model verification at this immutable revision remains historical evidence. No new full remote equivalence assertion is made, and the earlier Quality trajectory discrepancy remains unresolved.

## Existing API/backend inventory

{table(endpoints,[('Method','method'),('Path','path'),('Operation','operation'),('Capability/decision','selection')])}

The complete live OpenAPI route inventory is remote_endpoint_inventory.json; local server registration paths/lines are local_endpoint_inventory.json. The live schema agrees with the relevant benchmark paths. Current normal start accepts only codecs/n_trials; query filters mp_min/mp_max apply to results, not execution. Authentication is not declared for the existing benchmark routes. The regular backend appends a generated timestamp run_id into the existing SQLite database; callers cannot choose an isolated run ID or a small input/preset subset. Starting PBC alone means at least 80 encodes and 80 decodes. This exceeds the requested small smoke scope, and cannot evaluate undeployed modes, so it was not run. Quick RD is additionally unsuitable because it overwrites shared results and measures the encoder preview/internal encode duration, without independent decode latency.

The broader backend's _eval_pbc3 calls production compress and decompress, separately times them, scores decoded RGB and counts complete file bytes. It preprocesses EXIF/RGB from hpt_data. Its extra resize can hide a size mismatch; the research harness instead asserts original dimensions. Trials are grouped, not interleaved by implementation; preload does not guarantee per-config warmed paths or thread affinity. Its stored rows lack expanded configs/source/model/input/dependency digests and have no experimental implementation selector. Existing BENCHMARK_LOCK coordinates benchmark/sweep/training jobs, but ordinary public compression requests can still contend for CPU. Idle status does not prove absence of every other process. No observed HTTP RTT is reported as codec latency.

## Local equivalence and full timings

Before primary timing, the combined variant matched all 100 image/preset files, decoded pixels, patch signatures/order/counts, original RGB MSE and bpp, with {proposal_checks} full proposal-order/RNG comparisons. Sixteen extra RGB/RGBA cases cover all presets, both selected-delta and apply-grid mutation paths, and working-size downsampling. Prior Pass 3 bit layout/malformed/warmup proofs are protected and reused. v0 framing, palettes and decoder syntax are unchanged. Every full repeated current-control file also matches its frozen Pass 1 current file.

Modes: baseline current root; Pass3 grid+existing MSE opt-out; rectangle reuse alone; all three. No RNG replacement, model change, patch-budget change or allocator is mixed into A. Full confirmation is all25 images/four presets/three warmed interleaved repeats, 1200 rows. Means below average per-image/preset medians; summary CSV/JSON also retains actual-call means/CV and per-preset values. Rows include after-call RSS, not a controlled peak-RSS difference.

{table(aggregate,[('Set','dataset'),('Candidate','candidate'),('Control','baseline'),('Cases','cases'),('Exact','exact'),('Encode base s','encode_seconds_baseline'),('Encode s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Encode wins','encode_seconds_wins'),('Decode base s','decode_seconds_baseline'),('Decode s','decode_seconds_candidate'),('Decode %','decode_seconds_pct')])}

Per-preset and variability:

{table(summary,[('Set','dataset'),('Preset','preset'),('Mode','label'),('MSE','mse'),('bpp','bpp'),('Encode s','encode_seconds'),('Encode CV','encode_seconds_mean_cv'),('Decode s','decode_seconds'),('Decode CV','decode_seconds_mean_cv')])}

The all-versus-grid_skip pair measures the incremental cache benefit directly. Search state construction, namespace installation, rectangle updates, reductions, allocations and cleanup are inside compress wall time. The private mode-selection/grid hook itself is benchmark setup for all modes; it would become static dispatch in a production design. Small decoder differences between identical decoder paths are noise, not cache benefits. No separately measured percentages are added to claim a combined gain. All full per-image/preset wins/regressions and timing extrema remain in runtime_full_paired.json and runtime_full.jsonl.

Memory is a material tradeoff: int32 cached error maps occupy approximately 3MB for RGB at250k and6MB at500k, plus objects/transients. Grid I/O adds O(cells x bitcount) temporary arrays; Pass3's measured 500k-cell/9-bit tracked write/read peaks were18.0/14.5MB versus1.18/1.00MB scalar. These components do not establish additive whole-codec peak RSS. Metric skipping avoids the internal full-original-image metric allocation/work, but retains the preview and requires independent benchmark scoring. Arbitrary decoder header/resource hardening is outside this change.

## Minimal unapplied integration into this Space

proposed_hf_benchmark.patch changes only server.py and pbc3_benchmark.py, and remains unapplied. It adds an opt-in authenticated branch to the existing POST /api/benchmark/start, retains existing status/stop infrastructure, and changes neither ordinary benchmark execution nor public compression defaults. Stage only hf_runtime_job.py, hf_candidate_grid.py, hf_candidate_search.py and hf_runtime_manifest.json under research_pass4; no Pass1/2 framework or second Space is needed. Candidate copies are exact Pass3 implementations. Linux uses standard-library CPU affinity, with no new HF dependency; Windows packaging smoke uses existing psutil.

Static allowlists: images googlewiki.png/flower.jpg, up to two of four presets, modes baseline/grid_skip/search/all, repeats1..3, at most48 timed calls plus bounded per-mode/preset warmups. No code, module path, filesystem path, shell command or arbitrary config is accepted. baseline is required. A disabled-by-default PBC_RESEARCH_BENCHMARK_TOKEN secret gates the opt-in branch; compare the request's x-pbc-benchmark-token with hmac.compare_digest. Never put it into a URL, report or saved result. The normal route is unchanged. Handler/allowlist tests check disabled404, unauthorized403, invalid400, guarded valid dispatch and ordinary-route arguments. These are local extracted-handler tests, not a live server test.

The guarded backend starts a dedicated bounded subprocess, keeping monkeypatches outside the public FastAPI worker. All modes share one warmed serial worker and allowed CPU. Source/candidate normalized hashes, raw input/model hashes, installed versions, thread settings, module paths, expanded configs and output/pixel hashes accompany results. Full compress/decompress times are distinct from loading, scoring, IPC and HTTP. Each job writes only a new UUID-named research run directory and attaches results to the existing status object; it does not touch the benchmark DB or Quick RD output. Existing stop sets the stop event; the runner terminates the child, with a20-minute timeout. It does not stop the Space. No new endpoint is introduced.

Future first payload, after authorization and installation: {{"research":{{"images":["googlewiki.png"],"presets":["balanced"],"variants":["baseline","grid_skip","search","all"],"repeats":2}}}}. POST it to /api/benchmark/start using the secret header, then GET /api/benchmark/status until complete; inspect research_result and the new isolated results.json. A first run has8 timed operations plus4 warmups. HTTP request/status RTT stays separate. The job verifies manifest/input/model identity before any benchmark call and exact same-runtime files/pixels/counts/MSE/bpp before reporting success. Public requests can interfere with CPU, so use an agreed quiet interval and retain repeat variance; same-runtime exactness is necessary regardless of unresolved cross-runtime ties.

Required approval is explicit authorization to apply/stage the listed files, commit/push to the existing HF Space, configure its secret and permit the resulting rebuild/restart and bounded job. None was performed. Docker COPY means a push rebuilds the existing image. First verify source/runtime revision and actual package identity in returned metadata. Do not claim optimized HF speedups from current-deployment baseline timings.

Rollback: disable the optional branch by removing the secret (configuration changes/restarts require authorization), or restore only the two server/backend files to the recorded pre-change revision and remove only staged research job files, preserving databases/user work. Git destructive cleanup is unnecessary. No production grid/search default is proposed by this benchmark-only patch.

## UI and promotion limits

server.py still calls float(result.mse), so public compression must keep compute_final_mse=True. Only metric-free private/file benchmark modes opt out; independently decoded RGB MSE remains available in every research row. A future UI that computes the metric elsewhere still pays that work. Do not advertise the full skip-MSE speedup for an unchanged metric-producing UI. Grid plus search are candidates for UI engineering consideration after separate same-runtime HF validation; their unsafe global experimental hooks must remain isolated or be replaced by explicit per-encode ownership for concurrent production use. No PBC3.1 RD release follows from bit-exact runtime gains.
'''
    (OUT/'PBC_HF_RUNTIME_VALIDATION.md').write_text(body,encoding='utf-8')


def matched_report():
    conclusions=read('conclusions')
    rows=read('matched_rate_comparisons')
    summary=read('matched_rate_summary')
    compact=[]
    for r in rows:
        c,n=r['candidate'],r['nearest_fresh']
        compact.append(dict(dataset=r['dataset'],image=r['image'],candidate_bpp=c['bpp'],candidate_mse=c['mse'],
            control=n['label'],control_bpp=n['bpp'],control_mse=n['mse'],gap=r['rate_gap_pct'],
            mse_pct=r['direct_mse_pct'],close5=r['within_5pct'],strict_win=r['direct_strict_win'],
            lower_gap=r['fresh_below_gap_pct'],upper_gap=r['fresh_above_gap_pct'],qhq=r['qhq_region'],
            strongest_close=r['strongest_close_mse_pct'],dominated=r['dominated_by_measured_control']))
    diagnostics=read('regression_diagnostics')
    regressions=[]
    for r in diagnostics:
        regressions.append(dict(image=r['image'],types=r['selected_types'],retained_gain=r['changed_mean_retained_gain'],
            retained_bits=r['changed_mean_retained_raw_bits'],ordinary_count=r['ordinary']['iterative_patches'],
            integrated_count=r['integrated']['iterative_patches'],fixed_mse=r['fixed']['mse'],integrated_mse=r['integrated']['mse'],
            ordinary_mse=r['ordinary']['mse'],ordinary_bpp=r['ordinary']['bpp'],integrated_bpp=r['integrated']['bpp'],
            first_payload=r['first_changed_stream_patch']))
    confirmation=read('matched_confirmation')
    confirmation_summary=__import__('analyze').summaries(confirmation)
    body=f'''# Frozen allocator matched-rate validation

{conclusions['allocation_verdict']}

## Frozen policy and evaluation procedure

The exact Pass3 selector at weight0.4, Quality500k/35, remains unchanged. Same production model, custom RNG/seed, preparation, initialization, proposal/model policy, stop rule, utility and syntax. No weights, working resolution or candidate parameters were tuned on the holdout. Archived Pass3 source/model/input hashes are verified; fresh candidate confirmation must reproduce every saved Pass3 hash. The candidate's exact post-LZMA bpp is the target used to choose ordinary controls; their measured MSE is never used to choose the next configuration.

prior_rate_overlap.json records all25 candidate points, nearest existing nondominated measured controls on either side, absolute/relative gaps, bracket width, nearest500k control and observed dominance. Prior gaps are often wide; a measured bracket is not proof that its chord is achievable. The new protocol was frozen before its results in PLAN.md: ordinary Quality500k budgets10/25 first; then bitrate-only bounded interpolation/bisection in budgets3..50 using available20/35 controls, and only q0.7/0.9 at35 if discrete budget bracketing offers no unused point. At most six fresh settings/image, stopping at a3% point where possible. Every evaluated ordinary setting is retained, including nonmonotonic/worse results; no prefixes or new allocator adaptation are used.

106 fresh ordinary settings were encoded across the25 images. Their individual exploratory timings are not repeated speed evidence. The selected nearest fresh setting and the frozen allocator were then re-encoded with complete per-config warmups and three interleaved repeats. Fresh smaller budgets can change search-q interpolation, model step features and the complete trajectory. Their lower compute cost is real and must not be confused with reducing allocator-only overhead at equal35-budget work.

## Direct measured points and remaining gaps

Positive gap means the fresh control uses more bpp than the candidate. Negative MSE% favors the candidate; it is a measured-point difference, not exact equal-rate quality. Close5 means abs(control_bpp/candidate_bpp-1)<=5%, using the candidate as denominator. A5.0021% gap remains outside the5% set. Strict wins require both no higher MSE and no higher complete bpp, with one strict inequality. Missing matches are not wins.

{table(compact,[('Set','dataset'),('Image','image'),('Candidate bpp','candidate_bpp'),('Candidate MSE','candidate_mse'),('Fresh control','control'),('Control bpp','control_bpp'),('Control MSE','control_mse'),('Rate gap %','gap'),('MSE %','mse_pct'),('Within5%','close5'),('Strict RD win','strict_win'),('Fresh lower gap%','lower_gap'),('Fresh upper gap%','upper_gap'),('Q-HQ region','qhq'),('Best actual close MSE%','strongest_close'),('Dominated anywhere','dominated')])}

Candidate500k35 and control rates/raw/final bytes, decoded metrics, configs/dimensions/counts and source/model/input identifiers remain in matched_controls.jsonl, matched_rate_comparisons.json and matched_confirmation.jsonl. New below/above endpoints are full raw rows, not just rounded values. Direct closest-fresh and strongest-actual-close comparisons are separate: strongest-close chooses lowest MSE among all prior/new measured controls within5%, retaining its signed rate gap. A candidate can beat a poor fresh control yet be dominated by an older point at a different rate; mario64/pigeon demonstrate this.

Aggregate coverage and direct/interpolated results. qhq_bestclose requires a strongest actual close control; its nearest-fresh columns may include a gap outside tolerance and are descriptive, while the best-close columns are direct tolerance comparisons:

{table(summary,[('Set','dataset'),('Subset','subset'),('Fresh cases','cases'),('Fresh MSE %','direct_mse_pct'),('Fresh median %','fresh_direct_median_pct'),('Fresh MSE wins','direct_mse_wins'),('Fresh strict wins','direct_strict_wins'),('Abs rate gap %','mean_abs_rate_gap_pct'),('Best actual close coverage','strongest_close_mse_pct_coverage'),('Best close MSE %','strongest_close_mse_pct'),('Best close median %','strongest_close_median_pct'),('Best close wins','strongest_close_mse_pct_wins'),('Prior interpolation %','previous_matched_mse_pct'),('Densified interpolation %','densified_matched_mse_pct'),('Conservative interpolation %','strongest_matched_mse_pct')])}

The all subset's fresh measured-point average includes gaps outside tolerance and is descriptive only; close3/close5 are the meaningful direct tolerance subsets. qhq_close5 restricts to candidate rates between actual released Quality/HQ endpoints. Strongest-close coverage can exceed fresh-close coverage because an older point provides a match. The best actual close comparison still has rate mismatch and cannot define a continuous RD curve. Main/holdout arithmetic per-image percent means and medians are separate; no image-size or compressed-byte weighting is hidden.

{conclusions['matched_interpretation']}

## Interpolation and comparison with Pass3

The prior strongest-envelope estimates reproduce Pass3 exactly: main-5.1601% over20, holdout-9.2590% over5. Densified neighboring nondominated controls and exact weights are retained separately. Adding a measured point above an old interpolation chord can raise the interpolated baseline MSE, even when no old measured point worsens. This can make apparent candidate gains larger without a new codec benefit. It occurs notably for old_man_standing: the sparse prior chord bridges a large quality discontinuity, while new low-budget controls at nearby bpp remain poor. Therefore the conservative comparison uses the lower baseline MSE from the original and densified interpolants per image, never presenting a weakened old chord as a stronger control. These are still estimates, not achievable extra bitstreams or a lower-bound proof. No extrapolation or sparse BD-rate claim is used.

The ordinary500k-only dense curve, full prior/all-current-release envelope and actual close controls answer different questions. Their coverage/signs must not be pooled. A mean gain over all rates does not establish high-quality holdout performance; only one holdout image has an actual close control inside the released Q-HQ rate region, and it is essentially neutral/slightly worse against the best prior nearby ordinary point. The second target-region holdout case remains just beyond the5% tolerance after the bounded search.

## Named regression mechanisms

{table(regressions,[('Image','image'),('Selected types','types'),('Changed retained gain','retained_gain'),('Changed retained raw bits','retained_bits'),('Ordinary patches','ordinary_count'),('Integrated patches','integrated_count'),('Ordinary MSE','ordinary_mse'),('Integrated MSE','integrated_mse'),('Fixed-geometry MSE','fixed_mse'),('Ordinary bpp','ordinary_bpp'),('Integrated bpp','integrated_bpp'),('First changed payload','first_payload')])}

{conclusions['regression_interpretation']}

regression_diagnostics.json records every same-state original/alternative/selected cell size, maximum/actual precision, local clipped working-channel SSE gain, raw bits, utility scale and stop, and reproduces the frozen final file on all four cases. The fixed-geometry Pass3 counterparts separate a representation-only trajectory ablation from reoptimized later geometry; they do not isolate a unique causal percentage for every final-RGB loss. Coarser grids and lower precision visibly reduce the chosen correction's capacity, but missing fine detail, color/resizing and later overlap cannot be assigned separate causal shares from local SSE alone. Raw-bit proxy calibration is not final entropy coding: eiffel even grows both raw/final size after an altered trajectory, despite every local selector optimizing its own proxy. No per-candidate LZMA is introduced.

## Full latency at the selected actual points

{table(confirmation_summary,[('Set','dataset'),('Configuration','label'),('MSE','mse'),('bpp','bpp'),('Raw bytes','raw_bytes'),('Patches','iterative_patches'),('Encode s','encode_seconds'),('Encode CV','encode_seconds_mean_cv'),('Decode s','decode_seconds')])}

These controls often request far fewer patches than35. Any allocator encode penalty against them includes additional sequential search/model/action work, not just the price of alternative representations. Equal-budget incremental costs and exact cost optimizations are evaluated separately in PBC_ALLOCATOR_EFFICIENCY.md. Runtime improvements are not counted as RD improvements.

## Acceptance and next research

{conclusions['allocation_gate']}

The observed finite frontier expands on useful individual images, with actual close-rate wins; this is more evidence than a static local utility or sparse interpolated mean. It remains a content-dependent alternative rather than a universally better preset. Extreme-compression defaults are unchanged and no universal low-bpp preservation claim is made for applying the allocator everywhere. Four known regressions remain explicit, broad high-quality holdout generalization is unresolved, and25 previously studied images do not justify an adaptive classifier. Another broad fixed-weight sweep is not the next experiment. Retain the frozen setting, direct matched controls and negative cases for a future independent-image, tightly matched validation or a single targeted geometry/representation hypothesis.

RD plots rd_hpt_data/rd_hpt_data_val (PNG/SVG) show every measured prior/new point and the frozen candidate, with5% tolerance bands. Their raw source is matched_rate_points.json/.csv. Dotted lines are guides; the JSON endpoints/weights are authoritative. No production defaults, decoder, model, RNG or previous result was changed.
'''
    (OUT/'PBC_ALLOCATOR_MATCHED_RATE.md').write_text(body,encoding='utf-8')


def efficiency_report():
    conclusions=read('conclusions')
    functions=[r for r in read('allocator_profile_functions') if r['function'] in
        ('search_boxes','extract_features','_build_action','make_patch','project_residual_to_grid',
         '_score_patch','quantize_signed','palette_generator','mask_from_values','palette_bounds','patch_bits_for')
        and r['file'].endswith(('learned_filler.py','pbc3_ops.py','pbc3_features.py'))]
    aggregates=[]
    summaries=[]
    for name in ('efficiency_screen','efficiency_full','efficiency_fast'):
        if (OUT/(name+'_aggregate.json')).exists():
            aggregates += [{**r,'experiment':name} for r in read(name+'_aggregate')]
            summaries += [{**r,'experiment':name} for r in read(name+'_summary') if r['preset']=='all']
    body=f'''# Allocator efficiency

{conclusions['efficiency_verdict']}

## Attribution and corrected profiling

All C comparisons use the same generic runtime flags for ordinary/frozen/optimized methods. Ordinary/frozen requested config is Quality500k/35, unchanged model/RNG/proposal semantics and internal MSE enabled. Additional representations are a selector cost; additional accepted/attempted steps are a different, measured consequence of the evolving trajectory. A reduction in encode time is never counted as an RD improvement.

An initial harness prefix test used startswith('all'), which also matched 'alloc'. This accidentally applied grid/search to the first allocator timing/control profile. It did not affect files or RD, because the runtime operations are exact, and it did not affect A's four mode inputs. Those timing rows are excluded and retained under invalid_runtime_label, with the cause/fix in routing_correction.json. matched_confirmation and efficiency_profile were fully rerun after replacing the condition with exact all or all_ matching. The final gates reject generic search counters in ordinary-runtime allocation rows. No old invalid comparison is presented as a cost improvement.

Corrected profile: six main images, ordinary and frozen allocator, two warmed instrumented repetitions. Function cumulative time nests other work and must not be summed; self time and calls are retained. cProfile wall times contain instrumentation overhead and are not primary latency claims. There are no cold compilation calls in the corrected profile. The production legacy selector does not add a model inference for each representation, but its changed trajectory can run more model steps overall.

{table(functions,[('Method','label'),('Function','function'),('Calls/encode','calls'),('Self s','self_seconds'),('Cumulative s','cumulative_seconds')])}

The screened trajectory changes search/model calls from30.33 to35 per encode; representation builds/exact scores from30.33 to104.83, and total patch grid/palette constructions including init from39.33 to113.83. Native resampling, quantization, masks/palettes and repeated before-SSE allocations contribute to the extra action work. Raw patch-bit calculation is small (roughly0.0004s/encode in the instrumented allocator); LZMA is not added to selection. Extra selection time is separately retained in experimental_stats, and cannot explain the whole ordinary-versus-allocator wall difference by itself.

## Two exact mechanisms

1. Projection reuse: within one unchanged box/canvas/action, cache project_residual_to_grid by cell/mode. The original and lower-precision alternatives need the same unquantized projected grid; coarser cells get a different entry. Generated palettes/quantization, delta reconstruction, exact gain, raw bit cost, alternative order and the frozen utility selector remain ordinary. No values or prepared state survive a patch mutation.
2. Before-SSE reuse: compute the identical clipped working-space pre-patch SSE once per action, reduced in the same int64 order and converted to the same float. Each representation still reconstructs its full delta, clips the after-canvas exactly as before and calculates its own after-SSE/raw bits. No gain approximation or candidate rejection is introduced. Signed canvas/delta overflow semantics are unchanged.

Fresh caches wrap the original action build and the frozen Pass3 select call, then restore functions before applying any patch. Projection arrays are read-only inputs to subsequent rint/copy/quantization operations. Per-action state/closures/dictionaries/namespace creation, reductions, allocations and restoration are included in compress. Only one-time experimental function installation/code preparation is mode setup outside the compress timer, as in the other paired variants. The global hooks are private-process research mechanisms, not a concurrent-production design. A future implementation would pass owned action state explicitly.

No extra model/feature call, proposal, random draw, representation, rate weight, LZMA evaluation or stream syntax was added. Only these two methods and their composition were screened. No third optimization was pursued. Decision basis, exactness requirements and the pre-timing promotion gate are recorded in cost_decision.json.

## Exactness and uninstrumented timing

Twenty-four six-image correctness cases (four allocator variants) first matched the frozen Pass3 files/decoded pixels/patch signatures/MSE/bpp and selection counters. Primary screening is six images/five methods/three interleaved warmed repetitions,90 rows. Claimed exact variants must preserve all original/selected raw-bit and local-gain totals, changed-representation counts, stop counts and model-action counts, not merely final aggregate quality.

{table(aggregates,[('Experiment','experiment'),('Set','dataset'),('Candidate','candidate'),('Baseline','baseline'),('Cases','cases'),('Encode base s','baseline_seconds'),('Encode s','candidate_seconds'),('Encode %','encode_pct'),('Encode wins','wins'),('Byte-equal','bytes_equal'),('Pixel-equal','pixels_equal')])}

Full operating points and repeat variability where measured:

{table(summaries,[('Experiment','experiment'),('Set','dataset'),('Method','label'),('MSE','mse'),('bpp','bpp'),('Patches','iterative_patches'),('Encode s','encode_seconds'),('Encode CV','encode_seconds_mean_cv'),('Decode s','decode_seconds')])}

{conclusions['efficiency_interpretation']}

The primary overhead is the paired same-stage difference allocator_seconds-ordinary_seconds, with its relative percentage. It includes real changed trajectories, not just local microbenchmarks. Optimized-versus-frozen isolates the exact cache benefit because all files and choices remain identical. A negative/near-zero net difference on a small image is not evidence that the extra representations are free. Per-image wins/losses/CV/extrema, cache-hit counts and original extra_selection_seconds remain in raw rows and paired summaries. Micro/function savings alone are not a substantial whole-codec improvement.

{conclusions['fast_interpretation']}

Projection reuse holds at most the original/coarser projected arrays for the current action; it adds short-lived memory rather than a whole-image cache. Before-SSE reuse retains a scalar. It avoids repeated full-box clipping/subtraction/int64 square/reduction temporaries on cache hits, while after-SSE still allocates normally. Existing full-image search maps and NumPy bit-grid temporaries belong to generic runtime work and are not attributed to allocator cost reduction. No whole-process peak-RSS saving is claimed from avoided operation counts.

## RD and release implications

{conclusions['efficiency_gate']}

Exact output preservation retains both the allocator's measured wins and its content regressions. Runtime work cannot fix mario64/pigeon dominance, eiffel's rate/quality loss or limited high-quality holdout coverage. No new fixed weight, training, expanded action search, model or representation was tried. Production defaults, stream version, decoder and public UI remain unchanged.
'''
    (OUT/'PBC_ALLOCATOR_EFFICIENCY.md').write_text(body,encoding='utf-8')


def summary_report():
    c=read('conclusions')
    rt=[r for r in read('runtime_full_aggregate') if r['candidate']=='all' and r['baseline']=='baseline']
    matched=read('matched_rate_summary')
    integrity=read('initial_integrity')
    body=f'''# PBC Pass4 summary

{c['summary_verdict']}

## What ran

- Runtime:100 image/preset exactness pairs,2612 complete proposal/RNG checks,16 RGB/RGBA/mutation-path checks; six-image four-mode screen; all25 images/four presets/four modes/three interleaved repeats (1200 primary rows).
- HF: public health/OpenAPI/idle-status GETs, reported repository/runtime metadata and nine successful immutable source/dependency text checks. Existing Space only; no remote encoding or benchmark start. Authenticated bounded existing-API integration is supplied as an unapplied two-file patch plus four private job files.
- RD: full prior bracket/dominance analysis,106 new fresh ordinary-codec points selected by bitrate alone (at most six/image), then nearest fresh controls and the frozen weight0.4 allocator on all25 images/three repeats (150 rows). No prefixes, new weights, holdout tuning or model changes. Four named regressions have exact selector ledgers and existing fixed-geometry ablations.
- Cost: corrected six-image ordinary/frozen cProfile (24 rows), two exact caches and their composition;24 initial exactness cases and90 uninstrumented screening rows. {c['efficiency_scope']}

The immutable released source is research_pass1/release at2e59f1b06980aa467da0cde94fa33c3f9a0d4d51. Root remains3e9e8c94b7497ecefc5e482a66fde697dfad6e19. Current root has a different RNG; no current disabled-feature configuration is mislabeled released PBC. Existing default-output equivalence of old_pbc remains protected historical evidence, not source identity. Original Pass1/2/3 raw baselines are reused transparently; no stale database result establishes a new speed claim.

All primary timings are local, single-core/numerical thread, warmed and repeated; encode/decode calls are timed separately from loading, original-resolution RGB scoring, parsing, hashing, disk/IPC and HTTP. Complete post-LZMA bits/source pixels determine bpp. Raw bytes, actual init/iterative counts/dimensions, expanded configs and source/model/input/environment/module identity are retained. Invalid initial allocation timing/profile rows from a discovered mode-prefix bug are archived/excluded; corrected reruns and automatic guards prevent mixing generic runtime flags into allocator-only comparisons.

## Main numerical findings

{table(rt,[('Set','dataset'),('Encode baseline s','encode_seconds_baseline'),('Encode all s','encode_seconds_candidate'),('Encode %','encode_seconds_pct'),('Decode baseline s','decode_seconds_baseline'),('Decode all s','decode_seconds_candidate'),('Decode %','decode_seconds_pct'),('Exact cases','exact')])}

{table([r for r in matched if r['subset'] in ('all','close5','qhq_close5')],[('Set','dataset'),('Subset','subset'),('Fresh close/sample count','cases'),('Fresh measured MSE %','direct_mse_pct'),('Fresh wins','direct_mse_wins'),('Best actual close coverage','strongest_close_mse_pct_coverage'),('Best close MSE %','strongest_close_mse_pct'),('Best close wins','strongest_close_mse_pct_wins'),('Conservative interpolated MSE %','strongest_matched_mse_pct')])}

The all subset's nearest-fresh statistic includes points outside5% tolerance and is descriptive. Strongest-close selects the best measured MSE within5% among old/new controls and retains rate gaps; it is not exact equality or a continuous frontier proof. Different subsets/coverage cannot be combined. {c['matched_interpretation']}

{c['efficiency_verdict']}

## Explicit research answers

1. All three runtime mechanisms are compatible and byte/pixel/patch/RNG exact on the tested local cases, including all100 default cases and all primary repeats. Decode remains v0 and AI-free.
2. Full local combined encode is-25.48% main/-26.59% holdout; decode-40.57%/-40.24%. Exact MSE/bpp/patches do not change. Per-preset data, extrema and repeats are retained.
3. Search reuse adds a directly measured-8.32% main/-7.86% holdout encode relative to grid+skip. Cache alone is-7.37%/-5.82% versus current. Grid+skip is-18.72%/-20.32%; no speed percentages are added. Decoder gains come from grid I/O; cache-only decode differences are noise.
4. Existing HF has a real benchmark API, but it cannot select these undeployed modes or a bounded input/preset subset. Its normal PBC minimum is80 operations, persistent DB appends; Quick RD overwrites shared results and lacks independent decode timing/scoring.
5. Authorized current deployment checks were health/schema/status/source inspection. A full existing-codec suite is technically available but exceeds the small-smoke scope and cannot validate optimized code. Upload-based compression paths were not used.
6. No valid remote codec timing was obtained. Recorded HTTP GET durations measure requests only. No current-root remote time is relabeled optimized or compared numerically with local seconds.
7. The minimum prepared extension is proposed_hf_benchmark.patch (server.py and pbc3_benchmark.py) plus four staged job/manifest/candidate files. It reuses existing start/status/stop with a disabled-by-default secret and static bounds, executes only in a private child, appends isolated research files, and changes no production codec/UI/default. Applying/committing/pushing, secret configuration, rebuild/restart and remote job execution require explicit future authorization on the existing Space.
8. Public UI remains unchanged with internal MSE enabled; server's float(result.mse) remains valid. Private/file-only paths may skip and independently score. No metric-producing UI speedup is claimed from omitting a metric it still needs.
9. Weight0.4 survives many genuinely close actual controls as a content-dependent alternative: nearest fresh5%-matched main13/16 MSE wins, holdout4/4; best measured close old/new main16/19 and holdout3/4 wins. Signed rate gaps and strict RD wins remain separate. Being better than one fresh control is not sufficient if an older point dominates.
10. Fresh matching covers16/20 main and4/5 holdout at5%;14/20 and4/5 at3%. Reusing existing measured close controls expands coverage to19/20 main,4/5 holdout. field remains unmatched; ducks is5.0021% from its nearest fresh control, just outside tolerance. Fresh-only unmatched main images are field,flower,flowers,monkey. No additional unbounded search was performed.
11. Earlier prior interpolation is reproduced(-5.16% main/-9.26% holdout). The conservative old/densified estimate is-3.05%/-8.34%, with all20/all5 overlaps, but remains interpolation. Actual close-control means answer a different question and cannot be substituted for these percentages. Newly measured inefficient points can expose an old chord's optimistic shape; a weaker new chord is not automatically a stronger baseline.
12. Some main Quality-to-HQ finite operating points improve against actual close controls; broad high-quality holdout improvement is unproven. Main fresh-close Q-HQ subset has12 cases/9 MSE wins. Only one holdout Q-HQ case has a5% actual match, essentially neutral/slightly worse than the strongest nearby point. Candidate rates outside that interval support only an all-rate conclusion.
13. Gains are content-dependent. Two main candidates and one holdout candidate are dominated by an observed control somewhere on the measured curve. Large direct gains against low-budget500k controls do not repair mario64/pigeon losses versus an existing lower-resolution point. Five holdout images do not establish robust content generalization.
14. {c['regression_interpretation']}
15. Another fixed-weight sweep is not justified. Raw proxy/local-SSE, later geometry/stop behavior and original RGB/final entropy interact. Content-aware allocation is a hypothesis requiring independent data; no classifier was implemented. The experiment does not isolate a fundamental representation-capacity limit.
16. {c['incremental_cost_answer']}
17. {c['exact_cost_answer']}
18. Exact cache variants preserve all frozen allocator decisions/files, so their RD advantages and failures remain unchanged; this is runtime attribution, not a new RD result.
19. {c['fast_interpretation']}
20. Allocation remains plausible research for PBC3.1, but is not an accepted all-metric release improvement. Content regressions, incomplete high-quality holdout coverage, cost and undeployed HF behavior remain gates.
21. {c['pass5_answer']}
22. Bit-exact NumPy grid I/O and exact search reuse are ready for production review/HF consideration with explicit owned state and same-runtime validation. Existing MSE opt-out is suitable only for callers that do not require its value. No candidate has authorization for production installation here.
23. Stop broad fixed-weight/entropy/RL sweeps, shipping expensive fixed-geometry replay, and claiming whole-codec gains from local utility/microtimings. Do not introduce a classifier, broad native rewrite or PBC4 representation until the measured frontier/capacity evidence justifies it. {c['stop_answer']}

## Classification and next steps

{c['classification']}

{c['pass5_answer']}

Observations are actual decoded files/metrics, direct measured rates, proposal/RNG checks and complete call walls. Causal evidence is paired runtime flags, frozen exact cache output, and existing fixed/evolving geometry ablations. Attribution of final-RGB losses to fine detail versus precision/entropy beyond these diagnostics remains a hypothesis. No improvement is accepted merely because a proxy looks efficient.

## Files and preservation

All new work is under research_pass4/. The four required reports, raw CSV/JSON/JSONL, direct/interpolated comparisons and brackets, source plots, exactness records, API inventory/source checks, stage metadata, proposed patch and isolated helper copies are listed exhaustively in changed_files.txt; artifact_manifest.json records file sizes/SHA256. Caches are excluded from the measurement inventory. README.md lists reproduction commands and every artifact's role. Invalid runs remain clearly isolated and excluded.

Initial integrity verified{integrity['artifacts']} prior artifacts. final_checks.py checks them again, plus production/model/input hashes, unchanged commit/status/user work, determinism, exact streams/decisions, candidate packaging and unapplied-patch applicability. Before/after git status and snapshots are retained. Production codec/server/backend/defaults/model/datasets/databases and prior research remain unchanged. No commit, push, deployment, restart, synchronization, training, source/image/stream upload or remote benchmark occurred. Remote activity consisted only of authorized read-only public GETs. No PBC3.1 release is declared.
'''
    (OUT/'PBC_PASS4_SUMMARY.md').write_text(body,encoding='utf-8')


if __name__=='__main__':
    runtime_report()
    matched_report()
    efficiency_report()
    summary_report()
    for name in ('PBC_HF_RUNTIME_VALIDATION.md','PBC_ALLOCATOR_MATCHED_RATE.md','PBC_ALLOCATOR_EFFICIENCY.md','PBC_PASS4_SUMMARY.md'):
        path=OUT/name
        text=path.read_text(encoding='utf-8')
        for old,new in (('Pass1','Pass 1'),('Pass2','Pass 2'),('Pass3','Pass 3'),('Pass4','Pass 4'),
            ('PBC3.1','PBC 3.1'),('PBC4','PBC 4'),('Quality500k','Quality 500k'),
            ('weight0.4','weight 0.4'),('a20-minute','a 20-minute'),('a3%','a 3%'),('a5%','a 5%'),('a5.0021%','a 5.0021%'),
            ('by250k','by 250k'),('for250k','for 250k'),('Runtime:100','Runtime: 100'),('q0.7/0.9','q 0.7/0.9')):
            text=text.replace(old,new)
        text=re.sub(r'\b(at|within|over|are|is|was|has|have|from|to|versus|by|of|below|above|the|with|or|for|and|on|remaining|remains|verified|retains|retaining|every|all|full|only|before|after|roughly|exactly|main|holdout|repeats|cases|calls|steps|actions|patches|budgets|Python|Pillow)(?=[+-]?\d)',r'\1 ',text)
        text=re.sub(r'(?<=[A-Za-z])-(?=\d)',' -',text)
        text=re.sub(r'(?<=\d)(?=MB\b|ms\b|s\b|bpp\b|MSE\b)',' ',text)
        text=re.sub(r'([,;])(?=\S)',r'\1 ',text)
        path.write_text(text,encoding='utf-8')
