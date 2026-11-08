import json, pathlib, statistics
R=pathlib.Path(__file__).resolve().parents[2]
P=R/'research_pass35'
bas=json.loads((P/'outputs/baseline/baseline_results.json').read_text(encoding='utf-8-sig'))
cal=json.loads((P/'outputs/calibration/calibration.json').read_text(encoding='utf-8-sig'))
valid=json.loads((P/'outputs/profile/validation.json').read_text(encoding='utf-8-sig'))
over=json.loads((P/'outputs/profile/overhead.json').read_text(encoding='utf-8-sig'))
manifest=json.loads((P/'SOURCE_MANIFEST.json').read_text(encoding='utf-8-sig'))
rows=bas['rows']
lines=['| Image | Point | qcolor | AVIF bytes | bpp | RGB MSE | Encode median (s) | Decode median (s) |','|---|---:|---:|---:|---:|---:|---:|---:|']
for x in rows:
    lines.append(f"| {x['image']} | {x['point']} | {x['qcolor']} | {x['ref_bytes']:,} | {x['bpp']:.4f} | {x['rgb_mse']:.3f} | {x['median_encode_seconds']:.3f} | {x['median_decode_seconds']:.3f} |")
table='\n'.join(lines)
cal_lines=['| Image | qcolor grid | File rates (bpp) |','|---|---|---|']
for name in ['googlewiki.png','mario64.png','flower.jpg','cat.jpg','coast.jpg']:
    rr=[x for x in cal['rows'] if x['image']==name]
    qs=', '.join(str(x['qcolor']) for x in rr)
    rates=', '.join(format(x['bpp'],'.3f') for x in rr)
    cal_lines.append(f'| {name} | {qs} | {rates} |')
cal_table='\n'.join(cal_lines)
ov8=statistics.median(x['overhead_fraction'] for x in over['rows'] if x['speed']==8)*100
ov6=statistics.median(x['overhead_fraction'] for x in over['rows'] if x['speed']==6)*100
summary={'pass':'35','branch':'research/experiments','starting_head':'3723dc14de88c6c3dcbd98dd80a097e019a92a9f','libaom_commit':manifest['upstream_sources']['libaom']['commit'],'libavif_commit':manifest['upstream_sources']['libavif']['commit'],'gate_a':'passed','gate_a_evidence':['Pinned reference and research Release builds completed','avifenc and avifdec built; static libaom linkage verified','Ten speed-8 reference/research-off outputs byte-identical','Seven instrumented speed-8/speed-6 outputs byte-identical to reference','Fresh avifdec decode passed; Pillow AVIF frontend RGB matched on all seven profile outputs','Active partition and intra candidate counters nonzero; inactive RD/ML branches identified'],'gate_b':'not_run','algorithmic_encoder_modification':None,'instrumentation_patch':'patches/libaom-profile-counters.patch','development_images':[x['name'] for x in manifest['assets'] if x['role'].startswith('development')],'held_out_images_not_encoded':[x['name'] for x in manifest['assets'] if x['role']=='held-out'],'reference_equivalence':{'speed8_points':len(rows),'all_bytes_identical':all(x['bytes_identical'] for x in rows),'profile_cases':len(valid['rows']),'all_profile_bytes_identical':all(x['reference_bytes_identical'] for x in valid['rows']),'all_profile_decodes_passed':all(x['fresh_avifdec_ok'] and x['pillow_independent_decode_ok'] for x in valid['rows'])},'median_profile_overhead_percent':{'speed8':round(ov8,2),'speed6':round(ov6,2)},'commits_or_pushes':'none','worktree_status':'uncommitted research_pass35 changes'}
(P/'PASS35_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(P/'README.md').write_text('''# PBC V4 Research Pass 35

This pass pins and builds libaom 3.12.1 with libavif 1.3.0, establishes deterministic AVIF output, and profiles the actual all-intra paths. No PBC production source or model changed. The only source patch adds compile-time-guarded profiling counters.

- [Pinned sources and image hashes](SOURCE_MANIFEST.json)
- [Build and reproduction](BUILD_AND_REPRODUCTION.md)
- [Encoder source map](AV1_ENCODER_SOURCE_MAP.md)
- [Baseline measurements](AVIF_BASELINE_RESULTS.md)
- [Search profile](AV1_SEARCH_PROFILE.md)
- [PBC integration opportunities](PBC_AV1_INTEGRATION_OPPORTUNITIES.md)
- [Pass summary](PASS35_SUMMARY.json)

Builds, source checkouts, binaries, outputs, raw logs, and measurement JSON are under ignored paths. The reviewable source-only patch is `patches/libaom-profile-counters.patch`.

Gate A passed. Gate B was not run because profiling did not justify an algorithmic modification. The work remains uncommitted and unpushed.
''',encoding='utf-8')
(P/'AVIF_BASELINE_RESULTS.md').write_text(f'''# AVIF Baseline Results

## Inputs and operating points

The five development images were EXIF-transposed and converted to RGB. Hashes and dimensions for them and the three held-outs are in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Held-outs were not encoded or used for tuning.

The bounded unmodified qcolor grid was `{', '.join(map(str,cal['qcolor_grid']))}` (qcolor 80/90/95 were added only for flower and coast). Point 1 was chosen nearest 0.27 bpp; point 2 nearest 0.50 bpp above point 1. Photo rates do not map to one universal q value: coast brackets the higher target at 0.105 and 0.648 bpp.

{cal_table}

## Fixed settings

libaom 3.12.1 via libavif 1.3.0; single frame; AOM; 8-bit; YUV 4:2:0; full range; CICP `1/13/1`; speed 8; one thread; qalpha 0; EXIF/XMP/ICC omitted; SharpYUV off; automatic tiling. bpp uses complete AVIF bytes and original pixel count.

## Reference results

Each point had one warmup encode/decode pair and three fresh-process timed pairs. Times below are medians; raw samples and hashes are in `outputs/baseline/baseline_results.json`. Decode timing includes process startup and PNG output. RGB MSE is averaged over source pixels and channels after decoding at original dimensions.

{table}

## Equivalence and compatibility

All ten speed-8 points were byte-identical between the unmodified reference and the instrumentation-off research build. The final instrumented build preserved output bytes on all five speed-8 profile cases and both speed-6 cases. Those seven files decoded in fresh `avifdec` processes and Pillow AVIF frontend processes; decoded RGB and MSE matched exactly between the two frontends. No private syntax or decoder change was added.
''',encoding='utf-8')
(P/'AV1_SEARCH_PROFILE.md').write_text(f'''# AV1 Search Profile

## Instrumentation and coverage

The profile build enables libaom `CONFIG_COLLECT_PARTITION_STATS=2` and `CONFIG_COLLECT_COMPONENT_TIMING=1`, plus `PBC_PROFILE`-guarded counters in `partition_search.c`, `nonrd_pickmode.c`, and `intra_mode_search.c`. The patch does not change decisions or bitstream code. The instrumentation-off research control rebuilt with the counters compiled out retained exact reference bytes. Raw logs, CSVs and AVIF outputs are under `outputs/profile/`.

At speed 8 the active path applies variance-based precomputed partitions through `av1_nonrd_use_partition`; the ML/non-RD partition picker `av1_nonrd_pick_partition` is inactive. Across five cases the applied pattern had 101,535 NONE, 33,864 SPLIT, 101 VERT, and 29 HORZ nodes. Hybrid intra routed 60,439 blocks to RD intra and 41,264 to non-RD intra. RD luma loops considered 3,686,779 candidates, fully evaluated 207,655, pruned 3,479,124 before full transform/RD, and selected 60,439. Non-RD luma loops considered/evaluated 165,056 candidates, with 41,264 winners and no candidate pruning in this configuration.

Speed 6 takes the RD partition path. Its per-block-size CSVs record attempts and selected partitions; totals are:

| Speed 6 image | Partition attempts N/H/V/S | Selected N/H/V/S |
|---|---|---|
| cat.jpg | 39,564 / 7,500 / 2,343 / 11,002 | 34,660 / 3,047 / 203 / 5,639 |
| googlewiki.png | 8,959 / 3,657 / 3,516 / 3,001 | 8,224 / 985 / 270 / 2,032 |

Across these two cases RD luma loops considered 4,942,891 candidates, fully evaluated 216,650, pruned 4,726,241, and selected 64,123.

## Time distribution

At speed 8, `encode_sb_row_time` was 85–90% of libaom frame time. `nonrd_use_partition_time` was 79–88%; `hybrid_intra_mode_search_time` 36–63%; variance partition setup 0.8–1.8%; bitstream finalization 5–9%. These timers nest and must not be summed. At speed 6, RD partition search was 97–99% of `encode_sb_row_time`; RD intra mode evaluation was 10–16% of internal frame time.

One warmup and three interleaved timing repetitions measured median complete-call instrumentation overhead of {ov8:.1f}% at speed 8 and {ov6:.1f}% at speed 6. The speed-6 instrumentation is highly intrusive (individual medians rose about 411% and 591%); those internal times show path ordering only, not production runtime. All outputs remained byte-identical.

## Interpretation

1. RD intra search already rejects most candidate iterations before full transform evaluation; a feature-based change must improve on existing HOG, masks, angle-RD and model-RD pruning.
2. Speed-6 RD partition search dominates internal time; any speed change must be measured with profiling off.
3. Speed-8 variance partition setup is a small part of total encode time and is a weak speed target here.

Speed-8 partition-stat CSVs are zero because that configuration bypasses RD partition candidate search. They are nonzero at speed 6; this confirms speed-specific path coverage.
''',encoding='utf-8')
(P/'AV1_ENCODER_SOURCE_MAP.md').write_text('''# AV1 Encoder Source Map

References are pinned libaom `10aece4157eb79315da205f39e19bf6ab3ee30d0` (v3.12.1). Active branches were verified by the profile report.

## Partition and call paths

- `av1/encoder/encodeframe.c:encode_nonrd_sb` chooses fixed or variance-based partitioning (`av1_choose_var_based_partitioning`) and applies it through `av1_nonrd_use_partition`. This was active at speed 8.
- `av1/encoder/partition_search.c:av1_nonrd_use_partition` reads precomputed partition types, may call `try_split_partition`, then dispatches `pick_sb_modes_nonrd` and recurses through the tree. `hybrid_intra_mode_search` selects RD or non-RD intra per block.
- `partition_search.c:av1_nonrd_pick_partition` performs ML-assisted NONE/SPLIT evaluation, but is conditional on ML partition configuration and was inactive in these runs.
- `encodeframe.c` calls `av1_rd_pick_partition` at speed 6. `partition_search.c:av1_rd_pick_partition` recursively evaluates candidates under RD bounds and pruning (`av1_prune_partitions`, best-RD limits, speed features). Speed-6 partition CSVs were nonzero; this path was inactive at speed 8.
- `av1/encoder/speed_features.c` configures frame-size, independent and qindex-dependent speed features; speed level alone does not define the active branch.

## Intra modes

- `partition_search.c:hybrid_intra_mode_search` calls `nonrd_pickmode.c:av1_nonrd_pick_intra_mode` or `rdopt.c:av1_rd_pick_intra_mode_sb`.
- Non-RD intra tests `intra_mode_list` up to `RTC_INTRA_MODES`; it estimates block modes with `av1_estimate_block_intra` and scores rate/distortion. Existing pruning includes flat top-left blocks, neighbor modes, best-SAD/H-pred rules and optional palette search.
- RD intra calls `intra_mode_search.c:av1_rd_pick_intra_sby_mode` for luma, then chroma search. It considers directional and non-directional modes and angle deltas. Existing pruning includes HOG masks, mode masks, odd-angle RD checks, model-RD/top-mode selection, then full quantized RD evaluation. Palette, filter-intra and winner-mode processing are later paths.
- Candidate costs use `RDCOST` from `av1/encoder/rd.h`, combining estimated rate and distortion under the current RD multiplier.

## Transform, quantization and pruning

- RD intra calls `av1_pick_uniform_tx_size_type_yrd` in `av1/encoder/tx_search.c`; block size and speed features constrain transform evaluation.
- Quantization dispatches through `av1/encoder/av1_quantize.c` facades such as `av1_quantize_fp_facade` and `av1_quantize_b_facade`, selecting SIMD implementations at runtime.
- The search is mixed-cost: model-RD and rate thresholds reject candidates before full reconstruction; RD partition search uses bounds; non-RD mode uses precomputed variance partitions and estimated costs.
- AV1 already provides adaptive quantization, segmentation/delta-q, skip decisions, transform choices and coefficient quantization. Pass 35 did not establish that PBC's earlier mechanisms are absent from libaom.
''',encoding='utf-8')
(P/'PBC_AV1_INTEGRATION_OPPORTUNITIES.md').write_text('''# PBC–AV1 Integration Opportunities

These are proposals only. Speed 8 spends most internal time in non-RD partition application and hybrid intra search; speed 6 spends most in RD partition search.

## 1. RD intra candidate ordering

- Existing mechanism: `intra_mode_search.c:av1_rd_pick_intra_sby_mode` applies HOG masks, mode masks, angle checks and model-RD/top-candidate pruning.
- Idea: test whether cheap PBC-like edge/texture features improve ordering or thresholds for remaining modes while leaving AV1 RD cost as final decision.
- Integration: just before `av1_pick_uniform_tx_size_type_yrd`.
- Compatibility/cost: encoder-only and standard AV1; feature cost must stay below the expensive evaluations it avoids.
- Risk/evidence: duplicate current signals or remove a useful mode. Require exact or matched-rate RD, full AVIF bytes, speed and holdout results with instrumentation off.

## 2. Speed-6 RD partition work reuse

- Existing mechanism: `partition_search.c:av1_rd_pick_partition` recurses through partition choices using RD bounds and pruning.
- Idea: identify avoidable repeated candidate/context work and evaluate a bounded reuse optimization that preserves choices.
- Integration: around recursive candidate calls and `av1_prune_partitions`.
- Compatibility/cost: encoder-only if decisions stay identical; caching adds state and memory.
- Risk/evidence: profile timers are intrusive at speed 6. Use counters to locate duplicate evaluations, then paired uninstrumented timings and byte comparisons.

## 3. Local quantizer allocation

- Existing mechanism: AV1 adaptive quantization and segmentation adjust local q/RD multipliers; transform quantization already removes low-value coefficients.
- Idea: test whether PBC's local correction-value history improves an existing AQ decision.
- Integration: AQ/segmentation or delta-q selection, without adding a decoder-side residual layer.
- Compatibility/cost: ordinary AV1 syntax; cheap if based on existing block features.
- Risk/evidence: duplicate variance/complexity AQ or trade rate for distortion. Require matched-rate RD curves and encode-time limits.

## PBC-specific assessment

LibAOM already uses variance/gradient features, HOG pruning, model-RD, mode masks, adaptive quantization, transform evaluation and coefficient quantization. PBC's potential contribution is a signal that predicts where those decisions are wrong, not another copy of them. Sparse-detail results suggest testing a saliency prior, not a sparse decoder representation.

## Small future research list

- JPEG XL offers VarDCT and Modular paths, encoder tool-selection guidance and adaptive quantization; study its tool gating as an analogy, not evidence that AV1 lacks those tools. [JPEG XL white paper](https://ds.jpeg.org/whitepapers/jpeg-xl-whitepaper.pdf).
- WebP documents block prediction, adaptive quantization, transforms and entropy coding, all established coding tools that AV1 extends. [Google WebP compression documentation](https://developers.google.com/speed/webp/docs/compression).
- PICO and PULSE are learned-codec research directions, not drop-in AVIF heuristics. PICO is a learned perceptual codec; PULSE explicitly studies a low-complexity neural receiver. Their decoder-side model requirements do not fit standard-decoder-compatible Pass 35 changes. [PICO research repository](https://github.com/apple-aiml-research/ml-pico), [PULSE paper](https://arxiv.org/abs/2609.18602).
''',encoding='utf-8')
