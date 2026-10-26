# PBC Pass 3 summary

The combined performance optimization and exact rectangle error-map reuse are confirmed local runtime improvements with identical output. Rate-aware allocation at 500k/35, weight 0.4, is a credible measured RD research lead that survives this holdout under interpolated existing-frontier comparisons. It is not a release improvement: encoding remains slower than released HQ, regressions persist, and most holdout matches are not tight actual-rate pairs. Weight 0.8 does not generalize convincingly.

All three objectives were evaluated independently. No performance flag is combined with the allocation or reuse candidates. The authoritative immutable release is `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51`; current root remains `3e9e8c94b7497ecefc5e482a66fde697dfad6e19`. Model SHA-256: `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`. Old_pbc's verified default-output equivalence remains historical evidence; it is not relabeled source-identical release or redundantly fully benchmarked.

## What ran

- A: baseline/grid/skip/both, six main images/all four presets/three repeats; exact edge/API/alpha/malformed tests; then all 25 images/all presets/three repeats (1,200 primary rows). Full confirmation uses the exact unapplied integration copy's dispatch path. Allocation/startup probes and a local portable-HF-harness smoke check are separate.
- B: 30 frozen trajectories (six images, release/current Q250/HQ250, current 500k35), per-patch exact working cost/gain and sampled RGB/final-size deltas; a bounded four-representation static study; fixed-geometry identity replay (18 controls); three weights across three controls, fixed and integrated screens (144 rows each). Two 500k35 integrated settings and their fixed-geometry counterparts, with five fresh release/current controls, were frozen before new holdout results and run over 20 main/five holdout images/three repeats (675 rows).
- C: two reuse mechanisms; 48 initial exactness cases, 1,178 proposal-list/RNG checks plus full maps/totals/files, signed-int16 clipping/wrap and RGB/RGBA/warmup/owner tests. One fixed candidate is confirmed on all 25 images/four presets/three repeats (600 rows), plus separate component traces and explicit prototype-setup accounting.

All primary rows include original-size independently decoded RGB MSE, complete post-LZMA bytes/bpp, raw bytes including the six-byte outer wrapper, encode/decode call walls, actual patches, dimensions, expanded config, source/model/input hashes and module paths. `bpp = 8*complete_file_bytes/original_pixels`. NumPy error differences/squares are safe and reduced in float64. Image loading, independent metrics, inspection and IPC stay outside ordinary compress/decompress timers. Extra representation/replay/reuse work is charged as documented. One CPU core/numerical thread, warmed paths, rotated order and retained three-repeat variation. No local/HF timing mixture or pristine cold-start claim.

## Strongest numerical results

Combined runtime optimization versus unchanged current root, exactly identical output:

| Set | MSE | bpp | Encode baseline s | Encode combined s | Encode % | Decode baseline s | Decode combined s | Decode % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | 193.4631 | 0.2793 | 0.4686 | 0.3804 | -18.8176 | 0.0891 | 0.0525 | -41.0986 |
| hpt_data_val | 91.1562 | 0.7189 | 0.3732 | 0.2980 | -20.1445 | 0.0753 | 0.0454 | -39.6669 |

Integrated allocation operating points:

| Set | Candidate | MSE | bpp | Raw bytes | Iterative patches | Encode s | Decode s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | q500p35_integrated_t0.4 | 138.4631 | 0.3211 | 117060.8500 | 34.2500 | 0.8055 | 0.1125 |
| hpt_data | q500p35_integrated_t0.8 | 153.4165 | 0.2692 | 89977.6500 | 34.5500 | 0.7859 | 0.1021 |
| hpt_data_val | q500p35_integrated_t0.4 | 59.0152 | 0.6975 | 79048 | 29.6000 | 0.6047 | 0.0854 |
| hpt_data_val | q500p35_integrated_t0.8 | 72.9044 | 0.7059 | 57724 | 30.2000 | 0.6101 | 0.0787 |

Paired with ordinary current 500k35:

| Set | Candidate | MSE % | bpp % | Encode % | Decode % | Strict pointwise RD wins |
| --- | --- | --- | --- | --- | --- | --- |
| hpt_data | q500p35_integrated_t0.4 | 7.5932 | -20.3871 | 5.9179 | -4.1755 | 2 |
| hpt_data_val | q500p35_integrated_t0.4 | 34.3298 | -31.5992 | -0.4926 | -8.0740 | 0 |
| hpt_data | q500p35_integrated_t0.8 | 19.2128 | -33.2372 | 3.3426 | -13.0821 | 0 |
| hpt_data_val | q500p35_integrated_t0.8 | 65.9443 | -30.7802 | 0.3969 | -15.2412 | 0 |

Matched-rate comparison with all previously observed release/current controls:

| Set | Candidate | Overlapping images | MSE % | Wins | Within 5% actual rate | Near-rate MSE wins | Matched-MSE overlap | Matched-MSE bpp % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | q500p35_integrated_t0.4 | 20 | -5.1601 | 17 | 7 | 6 | 19 | -8.4129 |
| hpt_data | q500p35_integrated_t0.8 | 20 | -1.8897 | 13 | 4 | 3 | 20 | -1.9691 |
| hpt_data_val | q500p35_integrated_t0.4 | 5 | -9.2590 | 4 | 1 | 1 | 5 | -5.0251 |
| hpt_data_val | q500p35_integrated_t0.8 | 5 | -0.0535 | 3 | 0 | 0 | 5 | 3.1366 |

Specifically inside release Quality-to-HQ neighbor brackets:

| Set | Candidate | Q–HQ overlapping images | MSE % | Wins |
| --- | --- | --- | --- | --- |
| hpt_data | q500p35_integrated_t0.4 | 15 | -11.2574 | 13 |
| hpt_data | q500p35_integrated_t0.8 | 11 | -7.7061 | 9 |
| hpt_data_val | q500p35_integrated_t0.4 | 2 | -10.3426 | 2 |
| hpt_data_val | q500p35_integrated_t0.8 | 1 | -13.5312 | 1 |

Exact search reuse versus the identical current root:

| Set | Encode baseline s | Encode reuse s | Encode % | Encode wins | Decode % |
| --- | --- | --- | --- | --- | --- |
| hpt_data | 0.4798 | 0.4472 | -6.7971 | 72 | -0.5558 |
| hpt_data_val | 0.3788 | 0.3561 | -5.9865 | 17 | 0.0772 |

All interpolated means use only valid per-image overlap; missing cases do not count as wins. Different matched-rate/distortion populations cannot be combined into a dominance claim. The stronger existing envelope is an oracle over previously measured configurations, not a deployable routing policy. Exact brackets/weights, all pointwise outcomes, outliers and source points remain in allocation_matched.json, allocation_paired.json and allocation_points.json. No extrapolation or sparse-curve BD-rate claim.

## Explicit research answers

1. Grid I/O and internal-MSE skipping combine successfully on the 100 current image/preset cases, three repeats, with complete byte/pixel/count/MSE/bpp equality.
2. Direct measurements are approximately additive within observed variation; combined_effects.json records the measured residual rather than adding percentages. Cache/allocation effects are plausible explanations for small offsets, not separately proven causes.
3. All runtime and promoted exact-reuse outputs are identical to their current controls. Allocation intentionally changes payloads and, in integrated mode, later trajectories; it is not bit-exact optimization.
4. Vectorized grids add temporary arrays and stricter private-helper argument checks. Existing malformed-header/resource limitations remain. Internal MSE=None is valid only for callers that opt out; the public server's float(result.mse) requires keeping its flag True. Global mode monkeypatching is unsuitable for concurrent public requests.
5. The private portable harness, hash manifest, isolated candidate module and unapplied one-file patch are ready for a later explicitly authorized cpu-basic test. Reference/current workers are source-isolated on the same CPU; all current modes share a warmed serial worker. It uses repository inputs and no HTTP. Public server/HTTP-overhead validation is separate and requires explicit authorization. See PBC_PERFORMANCE_INTEGRATION.md for exact files, commands, gates and rollback.
6. Expensive index grids dominate raw syntax. High Quality's sampled per-patch raw cost is roughly four times Quality's, with far more cells and higher selected precision. Final entropy-coded byte ownership is nonadditive; raw totals cannot assign a unique compressed cost per patch.
7. Some cheaper representations retain nearly the same local gain: 44/262 retain at least 95% at at most 85% raw bits. None locally dominates or meets the stricter 98%/75% shortlist. Cheap density reduction often loses much more gain than a precision step.
8. The effect is not confined to static local scores. Integrated weight 0.4 improves mean matched-rate MSE versus the measured fixed-geometry family by 5.06% on 17 overlapping main images (14 wins) and 16.61% on all five holdout images. These are sparse interpolated comparisons, not a continuous dominance proof. Both integrated settings change geometry on all 25 images. Weight 0.8 loses versus that family on average, particularly on holdout, so the effect depends on the sequential policy rather than following automatically from the proxy.
9. Against release default Q-HQ brackets, integrated weight 0.4 gives estimated MSE -11.26% on 15 main overlaps (13 wins), and -10.34% on two holdout overlaps (two wins). Against the broader strongest previously measured envelope, it gives -5.16% on all 20 main and -9.26% on all five holdout images. Seven main images have a previously measured point within 5% rate, with six MSE wins; holdout has only one such point, also a win. The observed frontier expands in a useful but content-dependent way; denser actual matched-rate controls are required to establish its size securely.
10. The weight-0.4 interpolated all-rate signal survives: four of five holdout images improve, while pigeon.JPG regresses about 18.88% against the strongest existing envelope. Only two holdout cases fall specifically within release Q-HQ neighbor brackets, so broad high-quality generalization remains uncertain. Weight 0.8 is essentially neutral on that holdout envelope (-0.05% mean) and is not a convincing follow-up candidate.
11. The tested allocator is opt-in on Q/HQ-derived paths; Compression/Balanced defaults are not modified. No universal allocator/no-low-bpp-regression claim is made. The independent performance optimization is proven output-preserving across all four presets.
12. Large regressions are reduced in some cases but not eliminated. Weight 0.4 still has strongest-envelope matched-rate regressions of about +25.24% on mario64.png, +12.02% on monet.jpg, +10.25% on eiffel.png and +18.88% on holdout pigeon.JPG. Different rates/control envelopes make direct percentages versus Pass 2 outliers non-equivalent; do not claim those failures solved. Stronger weight 0.8 creates additional trajectory/entropy failures, including the 15k_kills raw/final-size reversal.
13. Full-channel visible error arrays can be shared with search until their channel changes. Rectangle-local error maps/totals can be updated after actual mutation; owner/stage changes require invalidation. Integrals, anchors and residual-dependent features cannot simply be reused stale.
14. Both screened reuse approaches preserve intermediate maps/totals, complete proposal order/RNG state and complete streams in verification. The promoted full-dataset candidate retains complete byte/pixel/count equality.
15. Rectangle reuse preserves all 100 default image/preset streams and decoded pixels on all three full repeats. Full encode latency, including per-call prototype state/installation/cleanup, changes -6.80% on main (72/80 case wins) and -5.99% on holdout (17/20). The average prototype installation is about 1.10 ms, included. This is a meaningful local runtime-only improvement; decoder differences are noise. It remains unvalidated on HF and requires explicit per-encode state ownership before concurrent production integration.
16. The arithmetic and avoided allocations are compatible with one CPU core, but transfer is unmeasured. HF requires same-runtime equivalence and interleaved latency/memory checks with actual versions, CPU dispatch and service load. Existing cross-runtime trajectory differences are not explained away or fixed.
17. Smarter allocation within the existing v0 format is a credible PBC 3.1 research path: one small deterministic rule improves actual-file estimated RD on main and frozen holdout, without new models, proposals or decoder syntax. It has not met the release bar because of encoding cost, content regressions, sparse matching and untested HF behavior.
18. The measured limitation is an interaction of representation cost, the one-action model's chosen geometry, ordinary early stopping and later residual states. Zero sampled cheap alternatives locally dominate the original, so simply declaring HQ precision wasted is unsupported. Fine detail still costs real SSE reduction; rate allocation can trade that reduction usefully on some trajectories. This pass does not isolate a fundamental representation-capacity ceiling or establish that a PBC 4.0 format is required.
19. First add tightly bracketing ordinary-codec controls near the frozen weight-0.4 operating points, including the difficult synthetic/texture/pigeon cases and new independent images, to test interpolation sensitivity. If that signal persists, test one output-preserving reduction in selector cost: reuse the projected residual grid when evaluating a lower-precision palette. Keep the geometry count, model, seed, rate weight and stop rule fixed for that cost ablation; do not add a new beam or retraining.
20. Prioritize PBC 3.1 runtime/allocation refinement and authorized HF validation. The existing-format lead makes a fundamental PBC 4.0 rewrite premature. Standalone decoder packaging is useful but a broad native rewrite is lower priority than validating the measured NumPy/grid and exact map improvements; PIL bicubic compatibility and the remaining native operations must be preserved in any port.

## Classification and next work

Runtime-only strict output preservation: combined grid/MSE option and rectangle reuse. Estimated observed RD lead with a latency cost: weight-0.4 integrated allocation. Operating-point rate-for-quality tradeoffs: both allocation weights versus ordinary 500k/35. Negative/inconclusive generalization: weight 0.8, universal HQ250 selection, content routing and HF transfer. Fixed-geometry replay is an expensive diagnostic, not a shipping encoder. No strict all-metric production release is declared.

Ranked Pass 4 directions:

1. With separate authorization, run the supplied hash-checked private cpu-basic benchmark for grid/skip/both; validate rectangle reuse separately in a serial worker before any production state ownership change. Keep public UI MSE enabled.
2. Validate the frozen weight-0.4 RD lead with denser matched-rate ordinary controls and new images, then independently reduce its extra action-evaluation cost through projection reuse. Retain the named regression cases.
3. After those independent gates, test a controlled composition of the proven runtime changes with the allocation candidate. Measure the combination directly; do not add speed percentages or claim that runtime improvements repair RD regressions. A universal classifier, broader model or new stream representation is not the next justified step.

Observations are the complete-file/metric/latency tables and exactness checks. Causal evidence comes from paired flag/control comparisons, same-state alternative representations and fixed-versus-evolving geometry ablations. Explanations involving entropy predictability, model distribution shift or future action reuse remain hypotheses unless directly measured. A new representation, retraining, broader beams and a universal content classifier are not justified merely by a proxy improvement.

## Reproduction and repository state

README.md lists exact commands and artifact roles. The four requested reports, raw CSV/JSONL, plots/source points, quantized-grid diagnostics, equality records, provenance, proposed patch and portable benchmark all reside under research_pass3/. changed_files.txt lists every created file; artifact_manifest.json gives byte lengths and SHA-256. Cache files are excluded from the measurement inventory.

Before/after git status and source/model/dataset snapshots are retained. final_checks.py verifies all protected Pass 1/2 artifacts, source/model/input hashes, repeated determinism, stream files, error accounting, fixed-geometry invariants, paired exactness, integration applicability and packaging smoke. No production source/default/model/public behavior was changed. Existing .gitignore/user work is preserved. No commit, push, synchronization, deployment, restart, image upload, remote compression request or training occurred. No PBC 3.1 release is declared.
