# PBC Pass 2 research summary

All four workstreams completed locally. The pass establishes matched-work engineering speedups, screens the single-stage quality/resolution frontier, rejects two tested lossless entropy transformations, and confirms fixed runtime candidates on main/holdout. It does not authorize or enable a production PBC 3.1 configuration.

## What ran

- A: 25 images × four presets × release/current-with-NumPy-RNG × three repeats (600 rows), plus separate component profiles and a bounded saved-HF-file investigation.
- B: six fixed main images, release/root, Quality/HQ, 250k/500k/1M, two repeats (144 rows); three targeted allocation ablations (36); four targeted rate settings plus fresh Q/HQ controls (72). Two fixed 500k Quality budget variants and fresh release/root Q/HQ controls were then confirmed on all 20 main and five holdout images, three repeats (450).
- C: 48 frozen release/root streams (six images × four presets × two sources), exact syntax/symbol diagnostics, two independent alternate representations, four LZMA controls/settings, three repeats (1,008 rows). Four separate dictionary-memory probes. No entropy candidate passed the screen, so none was promoted to main/holdout.
- D: three candidates on six images/four presets/three repeats (288); two fixed candidates plus current-root control on all 25 images/four presets/three repeats (900). Separate encoder/decoder component measurements, 640 bit-layout checks and integer clipping/RGBA tests. No combined variant.

Actual row counts: `{"allocation_screen": 36, "entropy_screen": 1008, "performance_components": 18, "performance_confirmation": 900, "performance_screen": 288, "quality_confirmation": 450, "rate_screen": 72, "resolution_screen": 144, "rng_control": 600, "rng_profile": 48}`. Dataset SHA-256 inventory contains 25 inputs. All expanded configs, source/model/input hashes, module paths, dimensions, actual patch counts, raw/final bytes, decoded RGB MSE and complete compress/decompress wall times are retained. See PLAN.md, README.md, initial_snapshot.json, each *_metadata.json and *_provenance.json.

All primary timings are local, warmed, single-core and single numerical thread; source/variant order rotates. Three-repeat timings are aggregated by per-case medians, with actual-call means/CV/extrema available. First-invocation warmup calls retain any cache/compilation cost; process-import startup and pristine cold starts are not measured. No HF timings are pooled with local results. `bpp = 8 × complete file bytes / original pixels`; RGB scoring uses actual decoded original-size files. The website's RGBA convention is not used for research distortion.

## Measured engineering results

With release-compatible RNG and exactly identical outputs, current root retains encode -10.29% and decode -19.31% overall. All 100 distinct cases match on all three repeats. This is the combined existing implementation benefit, not attribution to one change.

Independent D confirmation against unchanged current root:

| Set | Variant | Cases | MSE | bpp | Encode base s | Encode variant s | Encode % | Decode base s | Decode variant s | Decode % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | grid_numpy | 80 | 193.4631 | 0.2793 | 0.4765 | 0.4504 | -5.4824 | 0.0887 | 0.0539 | -39.3050 |
| hpt_data | skip_final_mse | 80 | 193.4631 | 0.2793 | 0.4765 | 0.4146 | -12.9839 | 0.0887 | 0.0886 | -0.2029 |
| hpt_data_val | grid_numpy | 20 | 91.1562 | 0.7189 | 0.3824 | 0.3614 | -5.5020 | 0.0765 | 0.0458 | -40.1161 |
| hpt_data_val | skip_final_mse | 20 | 91.1562 | 0.7189 | 0.3824 | 0.3230 | -15.5448 | 0.0765 | 0.0744 | -2.7853 |

All compared files, pixels, patches, MSE and bpp are identical. `grid_numpy` is a new isolated implementation; `skip_final_mse` exercises an existing API option and returns internal MSE=None while independent artifact scoring remains active. The channel-reduction screen was exact but its small/noisy latency gain did not justify full promotion. See PBC_PERFORMANCE_RESULTS.md for per-image wins, repeat noise, targeted profiles and memory limits.

## Quality and RD evidence

Fixed candidate points versus fresh release HQ (different configurations, so this table alone is not a matched-rate claim):

| Set | Candidate | Cases | MSE | bpp | MSE % | bpp % | Encode % | Decode % | Strict RD image wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | q500_p20 | 20 | 148.5846 | 0.3680 | 3.8085 | -32.1673 | -7.0649 | -41.2155 | 7 |
| hpt_data | q500_p35 | 20 | 128.6912 | 0.4033 | -10.0900 | -25.6545 | 20.8621 | -33.8330 | 5 |
| hpt_data_val | q500_p20 | 5 | 55.8609 | 0.9491 | 38.7944 | -19.1996 | -15.8169 | -46.5096 | 0 |
| hpt_data_val | q500_p35 | 5 | 43.9331 | 1.0197 | 9.1580 | -13.1811 | 17.0549 | -40.7951 | 1 |

Candidate-point comparisons inside the measured release frontier's rate range:

| Set | Candidate | Rate-overlap images | Mean interpolated MSE % | Wins within overlap | MSE-overlap images | Mean interpolated bpp % | Dominated measured points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | q500_p20 | 18 | 6.8190 | 10 | 11 | 30.9185 | 5 |
| hpt_data | q500_p35 | 14 | -5.9425 | 9 | 10 | 9.8136 | 1 |
| hpt_data_val | q500_p20 | 5 | 10.4003 | 2 | 5 | 18.9963 | 1 |
| hpt_data_val | q500_p35 | 5 | -5.9085 | 4 | 4 | 5.6524 | 1 |

Overlap-only means are not dataset-wide guarantees. rd_matched_comparisons.json retains every bracket/weight, nearest measured rate and outlier; no extrapolation occurs. The six screening images have denser 500k curves than the other confirmation images (which have two fixed points), so any curve-envelope aggregation must retain its coverage. PBC_QUALITY_SCALING.md gives absolute controls, preset/action costs, per-prefix marginal rates, catastrophic pointwise outcomes and class-specific interpretation. quality_rd_screen.png/.svg is backed by rd_raw_points.json/.csv and plot_rd.py.

Specifically inside measured Quality-to-High-Quality neighbor brackets:

| Set | Candidate | Q–HQ overlap | Interpolated MSE % | Wins |
| --- | --- | --- | --- | --- |
| hpt_data | q500_p20 | 16 | 3.8835 | 10 |
| hpt_data | q500_p35 | 13 | -7.2164 | 9 |
| hpt_data_val | q500_p20 | 2 | -3.4176 | 1 |
| hpt_data_val | q500_p35 | 4 | -0.5354 | 3 |

For p35, the main-set signal is stronger than the holdout Q–HQ signal; pigeon.JPG remains a +35.47% matched-rate MSE outlier. p20 has a +84.31% matched-rate MSE outlier on mario64.png and +61.80% on pigeon.JPG. These are not hidden by mean bitrate savings. Interpolation against all four default presets and interpolation restricted to Q–HQ cover different image/rate subsets. Matched-MSE means also use a different valid subset, so their signs must not be combined into a joint-dominance claim.

## Entropy outcome

Current-root post-LZMA changes at exactly fixed decoded quality:

| Preset | Transform | Bytes % | bpp | Image wins / 6 |
| --- | --- | --- | --- | --- |
| balanced | horizontal_delta | 3.0307 | 0.1592 | 1 |
| balanced | column_order | 5.2662 | 0.1723 | 3 |
| compression | horizontal_delta | 2.6986 | 0.0859 | 2 |
| compression | column_order | 4.7057 | 0.0945 | 2 |
| high_quality | horizontal_delta | 2.7397 | 0.6541 | 2 |
| high_quality | column_order | 5.1375 | 0.6860 | 2 |
| quality | horizontal_delta | 1.6794 | 0.3107 | 2 |
| quality | column_order | 3.5692 | 0.3227 | 3 |

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
9. About 10.3% encode and 19.3% decode benefit remains when RNG forces exactly matched work. Individual existing optimization shares were not separately ablated.
10. Vectorized grid I/O preserves streams and reduces full decoding latency; optional internal MSE skipping preserves artifacts and reduces encoding work for callers that accept no internal metric. Exact channel reduction is a smaller, inconclusive screen. Individual/full measurements are separate.
11. Bit parsing/packing and deterministic palette/index reconstruction are viable standalone/native building blocks. Remaining PIL bicubic resizing is already native and needs exact semantics preserved in a port. A generic language rewrite, repeated integral fusion or expensive sorting/feature replacement without full-codec evidence is a poor immediate candidate.
12. Ranked follow-up: first validate the isolated grid-I/O improvement and metric-free API path on cpu-basic in a later authorized controlled experiment (low implementation risk; deployment benefit unverified). Next pursue quality representation/allocation that can exploit spatial detail at a fixed final rate, using the fixed 500k image-specific leads and explicit texture/smooth controls (higher RD potential and model/search interaction risk). Then investigate proposal-state reuse/search efficiency with deterministic tie handling and byte-equivalence controls (moderate implementation risk). More generic entropy permutations and a broad native rewrite are lower priority; a fundamentally new PBC 4.0 representation is not established as necessary by this pass.

## Unresolved and repository state

The saved HF Quality discrepancy remains unresolved. Earlier payloads agree; the remote rectangle is absent from the locally replayed proposal pool. There are anchor ties and disabling local NumPy CPU features changes the trajectory, but no tested local variant reproduces the remote file. The checked one-candidate path chooses geometry before model action inference. Further resolution requires authorized running-process/version/CPU/RNG/anchor introspection or another explicitly authorized diagnostic; none was requested or performed.

No strict all-metric production release is accepted. Improvements are classified independently: fixed-output runtime improvements, class-dependent RD leads, rate/quality or rate/runtime trades, regressions and inconclusive low-overlap cases. No conventional-codec superiority claim is made and no AVIF/JXL comparison was added.

All new files are under research_pass2/. Production codec/defaults/model/datasets, existing user work and Pass 1 reports/artifacts are preserved. No training, remote image/compression request, commit, push or deployment occurred. Source/model/input and Pass 1 integrity, saved stream hashes, repeat determinism, output equivalence and syntax accounting are checked by final_checks.py. git_status_before.txt, git_status_after.txt, changed_files.txt, artifact_manifest.json and final_validation.json record the final inventory and verification.
