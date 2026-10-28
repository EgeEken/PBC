# Pass 18 low-bit-depth analysis

Stop at the frozen feasibility gate. The model rarely requests 1 bit, but these experiments do not show a broadly useful missed 1-bit opportunity. Two-bit patches are already common. No integrated replacement selector or new stream was implemented.

## Baseline and scope

Research start: `2e7d03e47f9db4d501b1f4d2e2bf1b46a1b17735`. Control: unchanged `pbc31_feature_histogram`, runtime/model byte-checked against Pass 16 commit `1c9c4016fdb213d7d16663542d268c7c3df4ed9b`. The actual model is legacy LearnedFiller, cheap=False, candidates=top_k=1, 30 actions. Its value head orders one proposal; termination is nonpositive exact gain, not a STOP head.

Six specified images, Quality/HQ, default seed 2003, 250k cap, YCbCr, initialization and internal RGB MSE enabled, normal LZMA. All expanded configurations, input/source hashes and environment are in diagnostics.json. Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0; i9-12900H logical P-core 12, numerical threads one. No actual-old-versus-new release claim is made: this research compares decisions within the current practical codec.

Twelve instrumented baseline encodes exactly reproduce Pass 17 full-file and independently decoded pixel hashes. The probe returns ordinary selections unchanged. It records every requested/effective precision, cell size, channel/rectangle, grid cells/bits, complete raw patch bits, local clipped channel SSE reduction, current whole-canvas clipped SSE, logits, proposal rank and acceptance. Raw logits/ledger remain local under ignored decisions_raw.json; frozen-state logits remain in diagnostics.json. decisions.csv contains every decision including four rejected stopping attempts. Local patch metrics are working-space SSE, not original RGB SSE.

## Actual decisions

There are 327 iterative attempts, 323 accepted iterative patches and 36 initialization patches. Requested maximum depths across attempts: 1/88/47/102/89 for depths 1/2/3/4/5. Accepted requested depths: 0/88/44/102/89. The sole requested 1-bit action, Monet Quality step 35, has zero gain and stops. This refines the initial observation about accepted actions: it is not zero requests across all attempts.

Actual accepted iterative payload:

| Effective bits | Patches | Patch share | Raw grid bits | Grid-bit share |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 2.167% | 284 | 0.001546% |
| 2 | 93 | 28.793% | 382,090 | 2.080634% |
| 3 | 40 | 12.384% | 385,851 | 2.101114% |
| 4 | 96 | 29.721% | 4,111,488 | 22.388710% |
| 5 | 87 | 26.935% | 13,484,400 | 73.427995% |

One-bit patches are 2.167% of accepted iterative patches but only 284 grid bits, **0.001546%** of iterative grid payload. Requested 2-bit actions are 27.245% of accepted actions; effective 2-bit patches are 28.793%, contributing **2.080634%** of iterative grid bits. The seven effective 1-bit patches came from larger requested maxima, not accepted 1-bit actions. They contain only 284 total cells.

Initialization is separate: 2 patches at 4 bits, 10 at 5, 24 at 6; 1,016,874 grid bits. Across all 359 serialized patches: 19,380,987 grid bits; 1-bit share 0.001465%, 2-bit share 1.971468%, and 6-bit initialization share 3.413510%. Initialization's 6-bit maximum is outside the legacy iterative action space.

| Image | Preset | Iterative patches | Effective 1-bit | Effective 2-bit | 1Ã¢â‚¬â€œ2-bit grid share | Final bytes | RGB MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cat.jpg | quality | 18 | 4 | 3 | 2.2768% | 63,297 | 299.192 |
| cat.jpg | high_quality | 20 | 0 | 7 | 0.8949% | 78,898 | 293.134 |
| flower.jpg | quality | 35 | 1 | 11 | 5.7693% | 28,979 | 43.435 |
| flower.jpg | high_quality | 20 | 0 | 6 | 1.8278% | 46,094 | 34.841 |
| monet.jpg | quality | 34 | 1 | 19 | 1.9023% | 53,871 | 96.796 |
| monet.jpg | high_quality | 20 | 0 | 2 | 0.1115% | 130,159 | 53.976 |
| googlewiki.png | quality | 50 | 0 | 40 | 21.9835% | 82,440 | 781.029 |
| googlewiki.png | high_quality | 20 | 0 | 0 | 0.0000% | 176,905 | 714.432 |
| mario64.png | quality | 50 | 0 | 2 | 0.0973% | 64,809 | 135.472 |
| mario64.png | high_quality | 20 | 0 | 0 | 0.0000% | 89,460 | 122.026 |
| eiffel.png | quality | 16 | 1 | 0 | 0.0072% | 48,183 | 28.757 |
| eiffel.png | high_quality | 20 | 0 | 3 | 0.0685% | 85,874 | 15.410 |

Four zero-gain stops occur at Cat Q step 19, Flower Q step 36, Monet Q step 35 and Eiffel Q step 17. HQ reaches its 20-patch budget for every image here. These current Python 3.11 trajectories differ from historical Pass 7's Python 3.13 sample; those historical counts were not substituted.

The whole-stream entropy wrapper is retained. Raw patch/grid bits do not assign compressed bytes to individual patches: LZMA savings depend on context. Final file bytes include initialization, metadata, padding and entropy framing. The validator parses every ordinary patch and checks exact bit positions against the ledger.

## Same-region counterfactuals

Captured steps 3/8/13/18 when reached: 47 positive accepted states. Target, canvas, selected rectangle, all 50 retained proposals, expanded config, logits and RNG before/after selection are preserved. Captured arrays stay ignored. Replaying stochastic proposals reproduces ordered boxes and RNG state.

Per state: ordinary action; requested 1 bit at the same cell; requested 1 bit at half cell size where distinct; requested 2 bits at the same cell. Only 120 alternative builds after deduplicating cell/bit pairs, never an exhaustive 30-action sweep. Seven builds exactly reproduce the ordinary requested representation; 73 are different requested-1-bit builds and 40 different requested-2-bit builds. Every candidate uses production construction, palette, quantization, bicubic reconstruction, clipping/SSE scoring and exact full raw patch bits.

Diagnostic categories, fixed before observing results: local dominance means at least as much gain at no greater raw cost, with one strict inequality; near-gain cheaper means >=98% gain at <=75% bits; useful tradeoff means >=90% gain at <=60% bits. These categories do not establish final-file RD.

- Same-cell requested 1-bit retained median **36.07%** of ordinary gain. No same-cell 1-bit alternative locally dominated or met either retention/cost category.
- One finer-grid 1-bit alternative dominates: Cat HQ step 8, cell 8 -> 4. Gain 11,728 -> 36,277, raw bits 4,725 -> 4,653. This is one geometry/precision opportunity, not a systematic same-grid precision error. It reduces an extra 24,549 working SSE against current total SSE 4,492,407; final RGB/entropy/trajectory consequences are unmeasured.
- One 2-bit near-gain opportunity: Googlewiki HQ step 8, cell 8, gain retention 99.622%, raw bits retention 69.097%.
- Two further 2-bit tradeoffs: Flower Q step 18 (90.861% gain/59.420% bits), Flower HQ step 13 (91.227%/56.250%). Neither is local dominance.

The strict 1-bit win is one of 47 states (one of 73 different 1-bit builds). Same-grid 2-bit savings occur only in three states, mostly trading away gain. This does not justify a general fallback, another weight sweep or an image-specific trigger.

## Why 1 bit fails here

In 46/47 same-cell tests, the actual generated 1-bit palette is zero plus one positive level. With mask[0] reserving zero and only one remaining palette entry, palette_generator assigns that entry to the first active nonzero range; negative ranges often receive none. The median negative residual share is 49.82% of squared residual energy. A single zero/positive residual correction cannot directly repair negative residuals. This is a concrete property of the present generated palette, not a claim that a 1-bit source image needs only positive corrections.

An explanatory, **unencoded** fit allowed two free signed levels on the same projected grid, initialized at extrema and updated four deterministic Lloyd iterations. Exact production interpolation/clipped SSE scoring retained median **52.83%** of ordinary gain, versus 36.07% for native 1 bit. Only three states retained >=90%; one is an already degenerate native representation. This is a small heuristic representation probe, not a globally optimal bound, a valid v0 palette, a serialized candidate or an RD result. Better two-level quantization alone does not recover ordinary gain across these states.

The working residual after initialization is often signed and multilevel even for graphic inputs. Fine-grid native 1-bit failures show that bicubic blur alone cannot explain the losses; palette capacity/placement and residual distribution also matter. No numerical causal share is assigned to quantization, interpolation, downsampling or later trajectory changes.

## Disposition

The measured model underselection of requested 1 bit is real, but useful missed alternatives are rare. Two-bit coding is already common in Quality Googlewiki (40/50 iterative patches). No completed-file candidate screen was warranted; no completed-file RD, runtime, generalization or holdout gain is claimed. Spatial and resolution diagnostics are in PBC_SPATIAL_PROPOSAL_RESULTS.md.
