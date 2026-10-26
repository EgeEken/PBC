# Decoder-aware grid refinement

Neither screened variant is promoted. The feedback changes real indices and improves local visible working-space SSE, but does not establish a meaningful final rate-distortion improvement. No full main/holdout confirmation, 500k/35 follow-up, combined method, second strength or further optimization was run. Nothing from this experiment is enabled or deployed on HF.

One forward correction is applied after the ordinary patch has been selected. Its geometry, cell size, grid dimensions, palette mask/bounds/precision and syntax stay fixed. Reconstruct delta R(Q) using the actual signed bicubic/rounded decoder path, calculate e = target − clip(canvas + R(Q), 0, 255), and project e to the stored grid. Propose Q′ by requantizing values(Q) + 0.5 × projected(e) onto the same generated palette. Reconstruct Q′ with the same integer resampler and accept only strictly lower clipped local SSE. int16 additive canvas semantics are retained. LZMA is never called in candidate refinement.

B-init refines the three selected full-image RGB initialization grids. B-iter tries at most the first three selected iterative patches with cell size >1 and nominal index payload ≥8192 bits. This uses only encoding-time information; geometry, model, proposal count, weights, RNG seed and configuration are unchanged. Both variants allow the ordinary sequential trajectory and stop decision to evolve afterward. The scoped class hooks are restricted to isolated research processes.

Screen: cat.jpg, flower.jpg, monet.jpg, mario64.png, eiffel.png, googlewiki.png; Quality and High Quality, default 250k working cap; optimized current control, B-init, B-iter; three warmed interleaved repeats (108 timed calls, six warmups). Full candidate encode walls include scope installation/restoration, projections, quantization, extra reconstruction/scoring and all later trajectory work. Independent original-size RGB decode/MSE and complete post-LZMA bytes determine final outcomes. All 108 previews equal normal v0 decoder output and each configuration repeats deterministically.

| Variant | Mean RGB MSE | Mean bpp | Encode s | Decode s | MSE change | bpp change | Encode change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| integrated | 220.059462 | 0.9288214 | 0.388288 | 0.038841 | +0.000% | +0.000% | +0.00% |
| init | 219.867921 | 0.9115209 | 0.430038 | 0.039570 | -0.087% | -1.863% | +10.75% |
| iter | 219.714491 | 0.9212087 | 0.391032 | 0.039376 | -0.157% | -0.820% | +0.71% |

MSE/bpp are equal-case arithmetic means; times average the 12 case medians. These fixed-preset aggregates are not matched-rate frontier measurements. Averaging relative per-case MSE changes instead produces −2.35% for init and −0.91% for iter; those different weightings must not replace the near-neutral mean RGB result. Small mean rate savings also do not prove per-image joint dominance.

| Mechanism | Attempts / accepts | Accepted indices changed | Extra grid projections / reconstructions | Local SSE reduction | Strict RD wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| init | 36 / 36 | 32230 | 36 / 36 | 49438754 | 4/12 |
| iter | 27 / 27 | 9127 | 27 / 27 | 1520862 | 3/12 |

Counts describe one deterministic encode for each of the 12 configurations, not all repeated calls. Local SSE reductions sum accepted same-state comparisons, not a global RGB improvement or a decomposition of final loss. The nominal bit cost of each considered patch is unchanged, but later geometry/counts and entropy context change complete raw and LZMA sizes.

| Image | Preset | Variant | MSE change | bpp change | Encode change | First changed geometry patch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| cat.jpg | quality | init | -0.40% | -10.30% | +37.63% | 5 |
| cat.jpg | high_quality | init | -0.30% | +10.05% | +6.69% | 5 |
| flower.jpg | quality | init | -0.37% | -0.90% | +42.21% | 3 |
| flower.jpg | high_quality | init | +2.67% | -7.15% | +2.28% | 3 |
| monet.jpg | quality | init | -6.13% | +3.12% | +26.65% | 3 |
| monet.jpg | high_quality | init | -3.70% | -1.30% | +16.33% | 3 |
| mario64.png | quality | init | +3.30% | -4.51% | -1.70% | 3 |
| mario64.png | high_quality | init | -7.16% | +3.79% | +5.15% | 4 |
| eiffel.png | quality | init | +1.47% | -1.49% | +13.47% | 3 |
| eiffel.png | high_quality | init | -19.67% | -6.29% | -21.06% | 3 |
| googlewiki.png | quality | init | +0.71% | +5.00% | +8.23% | 3 |
| googlewiki.png | high_quality | init | +1.39% | -2.32% | +1.08% | 5 |
| cat.jpg | quality | iter | -0.01% | +0.03% | +0.17% | None |
| cat.jpg | high_quality | iter | +0.01% | -0.05% | -1.84% | 10 |
| flower.jpg | quality | iter | +1.67% | -2.24% | -0.44% | 8 |
| flower.jpg | high_quality | iter | -10.97% | +28.27% | -0.82% | 7 |
| monet.jpg | quality | iter | +1.37% | -2.98% | +14.16% | 8 |
| monet.jpg | high_quality | iter | +0.28% | -3.88% | +2.50% | 7 |
| mario64.png | quality | iter | -1.40% | -0.80% | +0.56% | 18 |
| mario64.png | high_quality | iter | -0.19% | -5.12% | -1.88% | 11 |
| eiffel.png | quality | iter | -0.76% | +0.01% | +0.33% | None |
| eiffel.png | high_quality | iter | -0.90% | -0.07% | -0.90% | None |
| googlewiki.png | quality | iter | +0.00% | +0.00% | +0.27% | None |
| googlewiki.png | high_quality | iter | +0.00% | +0.00% | +1.81% | None |

First geometry position is zero-based in the complete stream; None means no differing common-prefix geometry, not necessarily identical coefficients or patch count. Compact output retains trajectory hashes and first differences; the complete measured trajectories remain locally in ignored refinement_raw.jsonl.

B-init accepts all 36 corrections, improving summed local SSE by 49,438,754, yet mean original-size RGB MSE improves only 0.087%. Eiffel HQ is a useful individual point (about −19.67% MSE and −6.29% bpp), while Mario Quality worsens MSE 3.30%, flower HQ 2.67%, and googlewiki worsens at both presets. Cat HQ spends about 10.05% more bpp for only 0.30% lower MSE. Quality cat/flower encode increases about 38%/42%; patch counts rise 20→53 and 16→53 including three initialization patches. Their ordinary trajectories now fill the unchanged budget instead of stopping early. Thus the penalty includes real additional search/patch work, not only the three corrections.

B-iter accepts 27 corrections and changes 9,127 indices. Googlewiki has no eligible coarser-grid patches, so both presets are unchanged. Mean RGB MSE improves only 0.157%; flower HQ improves about 10.97% MSE at 28.27% more bpp, a tradeoff rather than a fixed-rate win. Flower Quality worsens 1.67% MSE and Monet worsens slightly at both presets. There are only 3/12 strict RD wins. Its small cost does not compensate for inconsistent final gains.

Same-preset actual controls within 5% final rate cover 8/12 init cases, with 4 MSE wins and mean relative MSE change -1.311%. These are unequal-rate files within a tolerance, not exact matches or interpolation. Encode/decode mean repeat CV is 3.54%/5.31%.
Same-preset actual controls within 5% final rate cover 10/12 iter cases, with 4 MSE wins and mean relative MSE change +0.026%. These are unequal-rate files within a tolerance, not exact matches or interpolation. Encode/decode mean repeat CV is 2.73%/5.36%.

The screen fails the intended meaningful final-RD/cost gate. The initialization relative-percentage average is driven by a few cases; its arithmetic RGB mean is almost unchanged and its encode cost is material. The iterative variant is cheap but does not give consistent fixed-rate quality improvement. No dense rate sweep or release comparison is needed to reject promotion at this stage. The immutable release remains available for a future qualified candidate; no four-axis claim can be made from these fixed-preset means.

The negative result is not “indices never move.” Actual quantized indices move, exact local scoring improves, and normal decoder reconstruction matches. The unresolved issue is transfer through later channel/geometry/stop decisions, color conversion and final resizing to original-size RGB. Those mechanisms are not assigned causal shares here. Another feedback strength or more correction iterations is not justified by this screen.

A distinct next mechanism is a bounded stop-policy audit: only when the first learned proposal would stop, evaluate one already-ranked alternative proposal and measure whether useful corrections were missed. Freeze the model and proposal generation, cap the extra work, and require actual matched-rate evidence before broader confirmation. This addresses allocation/stop behavior rather than repeating coefficient feedback or training.
