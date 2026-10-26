# Robust generated-palette bounds

Rejected after the six-image completed-file screen. The 98th-percentile candidate produced useful fixed-state patches, but forward encoding gave **RGB MSE −0.271%, final bpp +6.983%, encode time +13.496%**. There were no strict RD wins. No full main/holdout test or RD deployment followed.

## Fixed-state gate

Replayed 12 existing saved Quality/HQ streams for Cat, Flower, Monet, Mario64, Eiffel and googlewiki: 331 selected patches, including 36 initialization patches. Their SHA256 hashes match the unmodified controls recorded in Pass 7. No proposal search or encode was run to collect these examples. The baseline unquantized projections, masks, bounds, indices and reconstructed final images were reproduced exactly at each saved canvas state.

Candidates use the 98th and 95th percentiles **separately over nonzero negative magnitudes and positive values**, rounded upward and capped at 255. Empty sides retain zero. Geometry, channel, cell/grid dimensions, maximum bitcount, target/canvas and signed bicubic decoder semantics are fixed. Normal mask generation, palette generation, clipping and quantization rebuild the representation. The complete raw patch bit cost is measured against the actual serializer; 662 candidate serialization/readback checks reproduce the generated values. Exact distortion uses the production clipped working-space reconstruction, including its int16 canvas semantics. LZMA is never called inside this candidate loop.

The gate was fixed before the diagnostic: at least 10% of patches with strict local SSE improvement and no greater raw cost, useful patches on at least four images, and summed useful local SSE opportunity at least 0.1% of summed evaluated baseline patch SSE. These overlapping patch-state sums are a feasibility statistic, not a final image quality estimate.

| Candidate | Changed bounds | Lower local SSE, no more raw bits | Lower raw bits, no more SSE | Summed local SSE opportunity | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 98th percentile | 296/331 | 85/331 (25.68%) | 0 | 1,573,415 / 1,458,242,896 = 0.1079% | Pass |
| 95th percentile | 310/331 | 35/331 (10.57%) | 0 | 315,200 / 1,458,242,896 = 0.0216% | Fail |

Both have useful patches on all six images. Neither produces a lower-SSE/higher-bit-cost case. Lower-bit-cost/higher-SSE tradeoffs occur on three/ nine patches respectively and are not accepted. No tradeoff weight or extra percentile was tuned.

## Frozen forward rule

Use only the 98th percentile. After the ordinary initialization or iterative head selects a patch, reconstruct its unquantized projected grid from the current residual, build one robust palette, and accept it only when exact clipped working SSE strictly improves at no greater complete raw patch cost. Otherwise retain the original patch. This changes neither the proposal pool nor model/preset/seed, cell, geometry, maximum precision or decoder. Later greedy trajectories may change naturally. The implementation is deliberately scoped to isolated workers using the deployed legacy model; it is not installed in the concurrent public codec.

Two persistent local workers, common CPU 0 and numerical threads one; six images × Quality/HQ × two modes × three interleaved repeats = 72 measured full encode/decode calls, plus four consistent warmups. Model/config/source/input hashes and environment metadata are retained. Full encode time includes reprojection, percentile calculation, rebuilding, exact scoring, scope setup/cleanup and all later search/patch work. Decode and independent original-size RGB MSE are measured from complete final post-LZMA files. This screen uses Windows Python 3.13.5 / NumPy 2.2.6 / Numba 0.62.1 / Pillow 11.3.0; it is separate from the HF timing comparison.

| Mode | RGB MSE | Final bpp | Encode s | Decode s | Mean total patches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current integrated control | 220.059462 | 0.928821 | 0.401698 | 0.041216 | 27.583 |
| Robust 98 | 219.463783 | 0.993677 | 0.455911 | 0.042126 | 29.750 |

Ratios of aggregate means: MSE −0.2707%, bpp +6.9825%, encode +13.4958%, decode +2.2095%. Means of per-case relative changes are separately −0.3919%, +8.1658%, +14.1919%, +2.4350%. Encode CV averages 10.35% control / 7.05% robust; decode 12.42% / 9.55%. All twelve encode medians are slower; the small aggregate decode change is within repeat noise.

The forward rule accepts 86 of 357 considered selected patches across one encode of each case. Five of twelve completed images have lower RGB MSE; seven regress. Zero strict RD wins. Same-patch raw-cost guards do not control final entropy or the cost of later patches.

| Case | RGB MSE change | Final bpp change | Encode change | Patches |
| --- | ---: | ---: | ---: | ---: |
| Cat Quality | −1.949% | +16.200% | +24.84% | 20 → 35 |
| Cat HQ | −1.394% | +15.829% | +7.11% | 20 → 23 |
| Flower Quality | +2.101% | −0.007% | +2.60% | 16 → 11 |
| Flower HQ | +0.888% | +6.153% | +9.30% | 23 → 23 |
| Monet Quality | −3.048% | +19.254% | +39.54% | 36 → 53 |
| Monet HQ | +0.159% | +22.045% | +7.17% | 23 → 23 |
| Mario64 Quality | −2.202% | +2.110% | +20.23% | 53 → 53 |
| Mario64 HQ | +0.230% | +1.203% | +10.57% | 23 → 23 |
| Eiffel Quality | −1.371% | +2.250% | +12.12% | 20 → 18 |
| Eiffel HQ | +0.905% | +8.745% | +9.28% | 21 → 19 |
| googlewiki Quality | +0.331% | +0.826% | +15.74% | 53 → 53 |
| googlewiki HQ | +0.648% | +3.381% | +11.80% | 23 → 23 |

Within-source repeat stream/pixel hashes are deterministic and every preview equals independent decode. The existing v0 serializer/decoder are used unchanged. Per-file rate, RGB distortion, timing samples, accepted positions, patch counts and trajectory changes are retained in `palette_screen.json`; `palette_decision.json` contains paired outcomes. No actual equal-rate frontier improvement is established by the sparse unequal-rate controls, and added rate/CPU plus named regressions already fail screening. There is no reason to tune more percentiles or run broader confirmation.

Reproduction uses `palette_feasibility.py --archive PATH_TO_EXISTING_CURRENT_STREAMS`, then, only after its gate passes, `screen_palette.py` and `analyze_palette.py`. `palette_feasibility.json` records the 12 stream/input hashes, raw-evidence hash, exact gate and candidate summaries. `palette_fixed_raw.jsonl` and `palette_screen_raw.jsonl` retain complete local rows/trajectories outside Git. Required source is the root codec, `research_pass2/stream_tools.py`, `research_pass6/worker.py`, and this directory's palette modules. No new model, dataset or dependency is introduced. All feasibility and forward work began after the mandatory HF comparison and final UI/API checks were complete.
