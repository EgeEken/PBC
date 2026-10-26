# PBC Pass 11 — Final-RGB objective feasibility

## Method

The frozen release-compatible source at GitHub `8006f30` was run locally on six main-set images (`cat`, `ducklings`, `flower`, `monet`, `panorama`, `seagull`) with Quality and High Quality presets. At steps 3, 8, and 13, the probe compared the selected legacy policy action with one already available alternative: the next-highest action logit for the same chosen box. Policy weights, proposal generation, RNG stream, and selected action were unchanged.

For each pair, working-space gain is the encoder’s exact patch-local channel SSE reduction. Final-RGB gain is measured by applying the patch delta to a copy of the current working canvas, using the codec’s clipping and YCbCr-to-RGB conversion, resizing to the original dimensions with the decoder’s PIL filter, and measuring RGB SSE against the source. Cost is the raw serialized patch bit estimate; it is not a prediction of the final LZMA size.

The probe captured 36 pairs. Its local environment was Windows, Python 3.13, NumPy 2.2.6, Numba 0.62.1, Pillow 11.3.0, one numerical thread. These timings are feasibility evidence only and are not HF timings.

## Results

Only one pair changed the ordering between working-space SSE gain and final-RGB SSE gain: Panorama High Quality, step 13. The chosen action reduced working SSE by 125,577 and final RGB SSE by 736,541 at 957 raw bits. The alternative reduced working SSE by 126,246 but final RGB SSE by 712,392 at 1,167 bits. Thus the current choice was better in final RGB and cheaper, despite its slightly lower working-space gain.

Two pairs had equal raw cost; selected and alternative working and final-RGB gains were identical in both. No equal-cost alternative improved final-RGB SSE. Across the sample the alternatives averaged higher raw cost by 39,757 bits and a slightly larger final-RGB MSE change, but this is cost-unmatched and provides no valid RD win. Clipping and final resizing were included in every final-RGB score.

One additional alternative build plus the selected/alternative/baseline final-RGB rescoring took a mean 0.434 s per sampled state (median 0.457 s, max 0.902 s) locally. This is expensive relative to the controlled baseline encode times and was measured on the local Windows runtime. The full-RGB score is therefore not a cheap production proxy.

## Decision

Stop at the feasibility gate. The single ranking disagreement favors the current lower-cost action after final reconstruction; other equal-cost cases tie. No opt-in ranking adjustment, completed-file candidate encoding, matched-rate comparison, or 20-image confirmation was run. There is no evidence that the final-RGB mismatch is both common and exploitable cheaply. The hypothesis is rejected as a production direction on this sample; no codec, model, decoder, stream syntax, preset, or public demo change was made.

The reproducible research-only instrumentation is `research_pass11/rgb_probe.py`; its compact raw output is `research_pass11/rgb_probe.jsonl` and is intentionally not committed. The frozen source was not edited. The probe uses an in-memory wrapper around the legacy action builder and returns the selected action unchanged.

- Baseline GitHub source: [8006f30](https://github.com/EgeEken/PBC/commit/8006f30debfc4a57289df06df21636aeb52baffd)
- Research branch at start: [678c029](https://github.com/EgeEken/PBC/commit/678c0293712cb63cb6acffa9de92d9cac36fb752)
- Source SHA-256: `PBC3.py` 74fe069600f8190f1a5ac1b4cb01dff29f4b1262d6d36da459236e7a9fb0016c; `pbc3_ops.py` 62243c349dd5f74c6248abad0013a38aeb416984074bd47e3dbdfec6fd11739e; `learned_filler.py` 65d2de14c1fd01be256dbbcdfa98001db8312cfdb7a032f4e40b2c54a58b0c50; model `patch_policy.npz` ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f.
