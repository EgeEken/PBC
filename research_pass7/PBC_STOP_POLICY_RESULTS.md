# Premature-stop recovery

Stop after the six-image screen. Early stopping is real, but a second existing proposal did not establish a matched-rate RD improvement worth promotion. The deployed model has **no stop head**, so S1 is inapplicable. S2 produces small final-file quality gains with higher bitrate and concentrated encode costs. No RD code was deployed, no full-dataset/holdout confirmation was run, and no PBC 3.1 improvement is claimed.

## Active policy and diagnosis

The unchanged 104,903-byte `patch_policy.npz` has the legacy action/value heads (`Wa/ba`, `Wv/bv`), not `head_sizes` or a factored STOP head. `LearnedFiller.load` therefore returns `LearnedFiller`, not `FactoredFiller`. With the default candidates=top_k=1, its value head only orders a singleton; there is no value threshold that means STOP. It terminates when there are no proposals or the constructed patch's exact reduction is nonpositive. A factored stop implementation exists in source but is not selected by these weights. Treating every termination as learned STOP would be incorrect.

An initial diagnostic hooked the inactive factored class, producing no observations. Model inspection exposed that mismatch; that attempt was discarded. The corrected diagnostic wraps the actual legacy method without replacing its decisions, and matches the unmodified codec's streams and decoded pixels on all 14 cases. This was a focused diagnostic check, not a repeat of the 100-case runtime suite.

| Image | Quality accepted / requested | HQ accepted / requested | Termination |
| --- | ---: | ---: | --- |
| cat | 17 / 50 | 17 / 20 | Both zero exact gain |
| flower | 13 / 50 | 20 / 20 | Quality zero gain; HQ budget |
| monet | 33 / 50 | 20 / 20 | Quality zero gain; HQ budget |
| mario64 | 50 / 50 | 20 / 20 | Budget |
| eiffel | 17 / 50 | 18 / 20 | Both zero exact gain |
| googlewiki | 50 / 50 | 20 / 20 | Budget |

These are iterative patches; each file also has three initialization patches. Six of 12 main screen cases stop early, each with 50 already-ranked boxes available. Remaining budgets are 33/3 for Cat, 37 for Flower Quality, 17 for Monet Quality and 33/2 for Eiffel. All observed terminating gains are exactly zero, not negative. No-proposal and stop-head counts are zero. Compression Cat stops after 34/50; Balanced googlewiki after 27/50, also at zero gain with 50 boxes.

## Frozen rule and measurement

`stop_policy.py` installs scoped hooks only inside an isolated worker. The original method still generates its ordinary pool, extracts features, runs the model, constructs and scores the first action. Diagnostic/S1 modes preserve that return value. S1 records zero attempts because the current model has no STOP output; its byte-identical result is a no-op control, not evidence about a different factored policy.

S2 acts only after a nonpositive first patch. It uses **boxes[1] from that same captured pool**, ordinary legacy feature extraction/normalization, both ordinary model heads and argmax action, then normal construction/exact scoring. It accepts strictly positive reduction only. It never regenerates proposals, advances search RNG separately, examines a third proposal, or alters model/config/serializer/decoder. At a subsequent ordinary stop it may again check one second proposal, until normal budget exhaustion or rejection. There is no unavailable second STOP decision to override in this model. The implementation deliberately asserts candidates=top_k=1 rather than generalizing to unrelated policies.

Screen: Cat, Flower, Monet, Mario64, Eiffel, googlewiki × Quality/HQ, plus the two compatibility smoke cases. Three persistent workers, one common pinned Windows CPU, numerical threads one, seed 2003, unchanged model, default 250k working cap and preset budgets. Each mode receives Quality/HQ warmups; 126 measured full encode/decode calls rotate mode order over three repeats. No-op S1 is included to verify outputs and expose hook overhead. Warmups initialize the workers consistently and are outside steady-state medians; this is not a cold-start measurement. All per-encode initialization, internal MSE, fallback inference, construction, continued search/patches, serialization, reconstruction and scoped hook setup/cleanup are inside encode time. Decode includes final resizing. External original-size RGB scoring and stream parsing are outside the timers.

`stop_screen.json` retains three timing samples per case, final post-LZMA bytes/bpp, RGB MSE, stream/pixel/trajectory hashes, patch counts and stop/rescue events. All repeat hashes are deterministic and previews equal independent decoding. Complete trajectories and rows remain in ignored `stop_screen_raw.jsonl`; its SHA256 and source/model/input hashes are recorded. Timing environment: Python 3.13.5, NumPy 2.2.6, Numba 0.62.1, Pillow 11.3.0. These are local results, not HF timings. Short fixture/audit commands overlapped the end of the screen, so scheduling noise remains; no fine-grained speed claim depends on the small differences for byte-identical cases.

## Final-file results

Equal weighting of 12 Quality/HQ cases; times are means of the three-repeat case medians. Compression/Balanced smoke cases are excluded from these aggregates.

| Mode | RGB MSE | Final bpp | Encode s | Decode s | Mean patches incl. init |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unmodified integrated | 220.059462 | 0.928821 | 0.397660 | 0.040044 | 27.583 |
| S1, inapplicable/no-op | 220.059462 | 0.928821 | 0.401424 | 0.039448 | 27.583 |
| S2 | 219.750621 | 0.931379 | 0.411683 | 0.040465 | 30.833 |

S2 aggregate changes: **MSE −0.1403%, bpp +0.2754%, encode +3.5263%, decode +1.0496%**. Mean per-case relative changes use different weighting: −0.5539%, +0.4186%, +4.5325%, +1.3570%. Mean encode CV is 4.56% control / 4.39% S2; decode CV 8.60% / 5.12%. S1 overhead/noise is about +0.95% aggregate encode; no subtraction from S2 was used to manufacture a corrected speedup.

There are ten fallback attempts and six successful rescue patches across the 12 cases, changing four completed files. The six rescues permit 33 additional ordinary patches, for 39 extra patches total. Direct additional inference/construction medians are retained separately in `stop_decision.json` and raw component times in the screen; they are roughly sub-millisecond to a few milliseconds per affected encode. The main cost is continuing the entire encoder after rescue.

| Changed case | MSE change | Bpp change | Encode change | Total patches |
| --- | ---: | ---: | ---: | ---: |
| Cat Quality | −0.515% | +2.665% | +17.30% | 20 → 32 |
| Monet Quality | −0.872% | +1.217% | +20.21% | 36 → 53 |
| Eiffel Quality | −5.193% | +1.014% | +19.00% | 20 → 28 |
| Eiffel HQ | −0.067% | +0.127% | +0.77% | 21 → 23 |

Eight cases have identical streams; no MSE regression was observed, but every changed file is larger. **Zero strict RD wins.** All same-preset controls are actual measured points within 5% rate, but their rates are unequal. Eight unchanged cases give exact matched-rate and matched-MSE ties. The four changed cases have no actual equal-rate/equal-MSE counterpart in this screen, so neither MSE-at-matched-bpp nor bpp-at-matched-MSE improvement is established. The Quality/HQ endpoint intervals are too sparse to replace this missing evidence with a credible frontier claim. A larger file with less error is a tradeoff, not sufficient acceptance evidence.

Smoke cases also remain tradeoffs: Cat Compression adds 16 patches for about −0.27% MSE / +3.98% bpp; googlewiki Balanced adds 12 for about −0.09% / +1.01%. Full hashes and timing rows are retained. No stream syntax or decoder change is needed.

The overall encode cost is below the approximate 5% preference, but the useful changes cost about 17–20% on three Quality cases and the strongest quality result is concentrated on Eiffel. Together with the tiny aggregate MSE change, larger files and absent matched-rate proof, that fails the screen. No more proposals, budget/lambda sweep, new model, training or full immutable-release comparison is justified in this pass.

Reproduce: restore the declared six main images with `research/prepare_assets.py`, then run `python research_pass7/test_stop_policy.py`, `screen_stop_policy.py --diagnose`, `screen_stop_policy.py`, and `analyze_stop_policy.py`. Fresh runs replace only ignored local evidence unless explicitly staged. The seven small fixture cases verify S1/no-op behavior, positive-only S2 acceptance, rejection, natural first success, missing alternatives, empty pools, a single ordinary RNG/pool call, no third proposal and hook restoration. Dependencies are the existing research environment plus `research_pass6/worker.py`; no additional weights or data are required.

Next: resolve benchmark persistence before any rebuild. For RD research, first diagnose why the legacy policy selects zero-gain actions on highly ranked boxes; do not extend this fallback search or assume there is a stop head to retrain.
