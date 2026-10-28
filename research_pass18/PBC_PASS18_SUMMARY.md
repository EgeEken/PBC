# Pass 18 summary

Stopped after baseline, same-state alternatives, deterministic spatial proposals and one targeted higher-resolution diagnostic. No practical candidate passed the feasibility gate. Production remains pbc31_feature_histogram; staged ZIP/defaults/model/decoder are unchanged.

1. **Policy frequency:** 327 iterative attempts; one requested 1 bit and was rejected at zero gain. None of 323 accepted iterative patches requested 1 bit. Requested 2 bits: 88 accepted patches (27.24%). Actual effective depths: seven 1-bit (2.17%) and 93 2-bit (28.79%) patches.
2. **Serialized grid share:** iterative 1-bit 284/18,364,113 bits = 0.001546%; 2-bit 382,090/18,364,113 = 2.080634%. Including initialization: 0.001465% and 1.971467%. Raw bits only; no per-patch LZMA-byte attribution.
3. **Missed same-region actions:** one of 47 states has a locally dominating finer-grid 1-bit alternative, Cat HQ step 8. No same-cell 1-bit win. One near-gain 2-bit saving on Googlewiki HQ, plus two weaker Flower tradeoffs. No systematic fallback justified.
4. **Spatial omissions:** three deterministic tiles differ from the retained stochastic pool, but none offers comparable total gain cheaply. This does not prove existing search optimal or identify a useful missing binary region.
5. **Content proposals:** one fixed mean-split residual binary-fit/component method, at most three rectangles. Returns three rectangles in two states, six exact tests, no local dominance or high-gain-retention win.
6. **Completed-file RD:** no integrated candidate, hence no completed-file improvement established. Twelve baseline files exactly match Pass 17. The one 500k Googlewiki point is a larger-file/better-quality tradeoff.
7. **Generalization:** no new mechanism demonstrated gains across text/graphics/photos. No holdout or broad validation warranted.
8. **Photographs:** native 1-bit alternatives generally lose substantial gain on Cat/Flower. Cat has one finer-grid local win, Flower two 2-bit tradeoffs. Production paths remain unchanged; no candidate photographic regression claim is possible without candidate files.
9. **Runtime:** median warmed proposal components about 6.36 ms ordinary versus 2.78 ms deterministic; not whole-codec results. No full encode/decode overhead was measured for an integrated mechanism. Instrumented baseline walls are not performance evidence. No promising candidate existed for three-repeat timing.
10. **Format:** every tested legal patch uses unchanged v0 construction/decoding. The free-signed-level explanatory fit is unencoded. No new mask format or decoder was built.
11. **Broader validation:** not recommended for this method. Stop rather than sweep thresholds, weights or candidate counts.
12. **Next hypothesis:** jointly choose signed hard-edge representation and geometry only where initialized residuals support it; explicitly fitted levels alone are insufficient. The ordinary 1-bit palette was zero/positive in 46/47 states despite median ~50% negative residual SSE. Even a four-update free-signed-level probe retained median only 52.83% of ordinary gain. Working-resolution loss is additional, not the sole cause.
13. **Research implementation:** diagnose.py captures unchanged decisions/states; frozen.py tests 120 focused alternatives and six content actions; proposals.py implements the fixed deterministic method; representation.py diagnoses signed levels and the one 500k encode; validate.py parses all baseline bits and round-trips 173 candidate/selected patch payloads; report.py regenerates reports/summary. Compact CSV/JSON are committed, generated arrays/streams/full logits ledger/caches ignored. Two fresh subprocesses decode saved files with policy loading forbidden.
14. **Git handoff:** focused research commit and verified GitHub HEAD are supplied in the chat handoff. Only research_pass18 files are staged; no HF/protected branch/ZIP publication.

Reproduce from research checkout using Python 3.11 and research_pass18/requirements.txt (the existing Pass 17 environment works; SciPy connected components is the only SciPy algorithm used here):

```
research_pass16/.venv/Scripts/python.exe research_pass18/diagnose.py
research_pass16/.venv/Scripts/python.exe research_pass18/frozen.py
research_pass16/.venv/Scripts/python.exe research_pass18/representation.py
research_pass16/.venv/Scripts/python.exe research_pass18/report.py
research_pass16/.venv/Scripts/python.exe research_pass18/validate.py
```

Restored hpt_data assets and unchanged model are required. Runtime/source/config/input identities accompany diagnostics; raw output digests and checks accompany validation. No network jobs, training, double-encode routing, full 100-case screen, DCT retuning or release packaging is invoked.
