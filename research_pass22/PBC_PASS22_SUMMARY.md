# Pass 22 summary

Outcome B: material exact-pixel decoder improvement, with a large remaining photographic cost. Preserve research_pass22.decoder.decode as the optimized experimental P21S reader and retain frozen Pass21 unchanged. No production promotion.

1. Empty component / spatial blocks: Googlewiki 69.84% /49.09%; Eiffel 33.77% /5.05%; Cat 75.29% /37.82%. Source pixels covered by any occupied component: 50.78%,94.93%,62.18%.
2. Block skipping is justified on Googlewiki/Cat, not by spatial sparsity on Eiffel. One compact component-batch SciPy IDCT plus dense scatter was implemented; no custom transform or selection redesign.
3. Buffer reuse succeeded: in-place dequantization/IDCT, uint8 base and in-place correction add/rint/clip. No fused scalar kernel. Dense int16, dense scatter, materialized block-to-image reshape, color correction and final output still exist.
4. Exact uint8 pixels, dimensions, SHA256 and original-resolution MSE pass all six frozen files and all 32 synthetic boundary/zero/sparse/dense/extreme fixtures. Repeat decoding is deterministic. 31 invalid/resource cases and two model/encoder-forbidden file-only processes pass. Representation and encoded bytes are unchanged.

| Image | Ordinary HQ ms | Frozen P21S ms | Optimized ms | Reduction vs frozen | Overhead vs HQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | 59.62 | 100.14 | 78.02 | 22.09% | +30.88% |
| eiffel.png | 36.43 | 39.72 | 37.72 | 5.04% | +3.55% |
| cat.jpg | 135.82 | 835.24 | 530.12 | 36.53% | +290.31% |

These are three warmed rotated full-call repetitions, saved streams, same pinned CPU and one numerical thread. Per-repeat values, means/medians/ranges/CV and all source/input hashes accompany the reports. Optimized CV is 7.06% Googlewiki, 2.59% Eiffel, 4.33% Cat. Small Eiffel changes merit limited confidence; Googlewiki/Cat reductions are larger. No HF speed claim.

5. Isolated process peaks: Googlewiki 146.58 ->143.08MiB; Eiffel 123.73 ->125.19MiB; Cat 551.73 ->502.86MiB (medians of two fresh processes each). Memory improves on Googlewiki/Cat and slightly regresses on Eiffel. Cat still requires hundreds of MiB.

Amdahl check using contemporary complete-call medians:

| Image | 1.15 x HQ target ms | Optimized ms | Further saving needed ms | Fraction of optimized time |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png | 68.56 | 78.02 | 9.46 | 12.1% |
| eiffel.png | 41.89 | 37.72 | 0.00 | 0.0% |
| cat.jpg | 156.19 | 530.12 | 373.93 | 70.5% |

Eiffel already approaches the aspirational target locally. Googlewiki needs about 9.5 ms more removed; its remaining layout/color/final profile contains enough work for that to be plausible, without a guarantee of pixel-exact implementation or transfer to other images. Cat needs about 374 ms more, roughly71% of its already optimized full call. Its remaining IDCT is only about 59 ms in the separate profile. Even removing that operation cannot close the gap; dense output passes, base decoding and coefficient recovery remain. Profile observations are not immutable lower bounds, but do not support a credible small-change route to ordinary Cat latency.

Recommendation: retain the useful optimized prototype and Pass21 storage work. Do not start another general-purpose hybrid optimization campaign or tune entropy further. A later bounded block-oriented/fused-output study could have credible upside for small/medium images if optional-mode generalization becomes the objective; it is not justified as a promise to solve the large-image target. No such kernel or activation study was launched here. Current evidence favors an optional experimental mode, not a general-purpose replacement: Cat/Flower RD failures remain and Cat decode remains 3.90x HQ. General-purpose progress should reconsider the architecture rather than accumulate tiny decoder tweaks.

Reproduction uses Python 3.11 with ../research_pass17/requirements.txt, the saved six P21S files matching research_pass21/files.json, saved ordinary HQ files matching research_pass17/baseline.json, and restored original hpt_data for independent MSE. No cached dense coefficient arrays or new encoder runs are required. Restore ignored saved streams from the local research archive; their exact hashes are in provenance.json. Run from repository root:

```
research_pass16/.venv/Scripts/python.exe research_pass22/occupancy.py
research_pass16/.venv/Scripts/python.exe research_pass22/screen.py
research_pass16/.venv/Scripts/python.exe research_pass22/benchmark.py
research_pass16/.venv/Scripts/python.exe research_pass22/profile.py
research_pass16/.venv/Scripts/python.exe research_pass22/memory.py
research_pass16/.venv/Scripts/python.exe research_pass22/allocations.py
research_pass16/.venv/Scripts/python.exe research_pass22/validate.py
research_pass16/.venv/Scripts/python.exe research_pass22/report.py
```

Direct decoder API: from research_pass22.decoder import decode; decode(saved_p21s_bytes) returns uint8 RGB. Default path is the measured combined method; diagnostic sparse/reuse booleans do not alter representation. Frozen research_pass21.sparse_hybrid.decode remains available. Sources and compact evidence stay on research/experiments; generated streams/arrays/caches and benchmark logs are excluded. Final verified GitHub commit/HEAD is provided in the chat handoff.

Practical pbc31_feature_histogram, learned model, v0, frozen Pass17/20/21, HF deployment and release staging are unchanged. Staged ZIP SHA256 remains 08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591. No release publication, tag, protected branch change, full 25x4 benchmark or followup experiment.
