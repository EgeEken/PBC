# Pass 25 completed transform-base results

Reject the tested architecture as a general-purpose improvement. Outcome C/D: useful initialization-only capacity does not produce a reliable completed-file RD gain with the existing patch policy, and conventional JPEG dominates 5 of 8 hybrid points. No extension or repeated timing campaign.

Research-only P25B is a 17-byte <4sIIBI> header (magic, complete P25I base length, raw spatial bytes, production entropy method, patch count), followed by the full base and generated-palette patch payload. Framing overhead is 41 bytes including the P25I header, before entropy-internal headers. Spatial records use existing patch syntax/channel bits 2/positive_bias=True; spatial entropy uses the unmodified production raw/LZMA selection. No three fabricated initial patches.

The original model loads as legacy LearnedFiller: it has no begin_encode, running base-bit accounting or initialization-patch history. Verified exactly that class; factored policies reject instead of receiving false metadata. Owned local FillerHead/SearchState, the normal legacy proposal/feature/action path, release NumPy RNG, target, channel selection, additive int16 semantics and early stopping are retained. Ordinary Q50/HQ20 iterative budgets remain fixed; all hybrid patches are freshly optimized from the decoded transform canvas. The policy runs correctly but its efficacy from this residual distribution is empirical, not assumed. No retraining or production monkeypatch.

Full input preparation, base encode/decode, refinement, framing/entropy, final conversion/resize and enabled original-source internal MSE are inside candidate encode timers. The decoder reconstructs from the actual file only. Ordinary defaults are freshly encoded and byte/pixel-identical to saved Pass17 controls. Base-only runs the same encoder with zero iterative budget; output exactly matches initial step 128. One warmed deterministic complete call per setting, alternating order, on i9-12900H P-core 12 with one numerical/FFT thread and Python 3.11.16/NumPy 2.4.6/Numba 0.68.0/Pillow 11.3.0/SciPy 1.17.1. Hashes/I/O/external scoring excluded. No rejected candidate gets three timing repetitions.

| Image | Preset | Ordinary bytes / MSE | Step128 base-only bytes / MSE | Hybrid bytes / MSE | Rate change vs ordinary | MSE change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | quality | 82,440 / 781.029 | 14,149 / 1148.740 | 119,892 / 770.262 | +45.43% | -1.38% |
| googlewiki.png | high_quality | 176,905 / 714.432 | 14,149 / 1148.740 | 172,187 / 711.262 | -2.67% | -0.44% |
| monet.jpg | quality | 53,871 / 96.796 | 5,289 / 252.540 | 63,892 / 128.647 | +18.60% | +32.91% |
| monet.jpg | high_quality | 130,159 / 53.976 | 5,289 / 252.540 | 131,826 / 71.850 | +1.28% | +33.11% |
| cat.jpg | quality | 63,297 / 299.192 | 3,405 / 447.120 | 57,087 / 308.829 | -9.81% | +3.22% |
| cat.jpg | high_quality | 78,898 / 293.134 | 3,405 / 447.120 | 85,359 / 296.166 | +8.19% | +1.03% |
| flower.jpg | quality | 28,979 / 43.435 | 6,685 / 160.654 | 43,121 / 80.801 | +48.80% | +86.03% |
| flower.jpg | high_quality | 46,094 / 34.841 | 6,685 / 160.654 | 120,436 / 45.216 | +161.28% | +29.78% |

Only Googlewiki HQ strictly improves its same-preset default: -2.67% bytes and -0.44% MSE. Googlewiki Q pays 45.43% more for 1.38% lower MSE. Monet Q/HQ MSE worsens about 33%; Flower Q/HQ worsens 86.03%/29.78%, with much larger files. Cat Q trades 9.81% fewer bytes for 3.22% more MSE; Cat HQ is worse in both. This fails the frozen two-class meaningful-completed-RD gate with no useful classes, even though some initial-only results passed.

Contribution of PBC refinement at the same frozen base:

| Image / preset | Base bytes | Raw spatial bytes | Final spatial payload bytes | Patches | Full bpp | MSE reduction from step 128 base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png / quality | 14,132 | 217,988 | 105,743 | 50 | 1.45111 | 32.95% |
| googlewiki.png / high_quality | 14,132 | 421,784 | 158,038 | 20 | 2.08406 | 38.08% |
| monet.jpg / quality | 5,272 | 87,112 | 58,603 | 50 | 0.60482 | 49.06% |
| monet.jpg / high_quality | 5,272 | 206,203 | 126,537 | 20 | 1.24791 | 71.55% |
| cat.jpg / quality | 3,388 | 123,918 | 53,682 | 50 | 0.05357 | 30.93% |
| cat.jpg / high_quality | 3,388 | 184,873 | 81,954 | 20 | 0.08010 | 33.76% |
| flower.jpg / quality | 6,668 | 57,537 | 36,436 | 50 | 0.04046 | 49.71% |
| flower.jpg / high_quality | 6,668 | 235,636 | 113,751 | 20 | 0.11302 | 71.86% |

PBC patches are not ineffectual: they reduce source RGB MSE 31-72% versus the same coarse base. They are also not demonstrated compression-efficient: they dominate final bytes and fail the external/ordinary controls. Flower Quality is strictly dominated by the already measured step 64 transform-only base, reframed as an actual P25B zero-patch file:12,717 bytes/MSE 74.156 versus 43,121/MSE 80.801. This uses a predeclared A quantizer, no new transform encode or sweep. All other base64/128/256 reframed files and pixel checks are recorded in analysis.json.

Conventional JPEG reference: one fixed quality 75 working-resolution file per image, with the same 14-byte original-size comment and final resizing. Encoder timing excludes common prepare for JPEG and is not a comparable full speed claim. No rate search. Actual sizes below hybrid rates are preserved; no equal-rate extrapolation.

| Image | JPEG q75 bytes / source MSE | JPEG dominates hybrid Q | JPEG dominates hybrid HQ |
| --- | ---: | --- | --- |
| googlewiki.png | 63,388 / 800.505 | False | False |
| monet.jpg | 46,041 / 71.910 | True | False |
| cat.jpg | 30,761 / 294.283 | True | True |
| flower.jpg | 30,693 / 31.366 | True | True |

JPEG strictly dominates Monet Q, both Cat points and both Flower points. Monet HQ is only about 0.08% lower MSE than JPEG while taking 2.86x its bytes; that remains a trade, not JPEG dominance. Both Googlewiki hybrids have lower MSE than the q75 JPEG at far higher bytes. No hybrid strictly dominates that JPEG point. Neither q1 nor q75 brackets any hybrid final rate, so a matched-rate JPEG conclusion cannot be invented. The bounded reference is sufficient to reject general superiority on 5 actual points, not to estimate a continuous JPEG frontier.

Stronger ordinary controls include saved Q/HQ, Quality-p10, HQ-p10, targeted budgets and Googlewiki 500k, all hash-checked with unchanged production source. Actual near-rate comparisons and conservative minimum measured-chord estimates remain distinct:

| Image / preset | Strongest actual ordinary within 5% | Signed rate gap | MSE change vs actual point | Bracket estimate MSE change |
| --- | --- | ---: | ---: | ---: |
| googlewiki.png / quality | high_quality_p10 | +0.088% | -3.250% | +12.516% |
| googlewiki.png / high_quality | high_quality | -2.667% | -0.444% | +29.341% |
| monet.jpg / quality | none | - | - | +41.679% |
| monet.jpg / high_quality | high_quality | +1.281% | +33.114% | - |
| cat.jpg / quality | none | - | - | +0.728% |
| cat.jpg / high_quality | none | - | - | - |
| flower.jpg / quality | high_quality_p10 | +4.341% | +110.410% | +122.385% |
| flower.jpg / high_quality | none | - | - | - |

Googlewiki Q has a narrow actual lead versus ordinary HQ-p10: +0.088% bytes/-3.25% MSE, distinct from its much worse comparison to default Quality. Googlewiki HQ has the small actual default win. Broad brackets including the stronger500k point estimate regressions; those sparse chords are not achievable single-file controls. No extrapolation or general matched-rate improvement is claimed.

Full-call timing, one-pass feasibility observations:

| Image / preset | Ordinary encode / decode ms | Hybrid encode / decode ms | Encode change | Decode change |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png / quality | 543.86 / 47.28 | 529.28 / 54.41 | -2.68% | +15.08% |
| googlewiki.png / high_quality | 685.66 / 55.07 | 546.41 / 73.27 | -20.31% | +33.05% |
| monet.jpg / quality | 493.58 / 44.53 | 516.95 / 54.02 | +4.74% | +21.31% |
| monet.jpg / high_quality | 522.81 / 59.76 | 422.64 / 57.99 | -19.16% | -2.97% |
| cat.jpg / quality | 709.28 / 140.81 | 835.36 / 133.87 | +17.78% | -4.93% |
| cat.jpg / high_quality | 715.14 / 120.93 | 673.22 / 131.57 | -5.86% | +8.80% |
| flower.jpg / quality | 682.01 / 111.68 | 734.31 / 129.12 | +7.67% | +15.61% |
| flower.jpg / high_quality | 705.76 / 122.89 | 683.39 / 138.00 | -3.17% | +12.30% |

Observed encode changes span roughly -20% to +18%, decode changes -5% to +33%. These are not statistically confirmed speed improvements; faster poor-quality points do not become RD wins. Googlewiki HQ misses the preferred 15% decode overhead, while several other points approach it. No HF extrapolation.

Isolated decoder memory: one fresh process per HQ file, same tiny ordinary/working-DCT/P21S warmups, then one actual target decode. Pre-call/peak/final RSS retained; native peak_wset captured before hashes. Imports and allocator retention included. MiB uses 2^20 bytes.

| Image | Variant | Pre-call RSS MiB | Peak MiB | Peak above pre-call MiB | RGB MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| googlewiki.png | ordinary | 110.54 | 117.39 | 6.84 | 714.432 |
| googlewiki.png | hybrid | 110.25 | 124.50 | 14.26 | 711.262 |
| monet.jpg | ordinary | 109.75 | 119.23 | 9.47 | 53.976 |
| monet.jpg | hybrid | 109.71 | 123.48 | 13.77 | 71.850 |
| cat.jpg | ordinary | 109.61 | 151.30 | 41.68 | 293.134 |
| cat.jpg | hybrid | 109.99 | 150.80 | 40.81 | 296.166 |
| flower.jpg | ordinary | 109.75 | 151.48 | 41.73 | 34.841 |
| flower.jpg | hybrid | 109.77 | 152.12 | 42.35 | 45.216 |
| cat.jpg | p21s | 109.72 | 500.61 | 390.89 | 91.126 |

The working-resolution decoder keeps memory close to ordinary PBC on photos; small images add roughly 4-7 MiB in process peak. Cat hybrid peaks about 151 MiB versus 501 MiB for the frozen optimized P21S decoder. Their errors differ substantially (about 296 versus 91), so this is a memory-scale comparison, not matched-RD superiority. A working float coefficient buffer is around 3 MB, compared with about 102 MB for Cat original-resolution P21S. No P21S code or decoder was modified.

Validation covers 48 recorded initializer/ordinary/base-only/hybrid/JPEG files; all bytes/pixels/dimensions/MSE and repeat decodes match, with internal MSE against the true source. Eleven fixtures include odd/1x1 deterministic bases, int16 coefficient endpoints and actual additive wrap from 32767 to -32768. Thirty-eight malformed/resource/magic cases reject, including sparse entropy counts/expansion/trailing bytes, raw patch limits, channel/geometry/cell bounds and canvas overflow. Two fresh processes decode P25B without policy/source/ordinary encoding. The ordinary decoder rejects both experimental magics.

Disposition: preserve the bounded research implementation and negative result; no integration, policy redesign, extra quantizer/budget search, extension or broad benchmark. The observed bottleneck is efficient refinement from the new base, with poor generalization and expensive spatial payloads. Model distribution shift, block artifacts and rate allocation are possible explanations, not isolated causes. These data reject this fixed scalar-DCT/XZ0/legacy-refinement formulation, not every transform-initialized architecture.
