# Competitive status

Pass 47 produced **no completed experimental PBC file and no demonstrated completed-file improvement** over canonical V3.1. The predeclared local gate failed its cross-image threshold, so the conditional continuation and selection-rule phases were not run. The oracle did find two large local RD-score alternatives, both on Mario64 Balanced; their effect after the remaining patch trajectory is unknown.

## 1. Canonical control correctness

The copied published release source and pinned model passed the instrumentation-only control on all six image/preset cases. Uninstrumented and instrumented streams, parsed patch sequences and decoded RGB pixels matched exactly. All six source hashes were checked. Three encode and decode timings were recorded for each canonical control, and each saved file also passed an independent fresh-process pixel-hash check. Stream and pixel hashes exactly match Pass 46's canonical rows.

The runtime was Python 3.13.5 / NumPy 2.2.6 / Numba 0.62.1 / Pillow 11.3.0, unlike Pass 45/46's recorded Python 3.11.16 / NumPy 2.4.6 / Numba 0.68.0. Exact output hashes matching across these environments provide a stronger compatibility check than expected, but timings remain specific to the current local runtime.

## 2. Local oracle results

The published path draws 200 random spatial boxes per step, retains up to 50 by prescore, passes only the first box to the model, predicts among 30 native actions, and reconstructs one action. In the fixed early/middle/late sample, 10 of 18 states were reached and 80 native patch candidates were scored (four retained boxes × two model-ranked actions). The exact same proposal pools and post-proposal RNG states replayed.

Two states passed the frozen local criterion, both on Mario64 Balanced: step 1 had 62.5% higher local RD score with 36.7% fewer patch bits; step 25 had 125.5% higher score with 75.2% fewer bits. At step 1 the candidate's one-step RGB MSE was 40.790 lower than the ordinary patch. At step 25 it was 1.231 higher despite the better local score. These are single-state diagnostics; the second result illustrates that the bit-penalized local score can prefer a less-distorting-efficient patch.

No other image/preset had an alternative with both a positive ordinary score, at least 5% higher score, and no more than 5% patch-bit increase. Negative ordinary scores at two late Coast states make their large relative percentages ineligible; their alternatives did not have positive local RD scores.

## 3. Completed-file counterfactuals

Not attempted under the frozen design: two qualifying states on one image did not meet the requirement of at least three qualifying states across at least two inputs. Consequently, there are no experimental complete-file byte counts, bpp, final MSE, SSIMULACRA2 scores, encode/decode timings, pixel hashes or AVIF comparisons to report. No hindsight oracle is presented as a practical encoder.

## 4. Perceptual quality and AVIF context

No source-aligned alternative continuation crops were produced because no continuation was selected. Immediate RGB MSE shows mixed local effects and is not a perceptual validation. The canonical reference quality scores are the validated Pass 46 SSIMULACRA2 values; their pixel hashes are identical in this pass.

Canonical and AVIF reference points are reproduced below for context only. The two quality-0 profiles are distinct; qcolor 0 is not a fixed bitrate. Near-rate AVIF points are included when available. The five near-rate rows had lower AVIF RGB MSE and higher SSIMULACRA2 than PBC on these three development images. This does not establish broad codec superiority. AVIF timing includes CLI startup and image I/O, so no speed comparison is made.

| Image / PBC point | Codec / point | Bytes | RGB MSE | SSIMULACRA2 |
|---|---|---:|---:|---:|
| Coast / Compression | Canonical PBC 3.1 | 6,461 | 73.134 | -45.35 |
|  | Pillow AVIF quality 0 | 7,647 | 72.378 | -18.90 |
|  | Pinned AVIF qcolor 0 | 6,747 | 88.537 | -26.68 |
|  | Pinned AVIF qcolor 7 (near Balanced) | 11,260 | 31.277 | 1.79 |
| Googlewiki / Compression | Canonical PBC 3.1 | 28,974 | 1,005.022 | -25.69 |
|  | Pillow AVIF quality 0 | 8,299 | 1,061.660 | -42.50 |
|  | Pinned AVIF qcolor 0 | 5,218 | 1,390.609 | -55.96 |
|  | Pinned AVIF qcolor 12 (near Compression) | 29,727 | 498.360 | 2.41 |
|  | Pinned AVIF qcolor 28 (near Balanced) | 55,025 | 291.464 | 40.55 |
| Mario64 / Compression | Canonical PBC 3.1 | 14,099 | 414.236 | -50.69 |
|  | Pillow AVIF quality 0 | 4,425 | 418.190 | -55.07 |
|  | Pinned AVIF qcolor 0 | 4,139 | 580.138 | -57.30 |
|  | Pinned AVIF qcolor 20 (near Compression) | 14,504 | 154.805 | 9.43 |
|  | Pinned AVIF qcolor 50 (near Balanced) | 39,785 | 63.726 | 55.04 |

PBC encoder times (three-run medians) were 0.136–0.437 s; decoder medians were 0.019–0.095 s in the current runtime. AVIF CLI timing boundaries are not comparable, so these values are not a speed ranking.

## 5. Computation and selection rule

Four boxes/eight candidates required 4.4–45.3 ms per sampled state, median 17.0 ms. The seven extra candidate evaluations beyond one ordinary patch cost an estimated median 14.9 ms and maximum 30.9 ms per state. This is not a full-encoder measurement and no deployable rule was evaluated. The screen does not justify adding multi-candidate search to the encoder.

## 6. Gate decisions

| Gate | Result | Reason |
|---|---|---|
| A — Canonical control | Pass | Six same-runtime uninstrumented/instrumented byte, patch-sequence and pixel matches; deterministic repeats; fresh-process decode hashes match |
| B — Local candidate opportunity | Fail | Two qualifying states, both on Mario64; frozen gate required at least three across at least two inputs |
| C — Completed-file opportunity | Not attempted | Conditional on Gate B; no full continuation outputs exist |
| D — Perceptual relevance | Not established for alternatives | No candidate continuation or visual crop; one-step RGB effects are mixed |
| E — Practical selection rule | Not attempted | No completed-file improvement to approximate |
| F — Competitive result | Not applicable to experimental candidate | No integrated candidate; canonical/AVIF references are contextual only |

## 7. Recommendation

Do not change the PBC encoder from this pass. The oracle disproves a blanket claim that the learned top choice is always close to the best local native action, but the observed opportunities are confined to two states on one image and have not been tested through full-file continuation. Do not start a generic reranking or proposal heuristic. If a further decision is made later, the only justified follow-up is a tightly bounded continuation test of these captured Mario64 Balanced alternatives; it must not be treated as an integrated rule without a repeatable completed-file win and measured compute cost. No Pass 48 was started.
