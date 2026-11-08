# Pass 47 bounded candidate oracle

## Search scope

The sample positions were fixed at steps 1, 25 and 50 for each of the three inputs and two presets. Ten of 18 states were reached: Coast Compression/Balanced at steps 1 and 25; Googlewiki Compression/Balanced at step 1; Mario64 Compression at step 1; Mario64 Balanced at steps 1, 25 and 50. Other states were unavailable because the normal encoder stopped before that step. No replacement positions were selected.

At each reached state, entries ranked 1–4 of the same canonical 50-box prescore list were passed through the model. For each box the two highest action logits were built using canonical `make_patch`, score and reconstruction functions: at most eight exact candidates. The ordinary first-box/top-logit patch was verified against the serialized control. Total evaluated candidates: 80. RNG proposal replay was exact for all ten states.

“Patch bits” below are the canonical exact pre-LZMA serialized patch cost. They are not completed-file bytes. Local score is `working SSE reduction - lambda × patch bits`, with the canonical `lambda_for_q` and preset q. One-step RGB delta is the source-resolution MSE after applying only that candidate at the sampled state, relative to applying the ordinary patch at that state; it is not the eventual final image.

| Image / preset / step | Ordinary action (cell, requested bits) | Ordinary patch bits / local score | Best alternative (box rank; action index; cell, requested bits) | Alternative bits / local score | Local score change | Patch-bit change | One-step RGB MSE change |
|---|---|---:|---|---:|---:|---:|---:|
| Coast / Compression / 1 | (4, 3) | 9,828 / 6,740,476 | rank 2; action 12; (4, 3) | 10,197 / 6,623,700 | -1.7% | +3.8% | -0.405 |
| Coast / Compression / 25 | (4, 2) | 1,109 / -104,688 | rank 3; action 26; (32, 2) | 128 / -25,996 | +75.2%* | -88.5% | +0.351 |
| Coast / Balanced / 1 | (2, 3) | 38,961 / 9,625,970 | rank 1; action 8; (2, 4) | 51,909 / 9,649,515 | +0.2% | +33.2% | -2.497 |
| Coast / Balanced / 25 | (2, 2) | 2,439 / -68,493 | rank 4; action 21; (16, 2) | 173 / -8,434 | +87.7%* | -92.9% | +0.171 |
| Googlewiki / Compression / 1 | (1, 2) | 246,127 / 36,633,627 | rank 4; action 1; (1, 2) | 227,995 / 22,767,328 | -37.9% | -7.4% | +83.034 |
| Googlewiki / Balanced / 1 | (1, 4) | 492,137 / 78,177,839 | rank 1; action 2; (1, 3) | 369,132 / 79,178,178 | +1.3% | -25.0% | +15.930 |
| Mario64 / Compression / 1 | (2, 3) | 24,885 / 27,989,597 | rank 2; action 7; (2, 3) | 62,157 / 46,030,818 | +64.5% | +149.8% | -77.763 |
| Mario64 / Balanced / 1 | (1, 4) | 130,837 / 34,895,368 | rank 2; action 8; (2, 4) | 82,837 / 56,688,496 | +62.5% | -36.7% | -40.790 |
| Mario64 / Balanced / 25 | (1, 2) | 6,567 / 107,123 | rank 3; action 2; (1, 3) | 1,629 / 241,512 | +125.5% | -75.2% | +1.231 |
| Mario64 / Balanced / 50 | (2, 2) | 279 / 723 | rank 4; action 6; (2, 2) | 333 / 6,417 | +787.2% | +19.4% | -0.053 |

`*` The ordinary local RD score is negative in these two states; the alternative is less negative but still negative, so neither counts toward the frozen gate. The large percentage is not a positive-RD opportunity.

Only Mario64 Balanced steps 1 and 25 meet the frozen local qualification: >5% improvement over a positive ordinary local score, ≤5% higher exact patch bits, and improvement at one state per count. Both are on one input. Step 1's alternative also improves immediate RGB MSE by 40.790; step 25 saves patch bits and improves the local proxy but increases immediate RGB MSE by 1.231. This is why local SSE/RD is only screening evidence.

## Computation

Each state replayed the existing 200-sample/50-retained search in about 2–6 ms. The eight-candidate oracle pass took 4.4–45.3 ms per sampled state, median 17.0 ms. Relative to one ordinary feature/model evaluation plus one patch construction, the seven additional candidates cost median 14.9 ms and maximum 30.9 ms in this runtime. These are sampled-state timings, not complete-encoder overhead. Repeating such work at many patch steps could exceed the codec's sub-second target.

The canonical encoder builds at most one patch action per state. The oracle builds up to eight and uses four model forward passes. It does not use a new detector, mask, palette sweep or patch syntax. Complete-file continuations were not run because the frozen local gate required at least three qualifying states spanning at least two images; this screen found two qualifying states on Mario64 only.

## Interpretation

The top-box/top-action policy is not locally optimal at every sampled state. It overlooked two strong local RD candidates on Mario64 Balanced. The remaining sampled states did not show a positive-score alternative satisfying the same 5% bit bound. The evidence therefore does **not** support the stronger claim that ordinary selection is near-optimal everywhere, but neither does it establish a content-general encoder opportunity. No local score, one-step MSE or raw patch-bit delta should be reported as a complete-file result.
