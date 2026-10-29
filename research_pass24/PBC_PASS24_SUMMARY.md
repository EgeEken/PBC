# Pass 24 summary

Stop after initial completed-stream feasibility. Eight saved ordinary Q/HQ files, 80 structurally selected iterative omissions out of 233 available patches. All initialization patches remain unchanged. Every test uses actual production LZMA repacking and independent ordinary v0 reconstruction against the original RGB source.

1. Yes, sampled completed streams contain patches whose removal lowers RGB MSE: 7/80 (8.75%). Only 3/80 (3.75%) also shrink the final file. No exact MSE ties.
2. Strict savings are Googlewiki HQ index 10: 97 bytes (0.0548%), Flower Quality index 36: 14 bytes (0.0483%), Flower HQ index 13: 74 bytes (0.1605%). MSE reductions are 0.00128%, 0.00871%, 0.03577%. These are real but tiny pointwise wins, not a meaningful codec improvement.
3. No omission saves 5% within 1% MSE growth. Best low-loss trade is Googlewiki Quality: 3.213% fewer bytes/+0.688% MSE. Other images stay below 0.5% savings at that tolerance. Cat HQ index 8 saves 10.492% at +1.172% MSE, just outside the tolerance; compare it with ordinary HQ-p10 before interpreting that larger rate change.
4. Early/costly/high-depth patches generally remain important. Late omissions are usually low-impact but save very few bits. High later overlap does not imply redundancy: its sampled group has larger median omission MSE damage. All strict wins use 2-3 effective bits; none of 58 depth 4+ omissions improves MSE. These are descriptive biased-sample associations, not causal findings or a validated predictor.
5. Actual LZMA context can reverse raw savings: removing a 689-bit Googlewiki patch grows the file 43 bytes despite slightly improving MSE. Final bytes, not raw patch cost, determine every result.
6. Nearby ordinary controls leave only tiny leads. Cat HQ index 8 is +0.136% bytes/-0.619% MSE versus actual HQ-p10; its ordinary bracket estimate is slightly unfavorable. Googlewiki Quality has an estimated 0.84% matched-rate MSE gain, not 5-10%. Strict omissions dominate their actual default points by negligible amounts; broader-frontier claims are unsupported.
7. Combined removal behavior is untested. The predeclared meaningful-omission gate fails (no strict saving reaches 0.5%; no 5%-saving/1%-loss point), so no greedy multi-removal candidate, working-SSE proxy, integrated postprocessor, Monet/Eiffel extension or repeated rejected-candidate timing.
8. Cheap working-space predictability and prediction cost are therefore not established. Stored-cost/position/overlap statistics alone did not reveal meaningful removable rate. No oracle result is promoted as deployable.
9. Full encode-plus-pruning overhead is unmeasured; no image encoding ran. Exact oracle evaluation costs about 136-354 ms per omission median, 1.32-3.50 s for ten omissions per file, excluding structural selection/I/O. No decode speedup is established; measured individual decode spans are not matched repeated before/after benchmarks.
10. The hypothesis is not supported as a practical cross-content improvement on this bounded sample: graphics retains useful corrections, photographic strict wins are tiny and only Flower, and the larger Cat trade is not a clear frontier improvement. No integration or broad validation is warranted.
11. This establishes little exploitable sampled single-patch redundancy under fixed initialization, not globally optimal PBC allocation or absence of joint effects. Keep existing codec/allocation results intact; do not rescue this result with another pruning algorithm.
12. Eight exact original roundtrips and all 80 omitted streams pass byte/count/order/initialization validation; six fresh practical/release decoder processes reproduce the strict-file pixels with model loading and image encoding forbidden. Source/model/input/config/stream/pixel hashes and signed per-patch measurements are recorded.

Reproduce from repository root with the existing Python 3.11 environment, ../research_pass17/requirements.txt, restored source images and saved ordinary Pass17 streams. Prior ordinary comparisons require the preserved Pass23 comparisons.json control manifest. Generated omitted streams remain ignored.

```
research_pass16/.venv/Scripts/python.exe research_pass24/omissions.py
research_pass16/.venv/Scripts/python.exe research_pass24/analyze.py
research_pass16/.venv/Scripts/python.exe research_pass24/validate.py
research_pass16/.venv/Scripts/python.exe research_pass24/report.py
```

The second requested pruning-results report is intentionally absent because no pruning stage beyond initial feasibility ran. Raw per-patch values/ledgers, exact selected indices, CPU/thread/warmup protocol and component timers accompany this summary. The 80-row CSV is compact evidence, not a large benchmark log.

Practical pbc31_feature_histogram, model, v0, P21S/Pass22, frozen earlier results, HF and staging remain unchanged. ZIP SHA256: 08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591. Focused source/tests/evidence are committed only to research/experiments; final verified GitHub commit and HEAD are supplied in the chat handoff. No protected branch, deployment, release/tag or staging change.
