# Pass 20 summary

Stop the frozen multiscale direction after twelve completed files on Googlewiki, Mario64 and Cat. No six-image extension, three-repeat candidate timing, activation classifier, production change or release rebuild.

1. Pass 17 cost: extreme XZ accounts for about 81-87% of measured disjoint encoding spans. Dense zero-heavy coefficient arrays still require copies/scans; base decode and full inverse transforms add work.
2. DCT versus entropy: DCT ~18/32/316 ms versus extreme XZ ~1.98/1.55/6.24 s in the instrumented Googlewiki/Mario/Cat calls. Separately measured complete calls are retained, not synthesized by adding overlapping profiles.
3. Exactly two settings: preset0+EXTREME versus normal preset0. Normal compression is 22-54x faster, but enhancement size rises 33-42%. Full normal P17 files preserve pixels and encode in about 0.366/0.467/1.708 s in one-pass observations; this rate/runtime tradeoff is not a free improvement or production promotion.
4. Coarse layer: reduced scale4 residual coding is compact on text but cannot cover all low/mid 8x8-band information. Cat coarse prefix alone is ~59 KB plus a 46 KB base; it does not solve its low-rate requirement.
5. Fine layer: improves actual MSE but consumes much of the savings. Omitted low bands still hold 16.70%/48.60%/66.44% of remaining residual energy. Restoring them in an unencoded diagnostic greatly improves MSE, with no proven byte cost or RD repair.
6. Versus frozen Pass17: nine files are strictly dominated by full frozen DCT controls; all twelve are dominated when the actual predeclared A-band repacks are included. The latter remain separate diagnostic controls, not relabeled frozen full architecture.
7. Versus ordinary: p10 lean dominates default HQ on Googlewiki and Mario and improves MSE against strong actual near-rate controls. Googlewiki also dominates its available 500k HQ control. These wins do not overcome worse RD than Pass17.
8. Content outcomes: two small positive cases retain ordinary wins, Cat remains expensive. Monet/Eiffel/Flower were not tested because the fixed extension gate failed; no claim that Flower is repaired.
9. Allocation: p5 saves bytes with worse quality; it never dominates p10 here. Budgets are actual fresh encodes and are not nested prefixes. No budget sweep.
10. Runtime: full calls, one-pass only. p10 lean encode/decode ratios versus ordinary HQ are roughly 0.70/2.17 Googlewiki, 1.00/3.64 Mario, 3.44/8.88 Cat. These are unrepeated feasibility observations, not precise benchmark estimates. Decoder targets fail; Cat encode remains >2x.
11. Activation: not evaluated after failed RD gate. No classifier, image-name switch, second-full-encode routing or oracle policy. Dimensions/sparsity alone are not established predictors.
12. Broader validation: not recommended for this representation. All completed files pass decoding/readback/determinism/framing checks. Existing pbc31_feature_histogram, pbc17_dct_detail_v1, v0 model/defaults and staged release are preserved.
13. Next: reduce entropy cost without sacrificing most of the extreme preset's compression, and preserve residual information with a filter-compatible scale split or efficient sparse single-scale representation. No rescue preset/transform/quantizer sweep launched.
14. Research GitHub commit and verified final HEAD are supplied in the chat handoff. Only research_pass20 source, fixed plan, reports and compact evidence are committed; generated streams/coefficient arrays/caches remain ignored.

Reproduce using Python 3.11 and requirements.txt with restored hpt_data and the existing Pass17 saved q128 files (or regenerate those archived artifacts using the prior scripts):

```
research_pass16/.venv/Scripts/python.exe research_pass20/profile_costs.py
research_pass16/.venv/Scripts/python.exe research_pass20/screen.py
research_pass16/.venv/Scripts/python.exe research_pass20/finish_costs.py
research_pass16/.venv/Scripts/python.exe research_pass20/time_controls.py
research_pass16/.venv/Scripts/python.exe research_pass20/validate.py
research_pass16/.venv/Scripts/python.exe research_pass20/report.py
```

Do not run screen.py --extend unless the fixed gate passes in a separately justified future study; it was not run here. Source/model/input/plan hashes accompany measurements, full rows are in files.csv, controls/brackets in comparisons.json. Memory fields are cumulative process peak working sets. Staged ZIP SHA-256 remains 08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591. No HF operation, protected-branch change, release/tag or full 25x4 benchmark.
