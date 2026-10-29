# Pass 25 summary

Outcome C/D: transform initialization has useful isolated capacity, but the completed transform-plus-PBC architecture fails. Keep the experiment isolated and stop before additional images, timing repetitions or redesign.

1. Ordinary standalone initialization costs 11,771/7,475/3,968/4,065 bytes on Googlewiki/Monet/Cat/Flower. It leaves source RGB MSE 1559.37/203.50/581.72/276.34. Its separately compressed size is roughly 5-14% of corresponding full Q/HQ files, not additive compressed-byte ownership.
2. The fixed 64/128/256 DCT screen passes on text plus photography: actual Cat step 128 is 14.62% smaller and 23.14% lower MSE; Googlewiki step 256 is 68.77% smaller at 1.29% more MSE. Bracket estimates at ordinary init bytes are -20.10% Googlewiki, -25.89% Cat, +11.20% Monet, -3.29% Flower. Initializer efficiency is not universal.
3. One low-rate JPEG q1 reference per image was measured; it is 26-70% below ordinary init bytes and is not a matched-rate control. Transform points can beat this low-quality JPEG point, which does not establish superiority over conventional JPEG at the final hybrid rate.
4. The actual legacy policy runs without initialization history; no fake three-patch metadata or retraining. Fresh PBC refinement from decoded step 128 bases reduces MSE 31-72% relative to the same coarse base, but costs large spatial payloads. Merely adding patches is not evidence of compression efficiency.
5. Completed results: only Googlewiki HQ strictly beats its default (-2.67% rate/-0.44% MSE). Googlewiki Q pays 45.43% more for 1.38% lower MSE; it has a narrow near-rate lead versus HQ-p10 (+0.088% bytes/-3.25% MSE). Monet Q/HQ MSE rises about 33%; Flower Q/HQ rises 86%/30% with larger files. Cat Q saves 9.81% rate at 3.22% more MSE, and Cat HQ loses both metrics.
6. The fixed working JPEG q75 point strictly dominates 5/8 hybrids. Monet HQ has only about 0.08% lower MSE than JPEG at 2.86x bytes; Googlewiki uses much more rate for lower MSE. No hybrid dominates q75 JPEG, and no JPEG equal-rate claim is supported. Flower Quality is additionally dominated by the already measured step 64 transform-only base (12,717/MSE 74.156 versus 43,121/MSE 80.801).
7. Complete one-pass encode changes are about -20% to +18%, decode -5% to +33%. Only research feasibility timing was warranted; no three-repeat speed claim or HF transfer. The fixed decoder adds working-size inverse transform and existing PBC patches, followed by normal source-size RGB output.
8. Isolated HQ decoder peaks are about 124.5/123.5/150.8/152.1 MiB for Googlewiki/Monet/Cat/Flower, close to ordinary 117.4/119.2/151.3/151.5 MiB. Frozen P21S Cat peaks 500.6 MiB but has much better MSE 91 versus this hybrid 296. Working-resolution transforms are cheaper in memory, not a proved better RD architecture.
9. All 48 measured files, 11 boundary/determinism/signed-wrap fixtures, 38 malformed/resource/magic rejections and two fresh model-free file-only decoders pass. Format P25I (24-byte header) holds the base; P25B adds 17-byte framing and ordinary generated-palette records with fixed positive-bias semantics. All coefficient/patch/entropy/side-information bytes count.
10. Initializer-only benefits do not survive refinement consistently. The existing learned spatial representation remains rate-inefficient from this fixed base; causes were not isolated to policy alone. No second base quantizer, JPEG quality or patch-budget rescue was run. Mario64/Eiffel and unseen-image generalization remain untested because the completed gate failed.
11. This tested formulation does not justify another optimization pass or integration. Preserve the smaller working-decoder insight and negative RD evidence. A future architectural choice should demonstrate complementarity against transform-only and conventional coding, rather than retain spatial patches by default. No followup was launched.

Reproduce with Python 3.11 and ../research_pass17/requirements.txt, original hpt_data, saved Pass17 streams/ordinary controls and Pass18 Googlewiki 500k control with recorded hashes. The script order below enforces the A/C gates; generated files/arrays/caches remain ignored.

```
research_pass16/.venv/Scripts/python.exe research_pass25/feasibility.py
research_pass16/.venv/Scripts/python.exe research_pass25/screen.py
research_pass16/.venv/Scripts/python.exe research_pass25/analyze.py
research_pass16/.venv/Scripts/python.exe research_pass25/memory.py
research_pass16/.venv/Scripts/python.exe research_pass25/validate.py
research_pass16/.venv/Scripts/python.exe research_pass25/report.py
```

screen.py --extend is guarded by the failed completed gate and was not run. Initial q1 and completed q75 JPEG references are separate fixed stage points, not a quality sweep. Saved source/config/model/input/file/pixel hashes, raw and compressed payload accounting, memory process records and compact measurements accompany the reports.

Practical codec/model/v0, earlier frozen experiments including P21S/Pass22, public defaults, HF and staging are unchanged. ZIP SHA256: 08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591. Only focused research source/tests/evidence is committed/pushed to research/experiments. Final verified research commit/GitHub HEAD is supplied in the chat handoff; no protected branch, release/tag or deployment.
