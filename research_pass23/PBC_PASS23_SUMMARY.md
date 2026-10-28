# Pass 23 summary

Stop after the four-image Quality/HQ screen. The target-only hypothesis succeeds; the tested encoder-only method fails the completed-file progression gate. No additional images or repeated timing campaign.

1. Ordinary YCbCr downsampling is suboptimal for these tested quantized decoder-equivalent reconstructions; this is not a universal optimality claim. Conservative back-projection reduces target-only MSE by 4.95% Googlewiki, 7.73% Mario64, 3.04% Cat, 11.26% Flower. The stronger predeclared target-only setting improves 7.71%, 13.26%, 5.97%, 19.39% respectively.
2. Conservative step 0.5/one iteration/8-level total correction cap was selected by the frozen rule. The research prepare subclass changes only arr/target, preserves true-source orig_compare, and successfully produces ordinary v0 files. Stronger was not encoded as a rescue.
3. Completed-file behavior is mixed: Cat Quality improves final size 9.33% and RGB MSE 2.41%; Googlewiki/Mario64/Flower HQ improve MSE 6.09%/6.65%/6.40% for 3.34%/1.47%/1.96% more bytes. Googlewiki and Mario64 Quality worsen both size and MSE; Flower Quality costs 40.04% more bytes for 10.68% lower MSE. Cat HQ costs 18.94% more for 4.21% lower MSE.
4. No universal 5-10% matched-rate improvement is established. Cat Quality has an estimated 4.65% matched-rate MSE gain. Three Quality estimates regress; Googlewiki HQ also loses against the conservative bracket including the saved stronger 500k control. Other HQ points have no measured enclosing ordinary bracket; their close-rate gaps remain explicit. Estimates are not achievable equal-rate files.
5. One strict same-preset RD win and 9.84% mean rate increase fail the frozen gate. The target-only MSE signal does not predict which preset will yield a good compressed result. No classifier or dual-encode activation rule is justified by four images; no activation was built.
6. Full one-pass encoding overhead is 7.1-15.6% on graphics, 53.8-58.9% on photos. Preconditioning alone costs about 80-90 ms graphics and 437-505 ms photos. This is substantial relative to ordinary encoding. Decoder syntax/computation is unchanged; actual payload-dependent decode times were measured, with one-pass variation explicitly retained.
7. All 16 actual files preserve v0 compatibility and correct original-source internal/external MSE; 12 prepare fixtures and four model-free practical/release subprocess decodes pass. Source/input/model/config/stream/pixel hashes, real bit-depth counts and full/raw byte sizes are recorded.
8. Monet/Eiffel generalization is untested because the initial completed-file gate failed. The stronger target setting, content classifier, large sharpening search, per-patch RGB rescoring, broad 100-case benchmark and another encoder optimization were not attempted.
9. The observed limitation is converting a better ideal target into efficient learned-patch/LZMA representation, plus full-source projection cost. Target variation and changed fitting/patch counts support this interpretation; causal attribution to learned policy alone is not established.
10. Keep the isolated Cat Quality and close-rate HQ leads as evidence, but do not broaden or integrate this method. A distinct future direction is native residual rate allocation using measured patch gain and completed-stream marginal cost, rather than further target sharpening or model retraining. No followup was launched.

Reproduction: same Python 3.11 research environment and ../research_pass17/requirements.txt, restored hpt_data, saved ordinary Pass17 controls and the Pass18 Googlewiki 500k control matching their recorded hashes. Run from repository root:

```
research_pass16/.venv/Scripts/python.exe research_pass23/feasibility.py
research_pass16/.venv/Scripts/python.exe research_pass23/screen.py
research_pass16/.venv/Scripts/python.exe research_pass23/compare.py
research_pass16/.venv/Scripts/python.exe research_pass23/validate.py
research_pass16/.venv/Scripts/python.exe research_pass23/report.py
```

screen.py enforces the Phase A selection. screen.py --extend enforces the failed C gate and was not run. The reproduction commands regenerate only ignored streams and this pass measurements/reports; they do not change production. Raw numerical rows are in feasibility.json/screen.json/comparisons.json; reports separate actual files, near-rate observations and interpolated estimates. Full config/model identity is recorded per case.

Practical codec, v0, model, frozen Pass17/21/22, HF and staging remain unchanged. ZIP SHA256: 08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591. Focused research-only commit/push to research/experiments is authorized; the final verified GitHub commit/HEAD is supplied in the chat handoff. No protected branch, release/tag, deployment or staging changes.
