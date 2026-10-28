# Pass 19 summary

The signed ternary hard-mask primitive failed the frozen feasibility gate. Stop after Phase A; no complete hybrid image files or completed-file results report were produced.

1. **Existing grids:** coarse interpolation can blur edges, but this screen does not establish a broad efficiency failure. Cell1/2-bit and cell2/3-bit native grids usually match or beat the new mask at comparable actual RGB gain.
2. **Explicit signed mask:** one original-resolution Googlewiki region locally dominates a native 2-bit grid: 11.19M SSE gain/540 packet bytes versus 10.82M/652. Across 40 regions, only that one meets the material gate; working-resolution wins are zero.
3. **Joint fitting:** a fixed residual/edge shortlist tests 64/128 rectangles with four ternary-level updates. Bounded and deterministic, but not a successful joint selector; no iterative global optimizer was built.
4. **Cost:** two-bit pixel mask, 13 bytes per correction including two signed levels, eight group bytes, actual XZ framing. Individual packets 164-6,796 bytes; twelve shared groups 45,044 bytes versus 51,792 separately. Shared native controls are also measured. No per-patch final-LZMA attribution.
5. **Complete-file RD:** untested because feasibility failed. Correction packets require a known canvas/domain/dimensions; they are not complete image containers. No claimed win against default HQ, stronger rate controls or Pass 17 files.
6. **Content classes:** no convincing gain beyond one text region. Mario does not pass; Eiffel loses against native controls.
7. **Photographs:** Cat's hard-mask efficiency is below native in both domains. Original regions remove only 0.127-0.300% of base SSE each. No photographic activation or generalization demonstrated; no Flower extension.
8. **Runtime:** no full candidate overhead measurement. One original Cat proposal scan took ~0.913 s; offline scoring is deliberately expensive verification work. No repeated timings or optimization rescue.
9. **Format:** experimental signed-mask packets were serialized and decoded; ordinary v0 code/defaults are untouched. Integrating explicit masks would require a new patch syntax/container; that complete-image format was not built.
10. **Broader testing:** not recommended for this frozen method. Gate required three useful regions across two images per domain; results were 0/24 working and 1/16 original.
11. **Next direction:** the richer Pass 17 base/detail architecture with joint allocation and multiscale coding, aimed at its measured rate/runtime failures. No automatic follow-up experiment launched.
12. **Git:** only focused research_pass19 source, fixed plan, validation, compact measurements and these reports are committed. Final verified GitHub commit/HEAD is provided in the chat handoff. No protected-branch/HF/release/staging change.

Reproduce from the research checkout with Python 3.11 and requirements.txt, restored hpt_data inputs and the existing ignored Pass 17/18 frozen artifacts:

```
research_pass16/.venv/Scripts/python.exe research_pass19/feasibility.py
research_pass16/.venv/Scripts/python.exe research_pass19/validate.py
research_pass16/.venv/Scripts/python.exe research_pass19/report.py
```

On a fresh clone, generate the needed ignored inputs using the existing `research_pass18/diagnose.py`, `research_pass17/screen_hybrid.py` and its baseline/control artifact prerequisites; those historical commands encode additional cases and are not invoked automatically by Pass 19. Pass 19 performs no new full-image encoding. Frozen arrays/streams are not committed. Exact input, source/model, plan and packet digests accompany feasibility/validation; regions.csv exposes all comparisons and signed RGB gains. No full 100-case suite, model training, release rebuild or deployment occurred. Staged ZIP checksum remains 08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591.
