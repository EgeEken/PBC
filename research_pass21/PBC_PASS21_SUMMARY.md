# Pass 21 summary

The primary zero-run/zigzag representation succeeds at the sparse-storage goal. It preserves every coefficient and frozen RGB pixel while approaching or improving extreme-XZ bytes with much lower entropy cost. No alternative layout, Zstandard, multiscale scheme, quantizer sweep or production integration.

1. Before entropy, the three q128 arrays shrink from3.995/5.972/51.151 MB to186,192/81,327/593,435 bytes including counts. Raw size alone is not the win; actual XZ0 sizes remain close to extreme.
2. Format: exact Pass17 frequency-plane order, unsigned zero gaps and signed zigzag LEB128,20-byte N/K header, one XZ0 stream. Small Numba loops directly traverse/scatter; no Python per-slot loop or dense reordered packing array.
3. Extreme / fast-dense / sparse payload bytes: Googlewiki79,288/109,544/80,656; Mario38,340/51,000/34,816; Cat193,688/275,048/202,032. Full tables and packing stages are in coefficients.csv.
4. Sparse payload changes +1.73%/-9.19%/+4.31%, versus fast-dense's33-42% penalty. All coefficient gate conditions pass.
5. Median pack+compress totals16.79/10.45/134.24 ms; decompress+unpack5.49/2.87/20.88 ms. Including all packing, encode is107.8/128.7/39.5x faster than extreme coefficient coding. These are coefficient, not whole-codec, factors.
6. Every quantized int16 coefficient round-trips exactly. Zero runs, endpoints, all-zero/empty arrays, signed extremes and resource bounds pass.
7. Six P21S files decode to exactly the frozen Pass17 RGB images. Four repeated odd-size fixtures match frozen pixels,31 malformed/resource/magic cases reject, two new processes decode files with model loading forbidden.
8. Four ordinary-HQ RD dominance points remain: Monet rate-26.70%/MSE-7.05%; Googlewiki-22.01%/-68.81%; Mario-16.57%/-46.91%; Eiffel-32.29%/-31.42%. Stronger ordinary close-rate and500k controls remain explicit; no universal frontier claim.
9. Cat/Flower remain RD-negative. Cat total bytes grow3.48% versus frozen; Flower shrinks9.84%, but their rates still substantially exceed ordinary HQ. Storage success does not repair the photographic architecture.
10. Three-repeat complete encodes: Googlewiki sparse0.3738s vs frozen2.5143s and HQ0.6135s; Eiffel0.2206s vs frozen0.6068s and HQ0.5851s. Full speedups6.73x/2.75x versus frozen. Other four full calls are one-pass feasibility observations.
11. Repeated sparse decodes0.0829/0.0453s vs HQ0.0485/0.0313s: about1.71x/1.45x. Full decode remains broadly similar to frozen; small differences have appreciable variation. Cat/Flower one-pass decode is around6x ordinary.
12. This is a useful storage improvement, not a practical general-purpose default yet. No claim of major total-memory reduction; decoder still materializes dense transform/output buffers.
13. Main remaining bottleneck is original-resolution inverse transform, color/float copies and final image work. Coefficient recovery is only3-8% of the stage profiles; further entropy tuning cannot erase that work.
14. Continue only with a bounded exact-pixel decoder feasibility question (nonzero-block-aware work/fewer copies). No broad confirmation or activation policy is justified until runtime improves. If it fails, keep the hybrid opt-in or change direction; do not rescue through more sparse-format tweaks.
15. Focused source, tests, compact evidence and reports are committed to research/experiments. Final verified GitHub HEAD/URL is supplied in the chat. Practical pbc31_feature_histogram, frozen Pass17/Pass20, v0, model and staged ZIP remain unchanged.

Reproduce with Python3.11 and requirements.txt, restored hpt_data, the saved Pass17 streams at the six fixed quantizers, and Pass20's three saved q128 NumPy arrays. Frozen hashes/readback are checked before use; generated inputs/outputs are not committed.

```
research_pass16/.venv/Scripts/python.exe research_pass21/compare_coefficients.py
research_pass16/.venv/Scripts/python.exe research_pass21/screen.py
research_pass16/.venv/Scripts/python.exe research_pass21/screen.py --extend
research_pass16/.venv/Scripts/python.exe research_pass21/time_promising.py
research_pass16/.venv/Scripts/python.exe research_pass21/profile_decode.py
research_pass16/.venv/Scripts/python.exe research_pass21/time_other_controls.py
research_pass16/.venv/Scripts/python.exe research_pass21/validate.py
research_pass16/.venv/Scripts/python.exe research_pass21/report.py
```

The extension is conditional on the fixed gate in PLAN.md; it passed here. Only Googlewiki/Eiffel receive repeated full timing. Input/source/model/plan/file hashes, exact configs, resource limits and all compact rows accompany these reports. No full25x4 benchmark, HF deployment, release/tag, protected-branch change or release rebuild. Staged ZIP SHA256 remains08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591.
