# Decoder optimization

The isolated implementation in `pbc4_decode_opt.py` changes only the 16×16 procedural interpolation kernel. It replaces Python row loops and per-row temporary construction with a precomputed integer bilinear-weight table and one integer matrix multiplication, preserving the original signed rounding rule. It does not change the stream, transform math, palette/base reconstruction, upsampling, region order, or SKIP handling.

Exactness was checked over zero, signed-limit, and 200 random control grids, plus both saved Coast and Googlewiki v1 streams. REF and OPT pixels and decoder-info records match. Fresh-process decode hashes also match. The odd-dimension v2 test separately verifies shape and protected SKIP.

| Stream | REF median | OPT median | Speedup | RSS delta REF → OPT |
|---|---:|---:|---:|---:|
| Coast, five warmed interleaved repeats | 14.305 s | 4.271 s | 3.35× | 76.68 → 76.41 MB |
| Googlewiki, five warmed interleaved repeats | 0.254 s | 0.258 s | 0.98× | 8.29 → 8.25 MB |

Coast fresh-process measurements were 15.13 s REF and 4.21 s OPT, with identical hash `37add1c691bf25a025a6ca6c002bb4e0f46750ead38083f397a617ff7fbe0656`. Googlewiki fresh-process timings were 0.222 s and 0.215 s, with identical hash `e57e01b2f1f2cd89c3a48a6d08a77540ec4687857f4504b440dd02d4c10c6f92`. Memory change is negligible. The optimization helps when many procedural regions are present; it does not improve the small image.
