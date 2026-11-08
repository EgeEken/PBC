# Low-rate intervention

## Decision

Phase A supports an optional V3-style working-resolution residual stage. The prototype adds one compact coarse base stream before the existing 16×16 native-resolution V4 mode map. It does not add a new transform, boundary filter, entropy coder, model, or decoder dependency on learned weights.

## v2 representation and order

The encoder captures V3’s selected initialization/refinement patch prefix. It extracts the raw P3 syntax body, prefixes its byte length, and appends the unchanged Pass 42 native-resolution V4 payload. One outer XZ stream contains both. The v2 header records source dimensions, V3 working dimensions, steps, total raw size, and checksum. The decoder reads the coarse syntax, invokes the frozen algorithmic P3 decoder to reconstruct the full-size coarse base, then applies V4 regional SKIP/procedural/transform corrections. SKIP is defined against this final coarse base. No V4 correction writes to a SKIP region.

The input prefix is captured from frozen P3 for experiments, not a separately compressed V3 file or sidecar. The final v2 file has one shared XZ frame. The measurement therefore tests integrated stream accounting and decoding; the timing wrapper also measures generation of the coarse prefix end-to-end.

## Bounded budgets and outcomes

Three patch counts were used: initialization only (3), initialization plus three refinements (6), and the available prefix near initialization plus ten (12 or 13). Each candidate is decoded, measured as a complete file, and scored by `SSE + lambda × complete_file_bytes` for its operating point. This selects among coarse budgets; native-region choices still use the existing local V4 rate-distortion chooser and are not globally capped.

With native detail disabled, the full V3 Compression base points reproduce V3 decoded pixels and RD within 0.7–3.3% wrapper overhead on all three images. With V4-D native detail enabled, low-point selected files are Googlewiki 423,996 B / MSE 70.16, Cat 978,704 B / 143.80, and Coast 182,676 B / 12.23. The corresponding V4-A D-low controls are 73,148 B / 24.83, 779,344 B / 167.37, and 138,848 B / 18.69. This is a quality gain on Cat and Coast at 25.6% and 31.6% more bytes; Googlewiki is much worse at a far larger rate. It fails the requested near-rate improvement gate.

The useful correction is therefore a **low-rate coarse base fallback with optional V4 detail**, not an unconditional patch append. Current native-mode allocation must be improved before this counts as a general V4 RD improvement. The exact Pass 42 V1 control stream is reproduced when the coarse mechanism is disabled.

## Quality ceiling and correctness

On Googlewiki, V4-B D points move from low 423,996 B / MSE 70.16 through intermediate 737,264 B / 2.546 to high 1,225,860 B / 0.0384. Original-resolution transforms still give substantial quality gains. The v2 high point is larger and slightly worse than V4-A D-high (638,424 B / MSE 0.030), so this is a capability check, not an RD win.

Three focused test groups pass: exact batched-surface arithmetic, exact REF/OPT decode and SKIP on saved v1 streams, and a 73×59 odd-dimension v2 stream with exact base preservation under all-SKIP. Fresh-process v2 Coast decode reproduces its pixel hash and reports `skip_exact=true`.

## Comparison with Pass 31–34

Those passes tested separately encoded frozen-base sidecars and sparse transforms. This prototype instead puts coarse syntax, V4 mode map, and native parameters in one V4 payload under one XZ stream; the coarse base is the reference for V4 SKIP and corrections. It remains a limited prototype: V3 patch proposal is reused only to generate the coarse prefix, and existing V4 detail allocation is unchanged. It does not establish broad rate superiority or AVIF competitiveness.
