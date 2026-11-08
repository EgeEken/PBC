# Low-rate diagnosis

## Method

Inputs are Googlewiki (814×812), Cat (3576×2384), and Coast (4240×2832), with hashes in `outputs/v3_stage_probe.json` and `outputs/v2_rd_comparison.json`. V3.1 is the copied frozen runtime at commit `8006f30debfc4a57289df06df21636aeb52baffd` with model SHA-256 `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`. Each V3 stage is an independently framed, entropy-packed complete PBC3 prefix; its measured bytes include the prefix file’s entropy coding. V4 fields share one XZ stream. Therefore per-field raw lengths are diagnostic estimates, not additive post-XZ byte attribution.

## Where V3 gains efficiency

| Image | V3 initialization only | After first coarse residual | After first 3 coarse residuals | V3 Compression final |
|---|---:|---:|---:|---:|
| Googlewiki | 11,771 B / MSE 1559.37 | 25,323 B / 1115.80 | 27,512 B / 1034.89 | 28,974 B / 1005.02 |
| Cat | 3,968 B / 581.72 | 5,117 B / 502.28 | 6,074 B / 461.29 | 7,036 B / 433.81 |
| Coast | 1,783 B / 156.15 | 2,502 B / 122.97 | 3,142 B / 99.53 | 6,461 B / 73.13 |

V3 gains at both stages. Its generated-grid initialization is exceptionally compact, especially on Coast: 1.8 KB for MSE 156.2. Its first coarse residual patch spends another 719 completed-file bytes and lowers Coast MSE by 21%; the first three spend 1,359 more bytes and lower it by 36%. Cat’s first patch spends 1,149 bytes for a 13.6% MSE reduction. Googlewiki’s first correction is more expensive (+13.6 KB) but still cuts MSE by 28%. Thus coarse refinement is a strong photo result, not a universal fixed-cost win.

V4-A’s base is much more accurate but expensive. At the closest saved low-rate configurations, V4-A initialization-only is 32,056 B / MSE 45.23 on Coast, 97,268 B / 344.32 on Cat, and 72,352 B / 31.57 on Googlewiki. Coast’s V4 initialization costs about 18× the V3 initialization for 3.45× lower MSE. The V4 base is not “bad”; it spends far more of the low-rate budget before regional refinement begins. By the final low point V4-D is 138,848 B / 18.69 on Coast and 779,344 B / 167.37 on Cat, versus V3.1’s 6,461 B / 73.13 and 7,036 B / 433.81.

## Sources of V4-A’s regression

- **Initialization quality and cost:** V4’s 64-color per-pixel palette/index base delivers better initial quality but uses many more transmitted index values than V3’s generated coarse grids. This dominates the floor on natural photos. V3 coarse residuals then improve its base efficiently.
- **Missing coarse residual stage:** V4-A moves from its base directly to 16×16 original-resolution regions. The V3 replay shows large early gains from working-resolution residual patches before that step.
- **Transform coefficient overhead:** This is substantial when V4’s regional chooser selects many transforms. On Coast V4-B low with the coarse base, the raw transform field estimate is 54,510 B while the complete XZ file is 182,676 B; the base and region payload jointly compress to the final size. Cat and Googlewiki also select thousands of transform regions. Transform-only and mode counts are in the saved JSON; raw sizes are not post-XZ attribution.
- **Mode-map signaling:** Raw map sizes are 651 B for Googlewiki, 8,344 B for Cat, and 11,727 B for Coast in the V4-A low initialization candidate records. They are small beside transform fields and the complete payload. Map signaling is not the main regression.
- **Regional procedural representation:** Constant/surface corrections are independent 16×16 records. V4-B intermediate Coast selects 28,211 procedural regions and 2,117 transform regions. This spends many small original-resolution instructions to represent broad residual structure that V3 covers with fewer coarse patches.
- **Rate allocation:** V4’s per-region mode chooser uses local estimated costs; complete-file XZ rate is coupled across fields. Its low-point choices are not a global fixed-byte allocation. With V3 base and V4-D low, Googlewiki selects 1,572 transform regions and yields 423,996 B / MSE 70.16, despite the V4-A D-low file being only 73,148 B / MSE 24.83. Coast and Cat improve MSE with higher rates, but by 31.6% and 25.6% respectively. Mode selection and transform syntax, rather than XZ decompression or map signaling, explain much of the overspend once the base is changed.

## Low-rate recovery test

The v2 no-native-detail control serializes V3’s captured prefix body as part of the V4 payload and applies one shared XZ stream. Using the complete V3 Compression patch count (12 Googlewiki, 13 Cat, 13 Coast), the resulting V4-framed files are 29,180 B / MSE 1005.02, 6,448 B / 453.44, and 4,436 B / 89.36. Those reconstructions are pixel-identical to the corresponding V3 prefix decodes; wrapper overhead is 0.7%, 2.3%, and 3.3%. This establishes a compact V4 base path with V3-like low-rate efficiency. It does not show that V4 native-detail modes are rate-efficient: when enabled, they can multiply file size. That remains the largest unresolved RD issue.
