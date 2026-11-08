# Pass 42 experiment design (frozen before measurements)

## Question and gates

Test whether an integrated coarse initialization plus full-resolution 16×16 mode map can improve on frozen PBC V3.1 while retaining its low-rate behavior and extending quality. The first screen uses `googlewiki.png` (graphics/text), `cat.jpg` (detailed photo), and `coast.jpg` (large natural photo). Hashes and dimensions come from `research_pass35/SOURCE_MANIFEST.json`. Reserved `cathedral.png`, `park.jpg`, and `ducklings.jpg` are not used.

Pass 26 identified a heterogeneous region representation as a plausible next architecture, while explicitly requiring shared entropy, continuous/fairly handled boundaries, and completed-file comparisons. Pass 31–34 show frozen-base refinements can improve selected images but often spend substantial extra rate and do not remove the joint-allocation problem. Pass 33's sparse DCT evidence is strongest on graphics and detailed photos, weaker on smooth photos, and has high encoder cost. Pass 35–41 do not change the V4 hypothesis. This pass keeps the brief's fixed 16×16 grid and compares procedural modes instead of forcing either one into the bitstream.

## Frozen V4 matrix

- Initialization caps: 250,000 and 750,000 pixels, 64-color median-cut palette, Pillow bicubic downsample, fixed integer bilinear decode upsample. Both candidates are evaluated through complete streams; the final candidate is selected jointly with refinement.
- Regions: fixed 16×16 pixels, row-major, clipped at image boundaries.
- Ablations: A init-only/SKIP; B init+procedural; C init+DCT; D init+procedural+DCT. All include implicit SKIP.
- Operating points: low `(DCT step 16, procedural step 8, lambda 1000)`; intermediate `(4, 2, 100)`; high `(1, 1, 0)`. No post-result parameter changes.
- DCT: 8×8 RGB blocks, all 64 coefficients allowed, fixed Q14 basis, RLE/zigzag signed varints.
- Rate: one shared XZ/LZMA preset-6 payload. Complete file bytes, not raw payload, are the rate.
- Boundary: hard region-local corrections are the default. One conditional taper diagnostic uses the same chosen modes/parameters on the smooth large-photo case and only attenuates active-region edges next to SKIP.

The first run schedule is fixed as follows: V4-D at all three operating points and all three images (nine complete-file points); V4-A/B/C at the intermediate point on all three images (nine ablation points); frozen V3.1 at all four released presets on all three images; and JPEG at all six qualities on all three images. Both initialization caps are measured inside each V4 encode and retained in the raw record. Repeat timing is limited to the large-image V4-D intermediate representative point; V3.1 runs one warmup and three measured encode/decode repetitions for each condition.

V4 `lambda` has units of approximate RGB SSE per parameter byte for local mode decisions and actual bytes for final initialization-candidate ranking. It is not a target bpp. Actual files will be reported without claiming rate matches where points are not within 5%; no broad calibration loop is allowed.

## Controls

- V3.1: frozen source commit `8006f30debfc4a57289df06df21636aeb52baffd`, model SHA-256 `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`, release RNG, four released presets. Verify source/config/model provenance before encoding.
- JPEG: installed Pillow/libjpeg-turbo, RGB input, 4:2:0 subsampling, `optimize=True`, qualities 15/30/45/60/75/90. Use measured nearest points only; no assumptions that q labels imply equal quality.
- AVIF is not required for this pass. Existing AVIF results are context only.

## Measurements

For each completed file: input hash/dimensions, bytes/bpp, RGB MSE/PSNR with uint8 promoted before subtraction, SSIMULACRA2 if its established binary is present, encode/decode timing samples, file/pixel hashes, and V4 mode/byte accounting. The V4 representative repeat has one warmup and three measured encode and fresh-process decode runs. The initial screen also records one fresh-process decode per V4 file and exact pixel equality against the encoder reconstruction. Peak memory is sampled on one large-image run. V3.1 uses the actual `PBC3.compress`/`decompress` paths from the frozen source. JPEG is decoded through Pillow and scored at source dimensions.

Correctness tests cover synthetic constant/gradient/edge/text/noise images, odd dimensions, partial regions, each/all modes, fresh-process decoding, deterministic files/pixels, exact SKIP pixels, and malformed/truncated payloads. Do not run reserved validation unless development results justify it.

## Decision rules

Gate A requires all codec correctness checks. Gate B requires lower MSE at the higher-rate setting for the same mode/initialization family and demonstrable source-resolution detail recovery. Gate C uses the brief's initial screen of roughly 10–15% lower MSE at comparable complete-file rate across at least two content classes, or comparable size reduction at similar MSE. Gate D reports speed and memory, treating >20–25% slowdown against frozen V3.1 as a warning. Gate E reports where JPEG dominates/is dominated, including unmatched rates. Gate F is not run unless the development architecture and parameters are frozen and results warrant the reserved set.

The test set is small and previously used in the project. Results are exploratory, not a release claim. Failed gates stop expansion rather than triggering a broad parameter search.
