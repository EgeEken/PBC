# Pass 17 architectural RD hypothesis

Control: `pbc31_feature_histogram`, commit `1c9c4016fdb213d7d16663542d268c7c3df4ed9b`. Production source, the learned model, v0 decoding and release staging remain unchanged.

The 250k working canvas loses original-resolution detail. A separate original-resolution transform layer may encode some of that information more efficiently than more coarse-grid PBC corrections. Residual energy alone does not establish compressibility; acceptance requires completed independently decoded files and ordinary-PBC rate controls.

Initial fixed scope: Cat, Flower, Monet, Googlewiki, Mario64 and Eiffel; ordinary Quality and High Quality. Primary prototype: Quality base with patch_count=10, followed by an original-resolution 8x8 orthonormal DCT of the RGB residual after a fixed orthonormal color decorrelation. Test quantization steps 32, 64 and 128 only. Store signed int16 quantized coefficients in frequency-plane order, with XZ/LZMA2 compression and a separate research magic/header. All bases, dimensions, quantization and payload lengths must be in the file. No residual clipping; final RGB clipping/rounding is measured. Keep the existing v0 decoder unchanged.

If completed files show useful rate-quality overlap on multiple images, test one tile selection rule using actual quantized RGB error reduction per compressed tile byte, including tile metadata. Otherwise stop the sidecar approach and select at most one different architecture from the observed failure. No 100-case benchmark, deployment, release staging rebuild, model change or training.

The initial ordinary controls require 12 PBC encodes. The primary screen requires 18 complete hybrid encodes (six images, three quantization steps), not a parameter grid. Add only a few patch-budget ordinary controls where a promising hybrid needs an actual near-rate comparison. Any interpolation must stay inside measured nondominated brackets, with signed rate gaps retained.

## Residual evidence

All values use independently decoded original-size RGB. DCT high-frequency means u+v>=4 in an 8x8 block; remaining energy is not necessarily recoverable cheaply. The source working-resolution round trip is a diagnostic, not a distortion floor or an additive causal share; baseline.json retains the exact cross term. Spatial concentration counts source pixels only (zero padding of partial tiles).

| Image | Q / HQ MSE | Working-resolution round-trip MSE | HQ high-frequency energy | HQ error on strongest 10% edge pixels | HQ error in highest-SSE 10% tiles |
| --- | ---: | ---: | ---: | ---: | ---: |
| cat.jpg | 299.19 / 293.13 | 280.57 | 24.4% | 30.9% | 27.2% |
| flower.jpg | 43.43 / 34.84 | 22.12 | 14.0% | 58.9% | 66.5% |
| monet.jpg | 96.80 / 53.98 | 40.30 | 59.1% | 28.0% | 25.0% |
| googlewiki.png | 781.03 / 714.43 | 684.31 | 87.2% | 44.6% | 25.0% |
| mario64.png | 135.47 / 122.03 | 89.18 | 59.3% | 73.5% | 50.1% |
| eiffel.png | 28.76 / 15.41 | 2.43 | 23.5% | 18.0% | 31.0% |

Googlewiki/Mario64 retain much high-frequency and edge error. Flower has localized errors and much lower smooth-region MSE, while Cat has more distributed loss. The full smooth-region, SSE and spectral metrics are retained per preset in baseline.json. Cat loses detail at frequencies that can look low-frequency within an original-size 8x8 block because its working image is much smaller. Eiffel is already below the 250k cap: its transform-layer gains therefore also expose correction-grid representation inefficiency, not only information lost by resizing. Unlike earlier transform-patch/AVIF experiments, this prototype codes the actual decoded original-resolution RGB residual over a reduced-bit base and measures complete matched-range files.
