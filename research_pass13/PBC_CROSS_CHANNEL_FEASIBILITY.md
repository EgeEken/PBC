# Pass 13 cross-channel feasibility

## Sample and method

The frozen PBC 3.1 release-RNG baseline, original model, and YCbCr presets were used on the eight Pass 12 images at Quality and High Quality. Each stream was generated once because no serialized Pass 9/10 streams were available locally. No codec, model, preset, serializer, or decoder was changed. The 16 streams totaled 2,974,222 raw bytes and 1,057,289 post-LZMA bytes.

The parser separated the three initialization patches from 384 iterative patches. It decoded each patch's palette values before comparing channels, reconstructed each patch's effective signed correction grid, and compared cross-channel rectangle IoU, pixel-space nonzero support, correction correlation, and patch-order distance. This avoids treating palette indices from different palettes as comparable. Approximate overlaps require resampling/alignment or a changed representation and are not counted as free savings.

## Measurements

| Measure | Result |
|---|---:|
| Cross-channel patch pairs | 2,402 |
| Pairs with any rectangle overlap | 597 (24.9%) |
| Pairs with IoU ≥ 0.5 | 56 (2.3%) |
| Exact rectangle matches | 0 |
| Exact rectangle and grid-layout matches | 0 |
| Mean rectangle IoU among overlaps, averaged per stream | 0.209 |
| Mean decoded nonzero-support Jaccard, averaged per stream | 0.367 |
| Mean decoded correction-value Pearson correlation, averaged per stream | −0.112 |
| Mean patch-order distance among overlaps, averaged per stream | 7.42 patches |

The support overlap does not imply payload equivalence: corrections are signed and their measured value correlation is slightly negative. No exact aligned groups exist, so the compatible-grid joint entropy estimate has no applicable groups and yields zero measured savings. There is no basis here for a reversible shared-support or conditional-payload coder.

## Cost estimate and decision

The extracted quantized grid indices account for 99.779% of the raw stream body bits in this sample. All remaining body bits together account for 0.221%; that remainder includes headers and patch metadata, so it is a ceiling on any metadata-only opportunity. Exact shared geometry saves zero bits on these trajectories. Pass 2 separately estimated geometry at about 0.204% of raw bits and about 0.3% final-size sensitivity in its sample; those are rough prior estimates, not additive results from this pass.

The observed trajectories provide no evidence for the required ~3% final-file reduction. Any payload opportunity would require changing patch selection, aligning or resampling corrections, and potentially changing sequential reconstruction behavior. That is a lossy representation change until pixel-identical reconstruction is demonstrated. No offline prototype was justified, and no bitrate or quality candidate was evaluated.

**Decision: stop the shared cross-channel format direction at the feasibility gate.** Reconsider only if a distinct future patch-selection study first demonstrates repeated, compatible correction support with meaningful conditional payload savings. Do not add a stream format for geometry alone.

The reproducible analysis is `run_pass13.py`; `measurements.json` contains the compact case rows and aggregate statistics. The full generated `channel_analysis.json` is local-only and can be rebuilt by rerunning the script.
