# Pass 17 completed-file RD results

The full-image DCT sidecar is promising on four of six images, but is not a production promotion. The current development codec remains pbc31_feature_histogram. Cat and Flower fail the studied rate range. Independent tile selection is rejected in its tested form. No alternative architecture or full-set confirmation was run.

## Actual completed files

Every hybrid uses an ordinary Quality base with patch_count=10. The table selects the best-MSE tested full-DCT point not exceeding ordinary HQ bytes; where none fits (Cat/Flower), it shows the smallest tested hybrid and marks the rate excess. Quantization choices are per-image observed operating points from the fixed 32/64/128 screen, not a learned selector or one universal preset. All byte counts include base, XZ-compressed signed coefficients and the 30-byte container.

| Image | DCT step | Ordinary HQ bytes / MSE | Hybrid bytes / MSE | Rate change | MSE change |
| --- | ---: | ---: | ---: | ---: | ---: |
| cat.jpg | 128 | 78,898 / 293.134 | 239,853 / 91.126 | +204.00% | -68.91% |
| flower.jpg | 128 | 46,094 / 34.841 | 80,752 / 22.832 | +75.19% | -34.47% |
| monet.jpg | 64 | 130,159 / 53.976 | 96,146 / 50.169 | -26.13% | -7.05% |
| googlewiki.png | 128 | 176,905 / 714.432 | 136,602 / 222.834 | -22.78% | -68.81% |
| mario64.png | 128 | 89,460 / 122.026 | 78,167 / 64.788 | -12.62% | -46.91% |
| eiffel.png | 32 | 85,874 / 15.410 | 59,265 / 10.568 | -30.99% | -31.42% |

Monet, Googlewiki, Mario64 and Eiffel have measured strict dominance over HQ: fewer final bytes and lower RGB MSE. This is evidence against these ordinary points, not proof against every possible PBC configuration. No measured ordinary point dominates those four selected hybrids. Cat/Flower trade much more bitrate for quality and are not counted as RD wins.

## Near-rate and bracketed comparisons

Ordinary controls comprise default Quality/HQ, one fresh HQ p10 point per image, the embedded Quality p10 base, and one additional size-predicted HQ budget for each promising image (Monet p8, Googlewiki p13, Mario64 p13, Eiffel p5). Those four budgets were chosen once from the p10/p20 byte sizes, without quality-based tuning or iterative retries. Requested patch budgets are not nested stream prefixes and rates are not assumed monotone. All measured controls, including dominated ones, remain in the CSV.

Strongest actually measured ordinary comparator within 5% final bytes for three useful hybrid points:

| Image / DCT step | Ordinary point | Ordinary bytes / MSE | Hybrid bytes / MSE | Signed rate gap | MSE change |
| --- | --- | ---: | ---: | ---: | ---: |
| monet.jpg / 64 | high_quality_p8 | 94,410 / 82.063 | 96,146 / 50.169 | +1.84% | -38.87% |
| mario64.png / 128 | high_quality_p13 | 81,086 / 147.774 | 78,167 / 64.788 | -3.60% | -56.16% |
| eiffel.png / 64 | quality | 48,183 / 28.757 | 50,217 / 19.879 | +4.22% | -30.87% |

Googlewiki has no comparator within 5%; its q128 file is 7.34% smaller than the measured HQ p13 file (147,416 bytes, MSE 727.003), with MSE 222.834. Its direct dominance and the measured bracket are reported separately rather than calling the files equal-rate.

Conservative bracketed estimates at the four selected hybrid rates:

| Image | Ordinary bracket | Estimated ordinary MSE | Hybrid MSE | Matched-rate MSE change |
| --- | --- | ---: | ---: | ---: |
| monet.jpg | quality to high_quality_p10 | 71.509 | 50.169 | -29.84% |
| googlewiki.png | quality to high_quality_p13 | 735.995 | 222.834 | -69.72% |
| mario64.png | quality to high_quality | 128.185 | 64.788 | -49.46% |
| eiffel.png | quality to high_quality | 24.833 | 10.568 | -57.45% |

The estimate takes the lowest interpolated ordinary MSE among all valid measured bracketing pairs, avoiding an artificially weaker curve caused by adding an inefficient intermediate control. It never extrapolates; endpoints/weights and all 18 candidate comparisons are in rd_comparisons.json. The brackets are sparse and not independent confirmation of a continuous frontier.

At ordinary HQ MSE, valid measured hybrid quality brackets exist for Monet and Eiffel only:

- monet.jpg: interpolated rate -29.72%, between DCT steps 64 and 128.
- eiffel.png: interpolated rate -36.47%, between DCT steps 32 and 64.

Googlewiki/Mario64 already use fewer bytes at better-than-HQ quality, but their tested hybrid qualities do not bracket HQ MSE. No equal-MSE interpolation or extrapolated savings is claimed for them.

## File accounting and correctness

Format P17D stores width/height, base length, enhancement length, float32 quantization step, block size, mode and tile count in a fixed 30-byte header. Full mode stores a complete ordinary v0 base followed by one XZ/LZMA2 stream. Coefficients are signed little-endian int16, ordered channel/frequency/block-row/block-column. A fixed orthonormal color transform and 8x8 type-II DCT/IDCT are part of the format definition. The decoder multiplies the stored integers by the stored step, inverts the transform, adds the correction to the decoded base, then rounds/clips final RGB. Residuals are not clipped or wrapped.

| Image / step | Base bytes | Enhancement bytes | Header bytes | Enhancement share | Residual reconstruction MSE | Final RGB MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cat.jpg / 128 | 46,135 | 193,688 | 30 | 80.75% | 91.409 | 91.126 |
| flower.jpg / 128 | 21,698 | 59,024 | 30 | 73.09% | 23.263 | 22.832 |
| monet.jpg / 64 | 42,932 | 53,184 | 30 | 55.32% | 50.046 | 50.169 |
| googlewiki.png / 128 | 57,284 | 79,288 | 30 | 58.04% | 295.659 | 222.834 |
| mario64.png / 128 | 39,797 | 38,340 | 30 | 49.05% | 76.547 | 64.788 |
| eiffel.png / 32 | 43,555 | 15,680 | 30 | 26.46% | 10.703 | 10.568 |

Unquantized float32 transform round-trip error on signed residuals: max absolute 0.000137329, MSE 9.01e-10. Quantization and final pixel clipping account for the much larger measured residual/final distortions; neither is hidden. All 30 actual full/tile files passed independent size, file-SHA, decoded-pixel-SHA and RGB-MSE checks. Signed int16 payload values spanning -32,000 to +32,000 round-trip exactly. Non-block-aligned images, repeated deterministic encodes, truncated files and research-magic rejection by the ordinary decoder passed. Fresh processes decoded Googlewiki/Eiffel using only the saved file, with policy loading forced to raise. The existing v0 decoder/source remains unchanged.

## Content-adaptive layer: negative result

A second mode uses disjoint 32x32 tiles, the same 8x8 transform and fixed step 64. Each tile is ranked by actual quantized/rounded/clipped RGB SSE reduction divided by its actual compressed bytes plus its 8-byte index/length record. Positive-gain tiles are greedily selected under the ordinary Quality or HQ complete-file budget. This is not ranking by residual SSE alone. Tile count is in the fixed header; every tile has its own XZ stream.

| Image | Quality budget MSE change | HQ budget MSE change | HQ chosen tiles | HQ metadata bytes |
| --- | ---: | ---: | ---: | ---: |
| cat.jpg | +3.20% | +3.03% | 141 | 1,158 |
| flower.jpg | +32.49% | +48.30% | 99 | 822 |
| monet.jpg | +13.17% | +29.33% | 374 | 3,022 |
| googlewiki.png | -2.77% | -38.62% | 191 | 1,558 |
| mario64.png | +85.89% | +47.27% | 133 | 1,094 |
| eiffel.png | +14.36% | +28.99% | 246 | 1,998 |

All tile files fit their target byte budgets; signed gaps remain in tiles_results.json. Only Googlewiki improved (2/12 points); the other 10 regress. Its HQ tile file is itself dominated by the smaller, lower-MSE full q128 file. This prototype loses cross-tile entropy context and pays independent XZ framing plus index records. Metadata alone is not established as the cause: the tile records, entropy framing/context and changed coefficient selection all contribute. A finer quantizer on a small tile subset also leaves much of the weaker p10 base uncorrected. Neither the rejection of this rule nor the two large-photo failures prove residuals fundamentally unpredictable. No tile-quantizer rescue sweep or alternative architecture followed.

## Runtime

Initial per-file encode/decode walls are in measurements.csv. Encode timers include base compression, base decoding, residual/color/DCT construction, quantization, entropy and container construction. Tile timers also include scoring/compressing every considered tile and selection. Decode timers include base decoding plus stored enhancement reconstruction. Image loading, external scoring, hashing, file I/O and reporting are excluded. These are CPU timings, one numerical thread, one pinned P-core.

Only two promising pairs received three warmed interleaved repetitions. They compare the actual different operating points named below, not equal-byte configurations.

| Pair | Ordinary encode / decode (s) | Hybrid encode / decode (s) | Encode overhead | Decode overhead |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png: HQ versus hybrid q128 | 0.6343 / 0.0545 | 2.5181 / 0.0847 | +297.0% | +55.5% |
| eiffel.png: Quality versus hybrid q64 | 0.2538 / 0.0244 | 0.4458 / 0.0374 | +75.6% | +53.0% |

The Googlewiki gain costs about 4x encoding time; this is not an output-preserving speed optimization. The added decoder work is a full-resolution inverse transform and extra buffers. Large-photo full q128 encodes took about 5.0 to 7.6 seconds in the one-pass feasibility screen and decoded in about 0.8 seconds, with no established RD win at the measured rates. No production-performance claim is made. Repeated timing streams reproduced the original file hashes.

## Disposition and reproduction

Freeze the full-image prototype as pbc17_dct_detail_v1 for further R&D; reject the tested independently compressed tile rule. This requires a research container/extension, not ordinary v0. Leave pbc31_feature_histogram and the staged ZIP untouched. No version/release decision was made.

Recommended next benchmark, subject to the user: 8 to 10 additional images spanning text, game renders, art, medium photographs and 8MP photographs; Quality/HQ rate region; the frozen three quantization steps and base p10; at most two size-selected ordinary budget controls per image. Include a few 500k ordinary-resolution controls in that later study so the claim is tested against stronger ordinary configurations, not only 250k budgets. This is a recommendation, not an executed benchmark. The strongest follow-up mechanism is joint base/detail allocation with a multiscale residual representation for large photographs: their lost information is often low/mid-frequency on an 8x8 original-resolution block, and a dense original-size layer is too costly at their very low source-pixel bitrates.

Reproduce with Python 3.11.16 and the pinned research requirements, restored hpt_data inputs, and no concurrent benchmark workload:

```
python research_pass17/analyze_residuals.py
python research_pass17/screen_hybrid.py
python research_pass17/screen_tiles.py
python research_pass17/rate_controls.py
python research_pass17/rate_controls.py --targeted
python research_pass17/time_promising.py
python research_pass17/validate.py
python research_pass17/report.py
```

SciPy routine definitions: [dctn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html), [idctn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idctn.html). Actual numerical dependency versions and source/model/input hashes are recorded in each measurement JSON. Generated files stay under ignored artifacts/.
