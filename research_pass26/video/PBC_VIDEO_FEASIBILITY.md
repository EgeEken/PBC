# PBC Video feasibility

The independently decodable container works. This temporal representation fails rate efficiency on every clip; preserve it as a negative result, not an effective video codec.

container.py implements P26V v1: 17-byte little-endian <4sB6H> magic/version, width, height, frame count, frame-rate numerator/denominator and key interval. Each frame has <BII> type, complete payload length and CRC32 (9 bytes), followed by normal PBC v0 (type 0), zlib signed residual (1) or PNG reference keyframe (2). Source frames/encoder state/model are not needed at decode. Limits: 64 MiB file, 512 frames, 2,073,600 pixels/frame and 32M total frame pixels. CRC/length/type/shape/trailing checks and bounded inflation apply.

Three deterministic 16-frame 256x144 RGB clips at 12 fps: repeated resized Flower, a 32x40 colored rectangle moving 9 px/frame on a fixed gradient, and a moving 256x144 crop of resized Cat (2 px horizontal/1 px vertical per frame). The photo pan is synthetic camera motion, not captured natural footage. Generated inputs/files remain in ignored ../artifacts/video.

Intra: every frame is ordinary Balanced PBC. Temporal: PBC keyframes at 0/8; subsequent prediction is the previous independently decoded frame, with little-endian int16 RGB differences [-255,255] compressed by zlib level 6. Residuals are lossless, not biased eight-bit images. Encoder reconstructs from the decoded reference and stored delta before advancing. A fully lossless PNG-keyframe/delta reference uses the same prediction/interval/container. H.264 is installed FFmpeg 6.1.1 libx264, CRF 28, veryfast, yuv420p, one encoder thread, complete MP4, no audio.

| Clip | Mode | File bytes | bpp/frame | Mean RGB MSE | Worst MSE | Encode fps | Decode fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| static_flower | intra | 102001 | 1.3835 | 75.299 | 75.299 | 9.3 | 139.0 |
| static_flower | temporal | 204845 | 2.7784 | 9.412 | 75.299 | 46.4 | 565.5 |
| static_flower | lossless | 134381 | 1.8227 | 0.000 | 0.000 | 450.0 | 1066.1 |
| static_flower | h264 | 7242 | 0.0982 | 37.120 | 37.377 | 147.7 | 229.4 |
| moving_shape | intra | 21816 | 0.2959 | 14.622 | 16.222 | 10.4 | 180.3 |
| moving_shape | temporal | 111536 | 1.5128 | 1.916 | 16.107 | 52.2 | 560.3 |
| moving_shape | lossless | 27096 | 0.3675 | 0.000 | 0.000 | 523.0 | 1106.3 |
| moving_shape | h264 | 3660 | 0.0496 | 10.311 | 18.036 | 220.1 | 307.8 |
| photo_pan | intra | 56484 | 0.7661 | 96.612 | 107.808 | 11.6 | 170.2 |
| photo_pan | temporal | 1112826 | 15.0937 | 11.697 | 97.205 | 30.3 | 474.8 |
| photo_pan | lossless | 1196462 | 16.2281 | 0.000 | 0.000 | 42.1 | 664.6 |
| photo_pan | h264 | 6132 | 0.0832 | 24.869 | 28.437 | 244.5 | 307.9 |

All metrics score every independently decoded frame and complete bytes. PBC timing includes real encode/decode and predictive reference reconstruction; FFmpeg includes process startup. One warmed feasibility pass per mode/clip, not a repeated speed benchmark. results.json retains sources/file/frame hashes, frame MSE, seconds, interval, payload ledger and residual fraction. Peak memory was not isolated for these small clips.

Temporal/intra size ratios are about 2.01x static, 5.11x moving shape and 19.70x photo pan. The first predicted frame after each lossy key must transmit a dense correction to make prediction lossless. Later static deltas become zero, but correction frames outweigh saved keys. Uncompensated motion makes photo deltas large. PNG keys can be cheaper than small lossy keys followed by lossless repairs, explaining the smaller fully lossless static/moving-shape references.

Temporal keys retain PBC distortion; every predicted frame is source-exact. Mean MSE is therefore much lower than intra at far greater rate, not an equal-quality comparison. Worst errors occur at keys and final-frame MSE is zero for temporal/lossless. No accumulating encoder/decoder drift: both advance from the same decoded reference. H.264 gives much smaller files, faster encode throughput and lower MSE than intra on all clips. Temporal has lower average MSE than H.264 at vastly more bytes; no equal-rate claim.

Nine files decode in a fresh process with PBC3.compress/model constructor forbidden and reproduce all frame hashes. Twenty malformed/header/CRC/entropy/type/range/truncation cases reject; a [-255,+255,-128] endpoint fixture reconstructs exactly. MP4 files independently decode all 16 frames. No original/cached residual is used by the decoder.

```
python research_pass26/video/experiment.py
python research_pass26/video/validate.py
```

Stop this formulation. A future project needs jointly lossy temporal residual coding with explicit quantization/RD, skip blocks and at least global/block motion; otherwise lossy-key repairs and motion dominate. This is a new research project, not a 3.2 feature or promised H.264 advantage. No interval/mapping rescue, second representation, long footage, GPU, audio or neural flow was attempted.
