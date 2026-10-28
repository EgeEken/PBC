# Pass 21 sparse coefficient representation

The primary zero-run layout passes the frozen coefficient gate. It preserves every quantized coefficient while removing most dense zero bytes before conventional XZ. No alternative layout or compressor was needed.

## Preserved information and format

Start research/experiments at 53410c1bc4bef78c77e31bffb36c9a466c923f0d. Primary runtime remains pbc31_feature_histogram; all eight runtime sources and the model are checked against immutable Pass16 source. Pass17 and Pass20 sources/results, v0 decoder, presets, policy and release staging are unchanged.

Exact original order: decorrelated color component, DCT row frequency, DCT column frequency, block row, block column. Quantization, 8x8 DCT, fixed color matrix, base p10 and all coefficient values remain unchanged. This is storage restructuring, not another quantizer, multiscale split or discarded-frequency method.

Raw sparse prefix `<4sQQ>` is 20 bytes: magic R21V, uint64 total coefficient count N, uint64 nonzero count K. For each nonzero, unsigned LEB128 encodes the number of zero slots since the previous nonzero, followed by zigzag-LEB128 for its signed int16 value. Initial previous position is -1. Gaps determine positions without coordinates; trailing zeros follow from N. Counts are mandatory even for all-zero arrays. Zero-valued nonzero symbols and noncanonical/oversized varints are invalid.

Numba traverses the original five-dimensional array directly in the required order. Packing counts nonzeros and reserves at most 8*K temporary bytes, then copies only the used prefix. It does not materialize a second N*2 reordered byte array. No Python loop visits coefficient positions. Unpacking allocates one zero-filled int16 array and writes strictly increasing nonzero positions. A view restores the original five-dimensional block geometry for the unchanged Pass17 inverse transform.

| q128 case | Coefficient slots N | Nonzeros K | Dense raw bytes | Sparse raw bytes incl. prefix | Packing scratch upper bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | 1,997,568 | 91,762 | 3,995,136 | 186,192 | 734,096 |
| mario64.png | 2,985,984 | 39,289 | 5,971,968 | 81,327 | 314,312 |
| cat.jpg | 25,575,552 | 294,461 | 51,151,104 | 593,435 | 2,355,688 |

Raw size reductions are not compression claims. The final XZ results and full pack/unpack costs are measured separately. Sparse scratch on Cat is about 2.36 MB versus 51.15 MB of dense ordered coefficient bytes; input coefficients still occupy N*2 bytes. The benchmark also retains a dense expected vector for verification outside timers. Its cumulative process high-water mark is not an isolated per-format peak. Actual full decoding still creates a dense coefficient array and large float transform/reconstruction buffers; this work is not removed by sparse storage.

## Reader bounds and correctness

Raw header counts must match container-derived N before allocation. N <=50,331,648; K<=N. Gaps use at most five bytes and unsigned32 range; signed zigzag values use at most three bytes and range 1..65535, mapping exactly to nonzero int16 [-32768,32767]. Positions cannot exceed N or repeat. The body must contain exactly K pairs with no trailing bytes.

XZ is read in two bounded stages: first the 20-byte prefix, then at most 8*K+1 body bytes. EOF, trailing/concatenated streams, exact body consumption and canonical varints are checked. A million-byte trailing expansion with zero declared nonzeros is rejected after the bounded read. Complete files additionally cap source pixels at 16,777,216, total input at 256 MiB, dimensions at uint16 range and coefficient count before decompression.

Six saved sparse files recover all frozen coefficients and RGB pixels exactly. Empty/all-zero arrays, long runs, first/last positions, signed extremes and dense random int16 pass. Four non-block-aligned repeated encodes match bytes and frozen pixels. Thirty-one malformed/resource/magic cases reject. Two new processes decode using only stored files with LearnedFiller.load forced to raise. No original image or model weight access is required by decoding.

The primary succeeds, so the conditional alternative layout and optional Zstandard comparison were not executed. There is one implemented sparse format and one conventional entropy setting. Ordinary decoding gains no new dependency; the research sparse decoder uses the existing Numba/SciPy stack and adds no package.
