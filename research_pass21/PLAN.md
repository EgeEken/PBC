# Pass 21 fixed scope and coefficient gate

Start research/experiments 53410c1bc4bef78c77e31bffb36c9a466c923f0d. Preserve production, Pass17 and Pass20, model, defaults and release ZIP.

Primary: in exact Pass17 channel/frequency/block-row/block-column order, encode each nonzero as unsigned zero-gap varint followed by signed zigzag varint. Store coefficient/nonzero counts. Small Numba loops perform packing/strict parsing, with no Python loop over coefficient positions and no extra dense reordered array during sparse packing. Use conventional XZ preset0, one frame.

Initial cases: Googlewiki/Mario64/Cat q128, saved Pass20 coefficient arrays and frozen Pass17 bases. Compare dense int16 + preset0 EXTREME, dense + preset0, sparse + preset0. Warm each path; three rotated repetitions of coefficient pack/compress/decompress/unpack. Final bytes, raw bytes, every stage, total costs, exact readback and cumulative process peak working set. No Zstd or compressor-level sweep initially.

Before complete files, seek <=10% enhancement-byte increase on at least two cases, no case >15%, >=5x faster total packing/compression on all three, and no >2x coefficient unpack/decompression regression. Prefer unchanged or better decoder totals; complete decode work remains a later gate. If the primary misses, inspect measured symbol/layout costs and choose at most one alternative. Freeze it before measuring; no third layout or repeated tuning.

If a layout passes, build a distinct P21S file, retaining p10 base, original color/DCT/quantizers and exact all-coefficient reconstruction. Initial three completed files must preserve frozen RGB exactly and retain the gate's bytes/runtime behavior before extending to Monet q64, Eiffel q32, Flower q128. No per-variant quantizer tuning. Three interleaved full-call timing repetitions only on promising candidates (at most two representative pairs); record ordinary, frozen Pass17, fast dense and sparse. Decoder stage profile separates base, decompress, unpack, IDCT, color/crop, round/clip. No activation investigation unless both completed-file and runtime feasibility pass.

Hard limits for the research reader: <=16,777,216 original pixels, <=50,331,648 coefficient slots, exact coefficient/nonzero counts, bounded XZ output from declared shape, canonical bounded varints, no duplicate/out-of-bounds positions, no trailing raw/entropy data. Test int16 extremes, long runs, empty/all-zero arrays, endpoints, malformed/resource cases, odd dimensions and fresh process/model-forbidden decoding if complete files are reached.
