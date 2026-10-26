# Frozen-trajectory entropy research

No screened lossless representation or LZMA setting establishes a general final-size improvement. Both new grid transforms increase every source/preset aggregate after LZMA. They are rejected as general coding candidates; no full-main or holdout entropy candidate was promoted. This is a bounded six-image/four-preset screen for both immutable release and current root, not proof that every entropy representation is exhausted.

## Bitstream accounting

All 48 input streams are frozen Pass 1 files; searches are not rerun. Exact accounting includes the six-byte wrapper, image header, patch coordinates/channel, generated-palette flag, mask length and mask, signed bounds, maximum bitcount, 16-bit cell size, quantized indices, and final 0–7 padding bits. Grid dimensions are implied by geometry/cell size, not separately serialized. Parsing asserts the sum equals the complete raw bitstream length.

| Component | Total raw bits | Raw share % | Zeroing effect on final bytes % |
| --- | --- | --- | --- |
| bounds_bits | 20592 | 0.0494 | 0.0780 |
| cell_size_bits | 20592 | 0.0494 | 0.0225 |
| geometry_bits | 84942 | 0.2037 | 0.3028 |
| image_header | 6624 | 0.0159 | 0.0326 |
| indices_bits | 41548013 | 99.6171 | 98.9198 |
| mask_bits | 5148 | 0.0123 | 0.0189 |
| mask_length_bits | 12870 | 0.0309 | 0.0150 |
| max_bitcount_bits | 5148 | 0.0123 | 0.0112 |
| representation_flag_bits | 1287 | 0.0031 | 0.0000 |
| wrapper | 2304 | 0.0055 | — |
| final_padding | 192 | 0.0005 | — |

The compressed column is a nonadditive diagnostic: replace only that component's bits with zeros at their original positions and recompress. These deliberately invalid streams are never decoded. Context/match changes make these effects nonadditive and occasionally negative; they are not a unique allocation of compressed bytes to fields. A large raw component alone is not evidence that a replacement will save final bytes. Exact raw and post-LZMA sizes, per-patch symbol counts by channel/preset, zero indices versus zero palette values, horizontal/vertical correlation and repeat fractions are retained in bitstream_accounting.json, symbol_distributions.json and index_statistics.json.

Unweighted patch means in C1 were approximately 55.0% zero indices, 58.1% horizontal repeats, 59.0% vertical repeats, and correlations 0.390/0.402. These statistics motivated two independent experiments; they did not predict a guaranteed compressed-size gain.

## Exact lossless representations

- `horizontal_delta`: leave the first index of each row unchanged; subsequent symbols are current minus left, modulo 2^bitcount. Decode with row-wise cumulative sums modulo the same alphabet.
- `column_order`: serialize a column-major traversal in the same fixed-width symbol slots; reverse that permutation before palette lookup.

Each changes only grid symbol representation, keeps header/palette/geometry semantics intact and has no side information beyond existing dimensions/bitcount. Research-only magic `P2D1` / `P2T1` identifies the alternate format (layout tag 0, existing entropy byte and body layout); the six-byte outer framing is included in all sizes. AlternateCodec in entropy.py provides the matching decoder. Production v0 decoding remains intact and explicitly rejects both research magics. Every trial recovers exact quantized indices and exact decoded RGB pixels. No combined predictor/traversal/RLE was tested.

Current-root stream results (release results also retained, with the same aggregate direction):

| Preset | Method | Bytes | bpp | Bytes % | Image wins / 6 | Repack s | Decode s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | control | 27183 | 0.1553 | 0.0000 | 0 | 0.0327 | 0.0735 |
| balanced | horizontal_delta | 28006.8333 | 0.1592 | 3.0307 | 1 | 0.0320 | 0.0716 |
| balanced | column_order | 28614.5000 | 0.1723 | 5.2662 | 3 | 0.0321 | 0.0731 |
| compression | control | 14266.5000 | 0.0839 | 0.0000 | 0 | 0.0183 | 0.0600 |
| compression | horizontal_delta | 14651.5000 | 0.0859 | 2.6986 | 2 | 0.0186 | 0.0616 |
| compression | column_order | 14937.8333 | 0.0945 | 4.7057 | 2 | 0.0169 | 0.0594 |
| high_quality | control | 98008.3333 | 0.6323 | 0.0000 | 0 | 0.1465 | 0.1288 |
| high_quality | horizontal_delta | 100693.5000 | 0.6541 | 2.7397 | 2 | 0.1451 | 0.1335 |
| high_quality | column_order | 103043.5000 | 0.6860 | 5.1375 | 2 | 0.1472 | 0.1348 |
| quality | control | 57618.6667 | 0.3024 | 0.0000 | 0 | 0.0634 | 0.0875 |
| quality | horizontal_delta | 58586.3333 | 0.3107 | 1.6794 | 2 | 0.0627 | 0.0905 |
| quality | column_order | 59675.1667 | 0.3227 | 3.5692 | 3 | 0.0596 | 0.0901 |

Each point retains unchanged original decoded MSE in entropy_screen.json. Repacking time includes serialization, optional transform and entropy packing of frozen patches; it excludes original search. Full decode is independently timed with the matching decoder. This is not a measured complete new encoder latency. Original full encode/decode and expanded configuration are linked in each row. No full-codec speed claim is inferred by adding differently timed historical search to repack time. The new inverse adds cumsum/permutation work and temporary arrays; it has no learned dependency.

## Bounded LZMA2 sweep

Production is raw LZMA2, `preset=PRESET_EXTREME`, meaning preset 0 plus extreme, with a 256 KiB dictionary. Independent alternatives are preset 1 (1 MiB), preset 6 (8 MiB), and production settings with explicit 64 KiB dictionary. Dictionary sizes are verified from the installed liblzma filter-property encoding in the memory probes. Preset 1 reduces entropy-pack latency but costs rate. Preset 6 is near neutral/slightly larger in aggregate and can emit references beyond the production decoder's dictionary. The 64 KiB option is a memory/rate tradeoff with negligible or adverse size changes.

| Preset | LZMA method | Bytes | bpp | Bytes % | Pack s | Unpack s | Full decode s | Existing decoder compatible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | production_0extreme | 27183 | 0.1553 | 0.0000 | 0.0128 | 0.0016 | 0.0708 | True |
| balanced | preset1 | 28419.1667 | 0.1608 | 4.5476 | 0.0042 | 0.0015 | 0.0720 | True |
| balanced | preset6 | 27223.3333 | 0.1554 | 0.1484 | 0.0165 | 0.0015 | 0.0709 | True |
| balanced | preset0extreme_dict64k | 27183.3333 | 0.1553 | 0.0012 | 0.0122 | 0.0015 | 0.0727 | True |
| compression | production_0extreme | 14266.5000 | 0.0839 | 0.0000 | 0.0066 | 0.0008 | 0.0613 | True |
| compression | preset1 | 14641.5000 | 0.0858 | 2.6285 | 0.0027 | 0.0008 | 0.0611 | True |
| compression | preset6 | 14298 | 0.0840 | 0.2208 | 0.0100 | 0.0008 | 0.0603 | True |
| compression | preset0extreme_dict64k | 14266.5000 | 0.0839 | 0.0000 | 0.0061 | 0.0008 | 0.0622 | True |
| high_quality | production_0extreme | 98008.3333 | 0.6323 | 0.0000 | 0.0802 | 0.0052 | 0.1344 | True |
| high_quality | preset1 | 103748.3333 | 0.6675 | 5.8566 | 0.0160 | 0.0057 | 0.1307 | False |
| high_quality | preset6 | 98067 | 0.6335 | 0.0599 | 0.0672 | 0.0051 | 0.1283 | False |
| high_quality | preset0extreme_dict64k | 98246 | 0.6343 | 0.2425 | 0.0732 | 0.0050 | 0.1298 | True |
| quality | production_0extreme | 57618.6667 | 0.3024 | 0.0000 | 0.0275 | 0.0031 | 0.0870 | True |
| quality | preset1 | 59752 | 0.3127 | 3.7025 | 0.0083 | 0.0031 | 0.0906 | True |
| quality | preset6 | 57687.6667 | 0.3030 | 0.1198 | 0.0305 | 0.0031 | 0.0896 | True |
| quality | preset0extreme_dict64k | 57637.1667 | 0.3024 | 0.0321 | 0.0269 | 0.0031 | 0.0893 | True |

All entropy round trips reproduce the exact raw body, and matching-filter complete decodes reproduce exact pixels. Existing-decoder compatibility is tested separately on every case. An incompatible row's six-byte wrapper size is a diagnostic configuration comparison, not a deployable v0 file: no dictionary metadata extension was designed or accepted, and an actual new self-describing format would need its full overhead counted. Incompatible variants are rejected for existing-wrapper use. The exact production control reproduces original compressed bytes. LZMA STORE fallback remains active when compression is not smaller.

Separate process memory probes, 4 MiB repeated frozen raw body for allocation visibility:

| Filter | Dictionary bytes | RSS construct delta | RSS after pack delta |
| --- | --- | --- | --- |
| preset0extreme_dict64k | 65536 | 28672 | 3035136 |
| preset1 | 1048576 | 2383872 | 8704000 |
| preset6 | 8388608 | 17068032 | 54972416 |
| production_0extreme | 262144 | 815104 | 5427200 |

RSS is observed allocation behavior for this synthetic memory probe, not an exact liblzma memory bound or an image RD result. Smaller encoder dictionaries do not automatically reduce RAM in the unchanged production decoder, which still uses its configured dictionary. All settings are warmed and rotated, three repeats per frozen case. Entropy-pack and entropy-unpack timings remain separate from whole decode and historical search.

The memory fixture artificially repeats a complete raw body across 4 MiB; large dictionaries can exploit those artificial long-distance repeats. Its packed size is not evidence of a single-image bitrate improvement.

## Extreme-compression decision

Neither transform retains savings at Compression or Balanced; both raise final bitrate at unchanged MSE. Obvious symbol repetition is already exploited effectively by the existing byte stream plus LZMA, and these two transforms disrupt useful patterns enough to lose overall. This last mechanism is a plausible explanation of the measurements, not a proof of optimal entropy coding. The bounded LZMA sweep gives rate-for-speed/memory tradeoffs, not the hoped-for fixed-quality size gain. Header tightening was already a small historical lead and was not repeated here.

No encoder rate reduction from B is compared with raw entropy savings: every rate comparison uses complete final bytes divided by original pixels. No C technique is combined with a D optimization or enabled in production. Reproduce `python research_pass2/entropy.py accounting`, then `screen`. Decision and all reasons are frozen in entropy_promotion.json. The optional `main`/`holdout` commands require a nonempty deliberate promotion; neither was run because no size candidate passed. Memory probes use `memory --filter NAME` in separate processes.
