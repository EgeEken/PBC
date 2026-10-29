# PBC 4.0 architectural directions

No new architecture was implemented: this conventional-codec screen does not expose a compelling, demonstrated PBC representation advantage to build on. The result is a concrete competitive baseline and bounded next design, not another transform-plus-patch rescue.

## Completed-file comparison

Five originals: Googlewiki text, Mario synthetic graphics, Monet artwork, detailed Cat and smooth Flower. Ordinary released Q/HQ keeps default presets/250k planning, internal MSE and LZMA. JPEG: Pillow/libjpeg-turbo, 4:2:0, optimize; WebP: method 4; AVIF: installed Pillow/libavif, speed 8, max_threads 1, 4:2:0; JPEG XL: installed FFmpeg/libjxl effort 3, distance 8/15, thread 1. No conventional codec/video toolchain was installed.

JPEG/WebP/AVIF use bounded measured-rate searches, endpoints plus at most four bisections per PBC target, shared across Q/HQ (at most ten points per image/codec). JPEG XL has just the named two distances; do not infer equal-rate results where they fail to bracket. With ten PBC points and explicit working controls, 238 complete files were produced. This is a comparative screen, not PBC default tuning.

Full-resolution conventional files preserve source dimensions. Separate working_JPEG/working_WEBP controls encode the same PBC working target and store original dimensions in counted EXIF ImageDescription text; file-only reconstruction performs defined RGB bicubic upscale. These are explicit resizing wrappers, not full-resolution ordinary codec points or a new PBC 4 bitstream. Metadata counts toward bytes. results.json retains actual sizes, independently decoded original RGB MSE/pixel hashes and encode/decode seconds/file hashes. comparisons.json separates actual points within 5% rate, nearest points outside that range, strict dominance and adjacent measured interpolation. Never extrapolate; an estimate is not an achievable file.

These are actual AVIF files with both smaller size and lower MSE than the named PBC file, without interpolation or rate tolerance:

| Image | Preset | PBC bytes | PBC bpp | PBC MSE | AVIF q | AVIF bytes | AVIF bpp | AVIF MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| googlewiki.png | quality | 82440 | 0.9978 | 781.03 | 43 | 80537 | 0.9748 | 233.15 |
| googlewiki.png | high_quality | 176905 | 2.1412 | 714.43 | 84 | 174211 | 2.1086 | 195.64 |
| mario64.png | quality | 64809 | 0.5209 | 135.47 | 68 | 62909 | 0.5056 | 50.92 |
| mario64.png | high_quality | 89460 | 0.7190 | 122.03 | 82 | 88121 | 0.7083 | 47.61 |
| monet.jpg | quality | 53871 | 0.5100 | 96.80 | 43 | 45175 | 0.4276 | 51.56 |
| monet.jpg | high_quality | 130159 | 1.2321 | 53.98 | 62 | 128072 | 1.2124 | 17.52 |
| cat.jpg | quality | 63297 | 0.0594 | 299.19 | 18 | 53869 | 0.0506 | 115.91 |
| cat.jpg | high_quality | 78898 | 0.0740 | 293.13 | 21 | 75274 | 0.0706 | 103.89 |
| flower.jpg | quality | 28979 | 0.0272 | 43.43 | 18 | 23171 | 0.0217 | 21.06 |
| flower.jpg | high_quality | 46094 | 0.0433 | 34.84 | 38 | 44582 | 0.0418 | 9.57 |

AVIF strictly dominates all ten PBC points. Full WebP strictly dominates nine (Flower Q is below WebP's measured minimum rate). JPEG strictly dominates the six text/synthetic/art points. PBC has real low-rate dominance over JPEG q1 on Cat Q and Flower Q/HQ, but it does not survive against AVIF here. Working-resolution conventional wrappers also dominate most photographic points. Cat HQ beats nearby q1 JPEG MSE at 3.55% more bytes, a near-rate comparison rather than dominance. No universal or perceptual PBC advantage is established.

Full-resolution JPEG/WebP/AVIF brackets favor conventional methods wherever valid. Some nearest AVIF points (Monet Q, Cat Q) remain over 5% lower-rate; direct dominance remains valid, but they are not called near-rate. JPEG XL is better on selected text/Mario/Monet points at fewer bytes, but photographic distances do not reach PBC rates. No photographic JXL equal-rate claim.

Times are one-pass complete-codec observations with common source loading excluded and no memory polling. FFmpeg JXL includes process startup; Pillow/PBC are in-process, so totals cannot rank their arithmetic speed. No repeated cross-codec speed or isolated memory claim. RGB MSE does not establish a perceptual ranking. Source/version/thread controls are in ../provenance.json.

## Structural evidence

| Image | Working size | Direct working-target RGB MSE | Raw metadata share Q/HQ range |
| --- | --- | --- | --- |
| googlewiki.png | [501, 499] | 684.31 | 0.073-0.403% |
| mario64.png | [577, 433] | 89.18 | 0.125-0.458% |
| monet.jpg | [490, 511] | 40.30 | 0.130-0.665% |
| cat.jpg | [612, 408] | 280.57 | 0.172-0.242% |
| flower.jpg | [612, 408] | 22.12 | 0.294-1.105% |

Direct reconstruction uses the actual unquantized working target and production color/resize. Cat Q/HQ MSE 299/293 is near this target's 281, while AVIF reaches about 116/104 at fewer bits. Google HQ is similarly near the direct-target 684 at MSE 714, versus full AVIF 196. Fixed working resolution materially limits detail on these cases. This is not a mathematical lower bound for every possible working canvas: decoder-aware optimization can change it, and Pass 23 already tested one unsuccessful route. Mario/Monet also leave substantial patch approximation error.

Raw image/patch metadata occupies about 0.073-1.105% of unwrapped bodies; grid indices dominate. This does not attribute post-LZMA bytes, whose contexts are coupled, but argues against header micro-optimization as the main RD remedy. Initializer efficiency differs from completed refinement: Pass 25 found cheap photo initialization yet inefficient refinement. No policy ablation establishes that learned selection is essential.

Prior evidence: Pass 17 original-resolution DCT detail has selected strong RD gains; Pass 21 makes sparse storage fast; Pass 22 still has substantial photographic inverse reconstruction/memory cost. Pass 18 low-bit patches, Pass 19 ternary masks, Pass 20 tested multiscale, Pass 23 target preconditioning and Pass 24 pruning did not deliver general completed-file gains. Pass 25 transform bases followed by existing patches failed. These reject specific formulations, not all mixed representations; intermediate residual improvement is insufficient evidence.

## Distinct future designs

1. Full-resolution heterogeneous region tree with shared entropy. Use mutually exclusive source-space modes: compact smooth surface, local detail transform, or sparse/lossless graphics predictor. Select modes by completed-file marginal RD and jointly code tree/mode/symbol contexts. Unlike Pass 17 global enhancement or Pass 25 mandatory refinement, an expensive representation can be omitted rather than always paying both. Potential benefit is sharp detail without a global low-resolution ceiling and cheap truly smooth regions. Costs include partition/mode search, context simulation, boundary handling and a new format. Decoder traverses local regions/buffers without AI. Tree, mode, quantizer and context state count toward rate. Earlier split-border failures require continuous/lapped boundaries, not independent hard tiles. AVIF/JXL superiority is unproved.

2. Progressive full-resolution predictive wavelet/bitplane codec. Replace greedy overlapping patches with a scale hierarchy and embedded significance/refinement decisions. A complete stream stops at a rate budget; shared contexts avoid independently compressed residual arrays. Unlike Pass 20, rate allocation/entropy are intrinsic to one representation rather than layers appended to a PBC base. Potential benefit is controlled rate and bounded inverse work. Costs are a new predictor/context model/decoder; existing wavelet competitors already exploit this structure. Encode/decode should be roughly linear per coded bitplane, with less proposal orchestration but possible entropy overhead. RD advantage remains unproved.

Recommended first substantial experiment: design 1, retaining this conventional baseline. Freeze a minimal format with tree/mode/quantizer plus one shared entropy stream. Start with continuous smooth-surface and full-resolution detail modes, without the existing filler by default. Address the 250k ceiling and inefficient mandatory refinement. Encode performs multiple local trials; decode should need one traversal/local inverse, without full-image enhancement float copies. Count every header/mode/entropy byte. Compare actual nearby rates against full AVIF/JXL and resized conventional controls. Continue only for at least 10% matched-rate MSE reduction on two classes including one photographic and one graphics/art case, without a large decode/memory penalty. Otherwise stop before policy training, native optimization or more modes.

No prototype is forced now: neither design has demonstrated that advantage. Packaging and the optional metric are the practical engineering work; a broader rewrite must pass an RD gate.

```
python research_pass26/architecture/compare.py
python research_pass26/architecture/analyze.py
python research_pass26/architecture/diagnose.py
```
