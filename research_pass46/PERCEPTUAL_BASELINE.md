# Perceptual and rate baseline

## Reference and measurement

The PBC rows use the canonical published PBC 3.1 source, model, and preset definitions pinned by Pass 45. Coast reproduces the archived Pass 44 stream and decoded-pixel hashes exactly. All three source files and their EXIF-transposed RGB forms are SHA-256 recorded in `outputs/baseline/baseline_metrics.json`. RGB MSE is computed at source dimensions from the decoded RGB pixels.

The two AVIF quality-0 results are distinct profiles: Pillow 11.3.0's AVIF save path, and pinned libavif 1.3.0/libaom 3.12.1 `qcolor=0`, speed 8, one thread, 8-bit 4:2:0, full range, CICP 1/13/1. AVIF qcolor is a quality control, not a target bitrate. Each file's completed byte count is reported. Additional qcolor points are included only as near-rate brackets; they are not called quality-0.

SSIMULACRA2 uses the same installed binary for all PNG pairs: `ssimulacra2.exe` SHA-256 `08a44e2d4cd44098695dbe47cdec1f0ba7c68bf9f0389b5785133d81e9b25fed`, paired with cjxl v0.11.2. Higher is better and 100 is identical. Negative values are valid. AVIF timing includes command-line process launch and image file I/O, so it is not directly comparable to in-process PBC timings.

## Quality-0 profile results

| Image | Codec/profile | Bytes | bpp | RGB MSE | SSIMULACRA2 |
|---|---|---:|---:|---:|---:|
| Coast | PBC 3.1 Compression | 6,461 | 0.004305 | 73.13 | -45.35 |
| Coast | Pillow AVIF quality 0 | 7,647 | 0.005095 | 72.38 | -18.90 |
| Coast | pinned libaom qcolor 0 | 6,747 | 0.004495 | 88.54 | -26.68 |
| Googlewiki | PBC 3.1 Compression | 28,974 | 0.350686 | 1,005.02 | -25.69 |
| Googlewiki | Pillow AVIF quality 0 | 8,299 | 0.100447 | 1,061.66 | -42.50 |
| Googlewiki | pinned libaom qcolor 0 | 5,218 | 0.063156 | 1,390.61 | -55.96 |
| Mario64 | PBC 3.1 Compression | 14,099 | 0.113321 | 414.24 | -50.69 |
| Mario64 | Pillow AVIF quality 0 | 4,425 | 0.035566 | 418.19 | -55.07 |
| Mario64 | pinned libaom qcolor 0 | 4,139 | 0.033267 | 580.14 | -57.30 |

These points do not establish general codec superiority. They do show that AVIF quality 0 behaves very differently by image and implementation. For Coast, Pillow quality 0 has nearly the same MSE as PBC Compression at 18.4% more bytes and a much higher SSIMULACRA2 score. The pinned qcolor-0 file is 4.4% larger but has higher MSE. For Googlewiki, both AVIF q0 files are much smaller but have higher MSE and lower SSIMULACRA2. For Mario64, both q0 files are much smaller, with slightly higher MSE and lower SSIMULACRA2.

## Near-rate pinned AVIF brackets

| Image | PBC point | AVIF point | Rate gap | PBC MSE | AVIF MSE | AVIF SSIMULACRA2 |
|---|---|---|---:|---:|---:|---:|
| Coast | Balanced, 11,729 B | qcolor 7, 11,260 B | -4.0% | 48.22 | 31.28 | 1.79 |
| Googlewiki | Compression, 28,974 B | qcolor 12, 29,727 B | +2.6% | 1,005.02 | 498.36 | 2.41 |
| Googlewiki | Balanced, 53,890 B | qcolor 28, 55,025 B | +2.1% | 845.44 | 291.46 | 40.55 |
| Mario64 | Compression, 14,099 B | qcolor 20, 14,504 B | +2.9% | 414.24 | 154.81 | 9.43 |
| Mario64 | Balanced, 38,900 B | qcolor 50, 39,785 B | +2.3% | 185.71 | 63.73 | 55.04 |

AVIF outputs were checked against the archived Pass 44 byte counts and hashes. At these five near-rate points, the pinned AVIF profile has lower RGB MSE and higher SSIMULACRA2 than PBC. This is evidence on three development images, not a broad validation claim and not an AVIF q0 comparison.

## Timing context

PBC complete in-process median encode/decode seconds (three samples each): Coast Compression 0.506/0.113 and Balanced 0.553/0.131; Googlewiki Compression 0.203/0.018 and Balanced 0.237/0.031; Mario64 Compression 0.199/0.024 and Balanced 0.343/0.035. Pinned AVIF CLI median encode/decode times and every sample are in the JSON data; the timings include process startup and file I/O and are not used to claim a speed winner.

## Visual files

The full-size error/edge diagnostics and identically scaled local candidate comparisons are under ignored `outputs/visuals/`. These are feasibility-only reconstructions from the canonical PBC base plus local stroke candidates, not a V4 decoder or complete V4 stream. The strongest visual limitation is that softened structures remain soft: the Coast rod is still not sharply defined; the Googlewiki text/logo does not become convincingly crisp; and the Mario64 fence/edges show only subtle local improvement. No clear halo or tear was seen in the examined crops.
