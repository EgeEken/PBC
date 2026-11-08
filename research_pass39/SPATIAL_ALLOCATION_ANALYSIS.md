# Spatial allocation analysis

Source: `hpt_data/cat.jpg`, 3576×2384, qcolor 35, speed 6. This point compares A and C within 0.33% in final rate. Region masks were computed once from the source's linear-light luma: top 20% Sobel magnitude is edge; among the remainder, top 20% local 8×8 variance is texture; the rest is smooth. Masks are mutually exclusive. Regional RGB MSE uses float64 squared error; SSE share is the region's share of total RGB squared error for that encode.

| Region | Area | A MSE | C MSE | C change vs A | A SSE share | C SSE share |
|---|---:|---:|---:|---:|---:|---:|
| Edge | 20.00% | 70.824 | 80.225 | +13.27% | 33.44% | 35.67% |
| Texture | 16.00% | 67.995 | 75.865 | +11.57% | 25.69% | 26.99% |
| Smooth | 64.00% | 27.049 | 26.241 | −2.99% | 40.87% | 37.34% |

B's reconstruction and regions equal A exactly for this point. C slightly reduces error over the large smooth area, but increases error in both edge and texture masks. These fixed masks do not imply a causal or universal content-class result.

## Effective qindex maps

The separate instrumented libaom build sampled qindex and segment ID during encoding. The AVIF produced by that build matched the reference binary byte-for-byte for all five configurations at Cat q35. To avoid uninitialized subblock fields, this summary retains one sample at each coded superblock root: 532 128×128 superblocks for REF/A/B, and 2,128 64×64 superblocks for C/D. The displayed maps expand each root value over its superblock; qindex is not a regional bitrate estimate.

| Config | Root qindex range | Mean | Root histogram |
|---|---:|---:|---|
| A | 164 | 164.00 | 164: 532 |
| B | 164 | 164.00 | 164: 532 |
| C | 108–164 | 134.37 | 108: 664; 116: 72; 124: 42; 132: 57; 140: 310; 148: 541; 156: 351; 164: 91 |

REF has the same flat qindex 164 and segment 0 as A/B. C and D have the same qindex map and file bytes at this point. The map, regional masks and underlying pixels are summarized in `outputs/benchmark/spatial_analysis.json` and shown in `visuals/spatial_allocation.png`.
