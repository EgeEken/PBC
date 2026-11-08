# Decoder profile

## Setup

Used unmodified Pass 42 v1 streams: Coast intermediate (4240×2832, 688,860 B) and Googlewiki intermediate (814×812, 106,968 B). Timers wrap header/XZ processing, map unpack, upsample, procedural surfaces, coefficient parsing, inverse-DCT matrix multiplies, and taper. The residual “decode other” is mode parsing and region composition. Instrumented totals are diagnostic; lightweight repeated timings in `outputs/decoder_benchmark.json` are the performance comparison. OpenBLAS threads were limited to one.

## Coast, 12.0 MP

| Stage | Time | Share | Relevant work |
|---|---:|---:|---|
| XZ decompress | 0.039 s | 0.26% | one payload |
| Mode/index bit unpack | 0.062 s | 0.42% | two unpack calls |
| Initialization upsample | 1.647 s | 11.15% | one 12 MP bilinear resize |
| Procedural surface generation | 10.832 s | 73.33% | 32,182 surfaces from 32,205 procedural records |
| Coefficient parse | 0.926 s | 6.27% | 20,040 coefficient reads |
| Inverse DCT multiply | 0.069 s | 0.47% | 40,080 matrix multiplies; 20,040 block/channel transforms |
| Other mode/region work | 1.194 s | 8.08% | 1,670 transform regions plus loop/composition |
| **Instrumented total** | **14.771 s** | **100%** | taper disabled |

The dominant cost is the Python-level procedural surface builder. It constructs a 16×16 correction array for each active region, performing repeated per-region interpolation and temporary-array work. It scales primarily with active procedural region count, not only image pixels. XZ and inverse DCT are minor. Upsampling is a secondary pixel-count cost. Taper was not called.

## Small image and memory

Googlewiki has 80 surface builds, 1,440 coefficient reads, and 2,880 matrix multiplies. Total instrumented decode is 0.257 s. Map unpack takes 0.103 s (40%); surface generation 0.023 s; coefficient parsing 0.057 s; inverse multiplication 0.004 s; XZ 0.006 s. This differs from Coast because fixed Python unpack/mode overhead and image-area initialization matter at small sizes.

Coast’s in-process sampled RSS rose from 39.3 MB to 116.2 MB (observed delta 76.8 MB). This is a decode measurement, not a high-water mark. Pass 42’s approximately 746 MiB working-set figure came from `profile_large.py`, which executes encoding only; it cannot be attributed to decoding.
