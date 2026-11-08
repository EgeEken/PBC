# Competitive reference — verified Coast points

Source: `hpt_data/coast.jpg`, 4240×2832, 12,007,680 pixels, RGB pixel hash `29af11f…`. Bpp is completed-file bytes × 8 / source pixels. PBC quality values are full-resolution decoded RGB MSE. PBC bytes and RD come from Pass 44’s frozen source; Coast Compression was independently re-encoded from the public release Python source and matched the frozen stream exactly.

| Image | Codec/config | Bytes | bpp | RGB MSE | Encode (s) | Decode (s) |
|---|---|---:|---:|---:|---:|---:|
| Coast | Published ZIP Python PBC 3.1 Compression | 6,461 | 0.004305 | 73.1336 | 1.109 median‡ | 0.198 median‡ |
| Coast | Published ZIP Python PBC 3.1 Balanced | 11,729 | 0.007814 | 48.2200 | 1.048 median‡ | 0.203 median‡ |
| Coast | Published ZIP Python PBC 3.1 Quality | 18,717 | 0.012470 | 40.8844 | 0.794 median‡ | 0.194 median‡ |
| Coast | Published ZIP Python PBC 3.1 High Quality | 39,339 | 0.026209 | 23.5568 | 1.004 median‡ | 0.199 median‡ |
| Coast | HF-compatible Pillow AVIF quality 0 | 7,647 | 0.005094 | 72.3782 | Not measured | Not measured |
| Coast | Pinned research AVIF qcolor 0 | 6,747 | 0.004495 | 88.5370 | 0.363‡ | 0.506‡ |

‡ Published ZIP embedded Python source, one thread, one warmup and three in-process repetitions; complete codec call excludes process launch and file I/O. Pinned AVIF timing includes CLI launch/I/O and is not directly comparable. Pillow point is 18.3% larger than PBC Compression, so it is not an equal-rate claim. The pinned AVIF q0 point is 4.4% above PBC Compression. The user’s HF PBC screenshot point (4.65 KiB/88.87) remains a user observation without a stream or exact upload hash and is intentionally not presented as a verified codec row.
