# AVIF reference reconciliation

Two configurations were measured on the Pass 44 Coast input (`hpt_data/coast.jpg`, source RGB SHA `29af11f…`).

| Profile | Configuration | Bytes | Bpp | RGB MSE | Encode/decode time |
|---|---|---:|---:|---:|---|
| HF-compatible Pillow | Pillow 11.3.0, `Image.save(format="AVIF", quality=0)` with other options omitted | 7,647 | 0.005094 | 72.3781849 | Not measured in this pass |
| Pinned research | libavif 1.3.0 / libaom 3.12.1, qcolor 0, speed 8, one thread, 8-bit 4:2:0 full range, CICP 1/13/1 | 6,747 | 0.004495 | 88.537 (Pass 44) | Pass 44 timing includes CLI launch and I/O; not comparable |

The local Pillow output reproduces the user’s rounded observation (7.47 KiB, MSE 72.38) on the research Coast asset. It does not reproduce the pinned research profile. Pillow’s AVIF plugin defaults to quality 75 unless passed quality 0, subsampling 4:2:0, speed 6, automatic codec, full range, and a platform-dependent default thread count. Its bundled `_avif` backend does not expose the pinned profile’s explicit CICP matrix-coefficient setting through the ordinary Pillow save call; Pass 44 observed CICP 1/13/6 in this path versus 1/13/1 in the pinned profile. The public HF Docker runtime’s exact linked libavif/libaom build was not independently identified, so the local reproduction validates the Pillow-compatible path, not the server binary.

**Cause of the AVIF mismatch:** the two runs used different AVIF frontends/defaults and color metadata, not the same q0 encoder configuration. AVIF quality 0 does not imply a fixed bitrate. Retain both profiles in future comparisons.
