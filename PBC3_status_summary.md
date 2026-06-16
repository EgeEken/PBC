# PBC3 — Status Summary

PBC v3.0 (Probabilistic Brush Compression) — lossy image codec. It builds an image from a
downsample "init" layer per channel plus N additive "patches" which are each selected and optimized to maximize gain in quality and minimize bit cost, each a resampled signed
residual grid quantized to a small palette. Working color space defaults to YCbCr; encoder
auto-downsamples large images to a pixel budget. Pre-release; backward compatibility is NOT a concern until we have a fully released PBC3.0 out there (not until numba port is complete at least).

## Pipeline (current)
1. Convert to working color space, auto-downsample to `auto_downsample_max_pixels`.
2. Per-channel init layer (heuristic auto-init or fixed cell/bitcount).
3. Patch loop: anchor selection → random box sampling → 3-stage funnel
   (pre-score → mid bound prune → exact RD score) → apply best patch.
4. Global entropy pass: LZMA (FORMAT_RAW) over the whole bitstream, with store fallback.

## Parameters and their observed effects (this version)
- `search_q_start/end` (0.4→0.1): stage-1 pre-score quality-vs-area weight. **Main speed knob.**
  Low = smaller/cheaper boxes = fast. High = favors large patches = much slower scoring for little gain.
- `q_start/end` (0.9→0.5): mid+exact scorer quality-vs-bits weight. 0.9 is good for the scorer.
- `q_init` (0.9): init-layer RD weight. Lower (≈0.6) weighs bits more and avoids quality creep.
- Separating search_q (pre-score) from q (scorer) was the key fix: lets the scorer stay aggressive
  without the pre-score flooding the funnel with expensive big patches.
- `downsample_rate` and `auto_downsample_max_pixels=250_000`: auto-downsample to a pixel budget. 250k works pretty well overall, but means inherently losing some irrecoverable detail on large images. But encoding speed scales really rough with pixel count so this is a compromise we have to make until a massive speedup is implemented (e.g. numba port, parallelization, or a better patch proposal and scoring model).
- `auto_downsample_init=True`: per-channel heuristic init (spatial frequency → smaller cell,
  variance → more bits), evaluates `init_search_depth` candidates, RD-picks by q_init. Good default;
  brute-force grid search was slow and gave no gains.
- `init_search_depth=20`: NOTE the candidate generator currently yields only ~7 unique combos, so
  depth>7 is effectively capped. Expanding the list (to ~15) shifted results to higher quality/size.
- `patch_count=20`: Total number of patches to add, relationship with quality is not exactly linear, since patches are optimized for quality/size in order and proportionally to the total patch count, and later patches are very small and cheap so difference between 50 and 100 patches is often small and made mainly by the first 10-20 patches in them. But still usually 50 patches means more room for quality than 20 patches and more encoding/decoding time.
- `search_depth=200`, `proposal_depth=50`, `exact_depth=10`: funnel sizes.
- `channel_cycle="Sum"` (default): as fast or slightly faster than "Max", it targets the channel that needs the most attention, in YCbCr this effectively means *all* the cycles go to the Y channel since there's almost always more error in it both in terms of max and sum; "Off" = round-robin (% channel count, each channel has their turn once per cycle).
- `palette_mode="generated"`: explicit/auto (binned + Lloyd k-means) give no gain — residual grids are
  unimodal/symmetric so the generated distributor is already near-optimal, and explicit pays header bits.
- `patch_bitcount_mode="constant"`: dynamic bitcount does nothing or slightly worse; leave off.
- Entropy: global LZMA saves ~15–22% on the bitstream. zlib always loses to LZMA (drop it).
  Per-patch entropy would be worse (per-stream overhead, no shared dictionary, tiny inputs).

## Grid coding decision: RAW to LZMA only
- Per-patch RLE/zero-run was REMOVED. Under global LZMA, forcing RAW grids was consistently smaller
  (tested −1.3% to −6.5% on a 3-image sample, never worse).
- Probably reason: RAW grids (long repeated zero/value patterns) are exactly what LZMA matches best; RLE/zero-run
  pre-tokenizes those runs into irregular variable-width tokens that scramble the regularity LZMA needs.
  The per-patch coding duplicated LZMA's job and obscured structure.
- Action taken: `_choose_grid_encoding` always returns `(MODE_RAW, 0)` (one-line change). The
  `_runs`, `_rle_chunk_count`, `_zero_run_token_count` helpers and the RLE/zero-run branches are now
  dead code — fully delete them (plus the 2-bit grid-mode field in the stream) during the numba cleanup.

## Known presets
- Default (quality-leaning; ~9s on a 500x500 image): patch_count=50, search_q 0.4→0.1,
  auto_downsample_init=True, init_search_depth=10.
- Fast preset (great MSE/bpp, very fast, quality is "locked" since patch_count is fixed):
  patch_count=20, search_depth=100, proposal_depth=10, exact_depth=5, search_q 0.35→0.1,
  q_init=0.6, q_start=0.9, q_end=0.8, init_search_depth=5, auto_downsample_init=True.

## Known issues / accepted tradeoffs
- Decode is now uniformly on the slow RAW per-cell Python read loop (since RAW-only). This is expected
  and temporary — RAW unpack is the FIRST numba target, which fixes it.
- Encode time is content-dependent (search_q reacts to image content). Accepted for now; could later
  cap total resampled pixels per step for a hard bound.

## To-experiment (existing features, not fully explored)
- Confirm RAW-only is smaller across the full test set (currently n=3).
- `channel_cycle` "Off" vs "Sum" vs "Max" full A/B.
- LZMA `preset=9 | PRESET_EXTREME` for a marginal ratio gain at higher encode cost.
- Lower `q_init` (≈0.6) paired with a richer init candidate list.
- `positive_bias` toggle, and tuning `top_k`, `anchor_block_size`, `min/max_patch_size`.

## Roadmap / next steps
0. Get rid of redundant, dead code (RLE/zero-run etc) and clean up codebase without breaking existing functionality.
1. Final call for last minute added feature (then freeze features for first release):
   - RGBA Support, would just be one extra channel (A) with its own init layer and patches. Could also support RGB+Alpha as a 4-channel
     YCbCr+Alpha, by treating the alpha channel as a separate, non-transformed channel (not a priority, but an option for future versions)
   - **Symbol-level range/arithmetic coder over grid indices** — the real entropy win. Fixed 2-bit
     indices waste fractional bits because index 0 dominates; an adaptive coder captures ~1–1.3 bits/cell.
     Global LZMA can't get sub-bit on the skewed index distribution.
   - Quality-target stopping (stop at target MSE/abs delta whichever is fastest to compute) — deferred for now but on agenda;
     makes quality a dial instead of a fixed patch_count.
   - Optional near-lossless residual patch mode to break the MSE floor.
   - Optional: byte-aligned structure-of-arrays stream layout (group headers vs indices) to help LZMA;
     pairs well with the range coder.
   - Optional: anything else? Let me know if you have any new ideas not mentioned here or changes you think would improve the algorithm beyond what is already planned.
2. **Numba port** (next): start with grid codec (RAW bit-pack/unpack) + `quantize_signed` (kills the
   RAW-decode slowness), then `_mask_from_values` and integral/box-sum helpers. While there, delete the
   dead RLE/zero-run helpers and the 2-bit grid-mode field. Maybe leave `signed_resample`? (PIL bicubic, C)
3. Hyperparameter tuning: map the quality/compression/speed frontier, lock recommended presets.
4. Port / parallelize: architecture is parallel-friendly (independent candidate scoring per step;
   fully parallel decode). Consider parallelizing exact-stage scoring even before a language port. Maybe a Rust port for clean parallelism and speed.
5. Reinforcement learning model for patch proposal (only after code is fast enough to make training cheap). Train model by iteratively scoring an image set, in smaller batches probably. Image -> patch selector model -> 20 patches -> final MSE/comp rate turned into reward for RL, repeat. So it learns to propose the best patches without having to actually try and score hundreds of candidates, although it could still probably benefit from a small number of scored candidates to refine its proposals. 
6. PBC Video: encode frame differences as few patches + periodic keyframes (better settings/more patches).

## Files
- `PBC3.py` — codec (BitWriter/BitReader, PBC3Config, PBC3Result, PBC3).
- `PBC3_animation.py` — `animate_pbc3(...)` renders the patch-by-patch build to MP4/GIF
  (channel-separated panels, error maps, per-patch box + grid-mode overlay).
