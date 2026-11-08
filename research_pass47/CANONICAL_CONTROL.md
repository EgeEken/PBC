# Pass 47 canonical control

## Source and model

The research-only copy in `codec/` was copied from the published release Python at `research_pass45/outputs/published_release_python/`. Its root Python file hashes are in `CODEC_COPY_MANIFEST.json`; each was copied without source edits. The preset factory's nested model and root model are both SHA-256 `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`. The unmodified copied source and model are used by `scripts/run_oracle.py` and by fresh-process decoding. No PBC production source, released preset, model, decoder or stream format was changed.

Inputs were EXIF-transposed and converted to RGB before encoding, matching the frozen benchmark. An earlier informal probe used the raw `googlewiki.png` PIL image, which is RGBA; it produced a different stream and was discarded. The canonical RGB-normalized harness reproduced the archived output exactly.

Runtime was Python 3.13.5, NumPy 2.2.6, Numba 0.62.1, Pillow 11.3.0, with numerical thread counts set to one. This differs from the Python 3.11.16/NumPy 2.4.6/Numba 0.68.0 runtime recorded in Pass 45/46. Despite the environment difference, all six outputs matched the saved Pass 46 canonical stream and decoded-pixel hashes exactly.

## Instrumentation identity

For each image/preset, three uninstrumented encodes after a warmup were deterministic. A runtime-only wrapper then captured the same fixed-step state while calling the original selection function exactly once and captured serialized patches at the writer boundary. An additional instrumented repeat was deterministic. All of these checks passed on all six cases:

- Instrumented stream bytes and SHA-256 equal the uninstrumented output.
- Instrumented reconstructed RGB SHA-256 equals the uninstrumented reconstruction.
- The captured patch sequence equals the patch sequence parsed from the uninstrumented PBC stream.
- Fresh-process decoding of every saved control reproduces the encoder reconstruction hash.
- Replay of each sampled state's proposal list produces identical boxes and identical post-proposal RNG state; the original selection itself leaves RNG consumption unchanged.

| Image / preset | Bytes | Patches | RGB MSE | Encode median (s) | Decode median (s) | Stream / pixels match Pass 46 |
|---|---:|---:|---:|---:|---:|---|
| Coast / Compression | 6,461 | 50 | 73.134 | 0.437 | 0.095 | Yes |
| Coast / Balanced | 11,729 | 42 | 48.220 | 0.435 | 0.094 | Yes |
| Googlewiki / Compression | 28,974 | 12 | 1,005.022 | 0.136 | 0.019 | Yes |
| Googlewiki / Balanced | 53,890 | 12 | 845.441 | 0.156 | 0.021 | Yes |
| Mario64 / Compression | 14,099 | 19 | 414.236 | 0.138 | 0.022 | Yes |
| Mario64 / Balanced | 38,900 | 53 | 185.712 | 0.256 | 0.026 | Yes |

Encode/decode values are medians of three complete in-process calls after warmup. A separate fresh process was used for pixel-hash verification. MSE is independently computed on the original-resolution RGB result with int32 subtraction and float64 averaging. SSIMULACRA2 values match Pass 46 because the decoded pixels are identical.

## Selection stages

The frozen model is the legacy 30-action model. At each accepted patch step, the search draws 200 spatial samples, keeps and orders up to 50 boxes by prescore, and the learned filler receives one box because `learned_filler_candidates=1`. The model predicts one action from 30 logits; `learned_filler_top_k=1` evaluates that single native reconstruction. Its value head has no ranking effect with one box. The local choice score is the existing `reduction - lambda * bits`, where the legacy model's q-dependent lambda and exact patch bit cost are used.

The state log records the pre-proposal RNG hash, post-choice RNG hash, target/canvas hash and ordered proposal-pool hash. The full states and RNG values are in ignored `outputs/states/`; the compact `PASS47_SUMMARY.json` carries their hashes and replay outcomes.
