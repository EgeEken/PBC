# PBC Pass 12 — Summary

RGB v0 encoding and decoding work, but RGB did not improve rate-distortion efficiency in the fixed Quality/High Quality screen. Keep YCbCr as the default and stop this direction.

1. **Strictly better cases:** 0/16. RGB improved MSE with a bitrate increase in 14 cases and lost on both MSE and bpp in two.
2. **Cases with lower MSE:** Quality: Cat, Flower, Monet, Googlewiki, Eiffel, Stockpoint, Rate-distortion. High Quality: Cat, Flower, Googlewiki, Mario64, Eiffel, Stockpoint, Rate-distortion. Monet High Quality and Mario64 Quality worsened MSE.
3. **Matched rate:** No gain demonstrated. Every RGB point used more bits, so the screen did not pass the matched-rate gate.
4. **Timing:** Local mean Quality encode/decode time rose 40.84%/23.62%; High Quality was +2.05%/−0.45%. Two repeats had up to 31.9% per-case encode CV; these are not stable speed conclusions.
5. **Cause:** YCbCr roundtrip floors are small (≤2.43 MSE). The larger changes follow different learned-policy actions, channel allocation, grids and patch counts. RGB increases raw and final byte totals on average; one Eiffel High Quality row had smaller raw bytes but a much larger final LZMA file.
6. **Global RGB:** Not justified.
7. **Cheap adaptive selection:** Not tested. RGB does not save bits on any screen point; selecting it would trade much higher bitrate for MSE on most cases. No evidence supports a deployable content rule.
8. **Confirmation:** No full main-set or holdout run; the screen failed the RD gate. No HF run or deployment.
9. **Decision:** Reject RGB as a default change; retain the local result as a limited research screen. The existing YCbCr baseline remains unchanged.
10. **Next RD direction:** Explore shared-support cross-channel patch coding so correlated channel corrections can share spatial geometry/payload structure. This addresses RGB’s observed channel/grid cost directly and is distinct from changing the entire color space or rescoring each patch in final RGB.

### Provenance

- Research branch at run: [GitHub b339f23](https://github.com/EgeEken/PBC/commit/b339f23576cda5c372c3fe9f3e37866e1d274b11)
- Frozen codec source: [GitHub 8006f30](https://github.com/EgeEken/PBC/commit/8006f30debfc4a57289df06df21636aeb52baffd)
- HF baseline: [98ade0c](https://huggingface.co/spaces/EgeEken/PBC/commit/98ade0c8072a4571e9674eaac3d91f861e3b9867)
- Model: `patch_policy.npz`, SHA-256 `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`
- Machine-readable screen: `research_pass12/paired_results.csv`, `preprocess_floor.csv`, `screen_summary.json`.
