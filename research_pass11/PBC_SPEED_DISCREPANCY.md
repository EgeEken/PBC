# PBC Pass 11 — Speed discrepancy audit

## Finding

The ordinary benchmark’s 0.728451 s encode / 0.107516 s decode means are not a demonstrated codec regression. Pass 10 covered 20 images; Pass 9’s controlled timing covered eight. Restricting Pass 10 to those exact eight images gives 0.552984 / 0.066456 s. The other 12 images raise the 20-image averages by 0.175468 / 0.041059 s. This sample-composition effect accounts for about 72% of the reported encode gap and 86% of the decode gap.

The remaining same-eight difference is 0.069652 s encode and 0.006885 s decode relative to Pass 9’s 0.483332 / 0.059571 s. It is consistent with known runner differences, but the saved results cannot apportion it among them. Pass 10’s timing includes `Image.fromarray(arr)` inside the encode timer, while Pass 9 creates and EXIF-normalizes the PIL image before timing. Pass 9 pins each persistent worker to one CPU and sets numerical thread variables to one; the production benchmark does neither. Pass 9 reports an average of per-case medians from three interleaved repeats. Pass 10 reports the ordinary arithmetic mean of three trials over the full image set. Numba was warmed before both measured paths; model hash, named input images, presets, and output work match; Pass 10 did not store input-file hashes.

No new HF run was needed: sample restriction and source inspection account for most of the difference and show that the residual is a runner/environment comparison, not an isolated codec change. No production fix is supported by the evidence. Do not compare the two absolute means as a regression pair.

## Saved-data checks

- Pass 10 is public run `20261026_172335`: 240 PBC rows, 20 images, four presets, three trials. The run metadata names baseline `pbc31_release_rng_baseline` and records all source and model hashes.
- All 32 overlapping image/preset cases have matching patch counts and stream hashes between Pass 9 `dev_release_rng` and Pass 10; all Pass 10 repeats also reproduce those hashes. Pass 10’s full 240 streams match the frozen release reference.
- The eight-image matched Pass 10 mean is 0.552984 / 0.066456 s. Across the 20 per-image means (averaged over four presets), mean / median / p90 / maximum are 0.728451 / 0.687918 / 1.062466 / 1.135317 s encode and 0.107516 / 0.105090 / 0.189569 / 0.199204 s decode.
- By preset, Pass 10 all-image per-image encode mean / median / p90 / maximum (seconds): Compression 0.676739 / 0.691300 / 1.011225 / 1.030595; Balanced 0.732021 / 0.701222 / 1.202157 / 1.237409; Quality 0.727759 / 0.748248 / 0.989693 / 1.350862; High Quality 0.777287 / 0.679446 / 1.193845 / 1.255629. Slow cases cluster among the approximately 12 MP inputs; no single outlier creates the whole mean.
- Complete-trial encode means are 0.748662, 0.705206, and 0.731486 s; decode means are 0.105852, 0.107297, and 0.109398 s. There is no monotonic trial drift that explains the gap.
- For all 20 images, arithmetic row means versus the mean of the per-image medians are 0.728451 versus 0.734102 s encode, and 0.107516 versus 0.105168 s decode. This aggregation choice is small relative to the workload and runner differences.

The matched eight are `cat.jpg`, `eiffel.png`, `flower.jpg`, `googlewiki.png`, `mario64.png`, `monet.jpg`, `ratedistortion.png`, and `seagull.jpg`. Pass 9’s saved rows include original and working dimensions. Those dimensions and the final patch counts agree with Pass 10; for example Cat is 3576×2384 → 612×408, Eiffel is 487×487 native, Googlewiki is 814×812 → 501×499, and Seagull is 1939×2424 → 447×559.

## Execution path inspection

- `pbc3_benchmark.py::_eval_pbc3` starts its encode timer immediately before `PBC3.compress(Image.fromarray(arr), config=cfg)`. The timer includes PIL image construction, codec preparation, encode, serialization, and returned reconstruction/MSE work inside `compress`.
- Pass 9’s `worker.py` loads, EXIF-transposes, and converts each input to RGB before starting its timer. Its timer covers `PBC3.compress(image, ...)` and therefore codec preparation through `compress` return, but not PIL construction or input loading.
- Both paths use `time.perf_counter`, the release-compatible RNG, matching public presets, the same model hash, the same 250,000-pixel working cap, and the same codec source baseline. Both warm the Numba/model path before timing.
- Pass 9 pins persistent workers to one CPU and limits OMP/OpenBLAS/MKL/NumExpr/Numba/VECLIB threads to one. The public runner has no affinity/thread pinning and runs a serial 20-image production loop. Scheduling and shared CPU load remain uncontrolled.

## Reproducibility and revisions

Read-only analysis script: `research_pass11/analyze.py`. It consumes the saved ignored Pass 9 `comparison_summary.json`, Pass 10 `public_benchmark.json`, and, if present, the local Pass 11 `rgb_probe.jsonl`; it does not execute codec jobs.

- Research branch at start: [GitHub 678c029](https://github.com/EgeEken/PBC/commit/678c0293712cb63cb6acffa9de92d9cac36fb752)
- Frozen development source: [GitHub 8006f30](https://github.com/EgeEken/PBC/commit/8006f30debfc4a57289df06df21636aeb52baffd)
- Pass 9 controlled HF revision: [4237d954](https://huggingface.co/spaces/EgeEken/PBC/commit/4237d954f4e1a04951c822c738113a53bbd58e9b)
- Pass 10 public benchmark HF revision: [98ade0c](https://huggingface.co/spaces/EgeEken/PBC/commit/98ade0c8072a4571e9674eaac3d91f861e3b9867)
- Source SHA-256: `PBC3.py` 74fe069600f8190f1a5ac1b4cb01dff29f4b1262d6d36da459236e7a9fb0016c; `pbc3_benchmark.py` 264844e43a7329f89bdad94643b24beaf09a4b85b0e45c4126b817afff7160e4; Pass 9 `worker.py` 52fe88508029cf1d2977bf964b5b01999aa64e86ff4883a792c41ea9aca4715f; model `patch_policy.npz` ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f.

No production or public benchmark code was changed. No full Speed Benchmark or new HF benchmark was run.
