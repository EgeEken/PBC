# Existing public speed run log

At the user's request, existing run `20261026_141310` was stopped and its completed rows exported before any restart. It stopped without error at 996/9,800 calls: one complete 980-call trial (20 images, 49 codec/variant points), followed by 16 PBC3 Compression calls in trial two. No Pass 9 HF job, deployment or restart occurred. HF remains `27f6c65225a5496fa065deb586c7e405ecdffc84` on cpu-basic.

| Codec | Screenshot encode s | Screenshot decode s | Matching image coverage | Complete first-trial encode s | Complete first-trial decode s |
| --- | ---: | ---: | --- | ---: | ---: |
| PBC3 | 0.534 | 0.082 | 20/20 | 0.534392 | 0.082203 |
| JPEG | 0.017 | 0.034 | 20/20 | 0.017368 | 0.034152 |
| JPEG2000 | 2.792 | 0.899 | 20/20 | 2.792432 | 0.898973 |
| WEBP | 0.456 | 0.055 | 20/20 | 0.456094 | 0.055082 |
| JXL | 1.827 | 0.107 | 20/20 | 1.826846 | 0.106959 |
| AVIF | 1.187 | 0.056 | 20/20 | 1.187166 | 0.055505 |
| PNG | 3.104 | 0.261 | 5/20 | 1.878271 | 0.164118 |

The screenshot's PNG values match precisely the first five PNG records (cat, cathedral, coast, ducklings, Eiffel), rounded to three decimals. Its other six bars match all 20 first-trial images at that precision. This supports an in-progress PNG snapshot; the screenshot does not show its run selector or filters. The complete first-trial PNG mean is 1.878271/0.164118 s. Do not interpret the partial PNG bar as the completed comparison.

The completed first-trial codec means above weight every codec/variant/image row equally. Each variant has 20 images, so this also equals the UI mean of variant means without filters. The later 16 PBC3-only calls are excluded to keep complete image coverage. Four PBC3 presets and the competitor quality grids represent different RD operating points; codec averages do not establish matched-rate superiority. There is only one complete trial and no per-encode source/module/RNG manifest from this public runner. These values are an observational public-run log, separate from the unperformed controlled three-way Pass 9 comparison.

The local ignored backup `benchmark_stopped.db` contains 6,396 total rows, including two historical runs and all 996 rows of this run; SQLite quick_check passed. Its SHA256 is `ceb40d7366f5c8291b1df2e09b848aed5702b6fb85eae09f0c06b3ffffd833a3`. A prior read-only snapshot preserves 945 active-run rows. Neither database nor the user screenshot is committed. `run_stop.json` records the authorized stop and unchanged HF revision; `public_run_log.json` records compact first-trial summaries and the 80 PBC3 per-image rows. Reproduce this analysis without running any codec using `python research_pass9/log_public_run.py` against the retained backup and ignored screenshot.
