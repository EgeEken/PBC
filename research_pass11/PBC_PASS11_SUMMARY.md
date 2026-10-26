# PBC Pass 11 — Summary

- **Speed discrepancy:** The 20-image Pass 10 run averages 0.728451 s encode / 0.107516 s decode. On the exact eight Pass 9 images, Pass 10 averages 0.552984 / 0.066456 s. Sample composition explains about 72% of the encode gap and 86% of the decode gap. Timer scope and runtime controls explain why the residual is not a controlled comparison: Pass 10 includes `Image.fromarray` in the timer and is unpinned; Pass 9 prepares PIL input before timing, pins workers and numerical threads, interleaves cases, and summarizes per-case medians. All 32 overlapping streams, patch counts, effective preset behavior, and dimensions match; Pass 10 did not record input-file hashes. No production regression or benchmark bug was established; no fix or new HF run was justified.
- **Final-RGB audit:** 36 sampled selected/next-best-action pairs across six images and Quality/High Quality found one working-vs-RGB ordering disagreement. The current action in that case was both cheaper and better after final RGB reconstruction. Equal-cost alternatives tied. Final-RGB scoring cost about 0.434 s per sampled state locally. No cheap ranking signal or completed-file RD gain was demonstrated. The idea is rejected for the current encoder; no ranking adjustment or RD candidate was implemented.
- **Changes:** Research-only scripts and reports were added. The codec, model, stream, decoder, public benchmark defaults, and HF Space were not changed. The raw probe output remains local and uncommitted.
- **Release status:** The optimized baseline has measured encode/decode speedups while preserving PBC 3.0 bytes. Lower MSE and lower bpp at comparable quality remain unmet; there is no PBC 3.1 release candidate.
- **Next research direction:** Reuse or incrementally update search summaries across patch steps to reduce repeated integral-image and proposal work without changing candidate semantics. Measure exact output identity and single-thread speed against the frozen baseline before considering an RD experiment.

## Revisions

- Research branch start: [GitHub 678c029](https://github.com/EgeEken/PBC/commit/678c0293712cb63cb6acffa9de92d9cac36fb752)
- Frozen source: [GitHub 8006f30](https://github.com/EgeEken/PBC/commit/8006f30debfc4a57289df06df21636aeb52baffd)
- Pass 9 controlled HF revision: [4237d954](https://huggingface.co/spaces/EgeEken/PBC/commit/4237d954f4e1a04951c822c738113a53bbd58e9b)
- Pass 10 public benchmark HF revision: [98ade0c](https://huggingface.co/spaces/EgeEken/PBC/commit/98ade0c8072a4571e9674eaac3d91f861e3b9867)

Reproduction: `python research_pass11/analyze.py` reads the saved Pass 9/10 JSON files and local `rgb_probe.jsonl`; `python research_pass11/rgb_probe.py` reruns the limited local candidate audit. See the two detailed reports for source hashes and measurement definitions.
