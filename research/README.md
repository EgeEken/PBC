# PBC research index

- **Current released version:** PBC V3.1. No V4 release exists.
- **Research status:** paused after Pass 47.
- **Canonical research baseline:** published V3.1 ZIP; [baseline note](PBC31_CANONICAL_BASELINE.md) and [JSON](PBC31_CANONICAL_BASELINE.json).
- **Read first:** [consolidated archive](PBC_RESEARCH_ARCHIVE.md), [future handoff](PBC_FUTURE_HANDOFF.md), [closure](PBC_RESEARCH_CLOSURE.md), [public maintenance audit](PBC_PUBLIC_MAINTENANCE_AUDIT.md).

## Pass index

Each `research_passN/` retains designs, scripts, patches/source, compact results and relevant tests. Large generated outputs are generally ignored. The [machine-readable index](RESEARCH_INDEX.json) lists all 47 passes and primary reports; the [archive](PBC_RESEARCH_ARCHIVE.md) synthesizes hypotheses, results, limitations and stopping decisions.

| Range | Stage | History |
|---|---|---|
| 1–16 | Baseline, V3.1 diagnosis, runtime and benchmark work | [Passes 1–16](PBC_RESEARCH_ARCHIVE.md#v31-diagnosis-optimization-and-packaging-passes-1-16) |
| 17–28 | Native residual/transform studies and distribution engineering | [Passes 17–28](PBC_RESEARCH_ARCHIVE.md#first-v4-representation-investigations-passes-17-28) |
| 29–34 | V4 grouping, prediction and sparse detail | [Passes 29–34](PBC_RESEARCH_ARCHIVE.md#v4-encoding-and-validation-research-passes-29-34) |
| 35–41 | Pinned AV1/libaom references | [Passes 35–41](PBC_RESEARCH_ARCHIVE.md#av1libaom-investigation-passes-35-41) |
| 42–47 | V4-A, canonical reconciliation, structural strokes and oracle | [Passes 42–47](PBC_RESEARCH_ARCHIVE.md#first-multiresolution-v4-a-and-closure-passes-42-47) |

## Reproduction

Read [reproduction guidance](REPRODUCTION.md) and [baseline provenance](PBC31_CANONICAL_BASELINE.md). Verify published ZIP/model/preset/input hashes; apply EXIF transpose and RGB conversion; measure actual completed files and source-resolution RGB reconstruction; compare matched or tightly bracketed rates; state CPU/thread, warmup, repetition and timer boundaries. `hpt_data` and `hpt_data_val` are external/local inputs and were not committed. Pass 35 documents pinned AV1 source and build dependencies.

## Branch and provenance

`research/experiments` is for archival/reproducible experiments and must not be merged wholesale to `main`. Production changes, releases, tags, Hugging Face deployment and portfolio changes need separate authorization. Model SHA-256: `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`. Published ZIP SHA-256: `3ee8f68b45adb3a89128251b8905d93cb99a96d911a055f91e124e724abbe069`. Preset JSON SHA-256: `5a2eaab04bbfe373ec0d8e4e1f059b2003b5d90f31d12178382550682780338d`.

## Known unresolved points

- The reported HF screenshot 4.65 KiB / MSE 88.87 Coast output cannot be reproduced from the available returned stream/input. Cause is unknown.
- Canonical manifest is provisional: desktop executable, exact HF Linux native library build and cross-platform byte identity were not verified.
- HF-compatible and pinned research AVIF implementations differ; do not equate their “quality 0” points.
- Large raw outputs/traces may be excluded or ignored; see [artifact manifest](PBC_ARCHIVE_ARTIFACTS.json) and pass reproduction notes. The archive does not claim complete rebuildability from Git alone.
