# Pass 45 results

**What caused the Hugging Face / research mismatch?** The PBC mismatch is not reproduced by the available published or research sources. The exact HF upload and returned stream are unavailable, so the incident’s cause remains unproven. Current HF preset/source/model identity matches the frozen research baseline; all 12 Coast/Cat/Googlewiki preset outputs from current release Python match the archived research streams and decoded pixels. The older Pass 26 ZIP’s Coast Compression output also matches exactly. The screenshot reports 4.65 KiB and MSE 88.87, which none of these verified Coast Compression runs produces.

## Findings

1. **Implementations.** HF HEAD is `98ade0c…`; frozen research is tied to `8006f30…`. Their relevant codec source hashes and model match. The current public release ZIP differs from its tag source and HF/research source. Its embedded Python matches all 12 archived frozen streams/pixels for Coast, Cat and Googlewiki. The user-installed desktop executable was not run.
2. **First divergence.** The Pass 26 record identifies an earlier 130 KB public ZIP (SHA `edec3ec…`); the current public release ZIP is 76.9 MB (SHA `3ee8f68…`). The earliest repackaging or commit event is not established. The older archived asset matches current release Python on eight Quality/High Quality streams for Cat, Flower, Googlewiki and Mario64.
3. **Input.** Research Coast file and decoded RGB hashes are recorded in `INPUT_PROVENANCE.md`. The demo upload pixels cannot be hashed. Pillow AVIF’s exact reproduction makes same-source identity likely, but not proven.
4. **Model/presets/RNG.** Model hashes match. Current HF live presets match the frozen config definitions and release Python preset factories. Release-mode RNG is NumPy `default_rng(seed=2003)`. All four presets match across current release Python and frozen research for Coast, Cat and Googlewiki; HF Auto submits only its preset name, while Manual is a separate config path.
5. **Metrics.** Independent safe int32 RGB MSE for the stored Coast reconstruction exactly matches the internal MSE, 73.1335828. No metric bug is indicated for this record. The demo incident MSE cannot be verified without the stream/upload.
6. **AVIF.** Pillow 11.3.0 quality 0 on research Coast emits 7,647 bytes and MSE 72.3782, matching the user’s rounded report. Pass 44’s pinned libavif/libaom profile emits 6,747 bytes and MSE 88.537. They differ in frontend defaults and CICP metadata (Pillow path observed 1/13/6; pinned 1/13/1); q0 is not a bitrate.
7. **Canonical choice.** The published GitHub ZIP is the provisional compatibility reference; use embedded Python as the source-level oracle pending execution of the bundled binary. The canonical manifest is marked provisional.
8. **Timings.** Published ZIP Python on Coast, one warmup and three single-thread in-process calls: encode medians were 1.1085, 1.0479, 0.7941 and 1.0035 seconds for Compression through High Quality; decode medians were 0.1984, 0.2026, 0.1944 and 0.1986 seconds. Timings are local and not comparable with HF UI values. Pass 44’s frozen-source Coast Compression decode median was 0.116 seconds. A fresh attempt to re-time the frozen source exceeded 60 seconds without output and was stopped, so the source-level timing gap is unresolved.
9. **Corrections and tests.** Added a reusable encoder probe, HF preset checker and corpus runner. Three regression tests passed against the published ZIP’s embedded Python. The corpus runner verified all four presets on Coast, Cat, Googlewiki, Mario64 and Flower, deterministic repeated encodes, file-only decoding, and separate RGB/alpha metrics for synthetic RGBA. All 12 Pass 44 rows for Coast/Cat/Googlewiki matched; eight Pass 26 archived Quality/High Quality streams for Cat, Flower, Googlewiki and Mario64 matched. No product-alignment patch was justified. The desktop executable remains unverified.
10. **Prior research classification.** Pass 26 PBC points are valid with qualifications: eight preserved Quality/High Quality files match current release Python, while Monet and its Compression/Balanced outputs were not rechecked. Its Pillow AVIF points remain a distinct profile. Passes 31–34 are valid with qualifications for the pinned `8006f30…` source/model and tested streams; retain each named AVIF profile and limited dataset. Passes 42–44 V3 rows are valid for their three development images and four presets because all 12 current release Python streams/pixels match. Pass 44’s AVIF values remain valid only for its explicit pinned profile. The HF screenshot remains unverified. No historical reports were rewritten.

## Gates and outstanding work

| Gate | Status |
|---|---|
| Public artifact/revision inventory | Partial: archive and hashes verified; binary not executed |
| HF source/preset/model inventory | Pass for current read-only revision/API |
| Exact Coast source identity | Partial; research hash known, demo upload unavailable |
| Coast release-Python vs research output | Pass for all four presets |
| Four preset cross-implementation equality | Pass for Coast/Cat/Googlewiki between release Python and frozen files; release Python deterministic on Mario64/Flower; direct HF encode not run |
| AVIF profile reconciliation | Pass for local Pillow reproduction and pinned-profile distinction; HF linked encoder build unverified |
| Local installed desktop app | Not verified |
| Bitstream compatibility matrix | Partial: 12 frozen research files and eight Pass 26 files match/decode under release Python; bundled executable and other platforms untested |
| Alignment/deployment | No changes prepared for products; deployment remains unauthorized |

No files were staged, committed, pushed, deployed, or uploaded. Continue only with the remaining evidence needed to attribute the incident; do not restart V4 tuning from an assumed baseline.
