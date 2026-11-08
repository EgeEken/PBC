# Preset consistency

The public HF `/api/presets` endpoint was queried read-only. Its four resolved configs matched the frozen V3.1 source’s `PBC3Config` preset factories, including dataclass defaults. The published ZIP’s embedded `pbc3_types.py` has the same preset values and dataclass fields as the frozen source in the inspected ranges. The four preset factories set:

| Preset | patch_count | search_q_start/end | q_init/start/end | learned_filler_q | learned enabled | learned candidates/top-k | other defaults |
|---|---:|---|---|---:|---|---|---|
| Compression | 50 | 0.5 / 0.2 | 0.7 / 0.8 / 0.8 | 0.4 | true | 1 / 1 | defaults below |
| Balanced | 50 | 0.5 / 0.2 | 0.7 / 0.8 / 0.8 | 0.6 | true | 1 / 1 | defaults below |
| Quality | 50 | 0.5 / 0.2 | 0.7 / 0.8 / 0.8 | 0.8 | true | 1 / 1 | defaults below |
| High Quality | 20 | 0.7 / 0.2 | 0.7 / 0.8 / 0.8 | 0.95 | true | 1 / 1 | defaults below |

Shared defaults include search/proposal/exact depths 200/50/10; patch size 16–400; cell size 1–64; top_k 20; YCbCr/Sum; init depth 3; downsample init enabled, cell size 12, palette bits 6; auto working-size cap 250,000 pixels; `warmup_ratio=-1`; patch palette bits 2; LZMA enabled; seed 2003; `reuse_selected_delta=true`; bicubic residual projection; gradient cell selection; representability threshold 0. The model SHA-256 is `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`.

HF Auto mode directly calls the selected factory and sets the learned candidate/top-k values to 1 (already the preset values). If a learned-q form value is submitted, it overrides that preset value. Manual mode constructs a new config from submitted fields and is not a preset. The inspected `static/app.js` Auto request sends `mode` and `auto_config`, not the learned-q slider. The API’s default Auto preset is Quality; a screenshot’s “Compression” label alone does not prove the request or config that generated the file.

**Classification:** No resolved preset difference was found among HF, frozen research, and the inspected release Python config. Published release Python matched all 12 archived frozen streams for Coast, Cat and Googlewiki. The five-image published-source corpus also completed and was deterministic. This was not a direct live HF encode. Local installed desktop app config is not verified.
