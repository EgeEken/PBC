import json,pathlib,statistics,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
PASS=ROOT/'research_pass41'; OUT=PASS/'outputs'
points=json.loads((OUT/'phase_c_point_summary.json').read_text())
raw=json.loads((OUT/'phase_c_results.json').read_text())
planes=json.loads((OUT/'chroma_error_analysis.json').read_text())
phase_b=json.loads((OUT/'phase_b_results.json').read_text())

def table():
  lines=['| Image | q | Config A→B bytes | Δ bytes | Δ RGB MSE | Δ PSNR | Δ SSIMULACRA2 | Δ encode median | Δ decode median |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for x in points:
    a,b,d=x['A'],x['B'],x['delta']
    lines.append(f"| {x['image']} | {x['qcolor']} | {a['bytes']:,} → {b['bytes']:,} ({d['bytes_percent']:+.3f}%) | {d['bytes']:+,} | {d['rgb_mse_percent']:+.3f}% ({d['rgb_mse']:+.4f}) | {d['rgb_psnr_db']:+.4f} dB | {d['ssimulacra2']:+.4f} | {d['encode_seconds_median_percent']:+.2f}% | {d['decode_seconds_median_percent']:+.2f}% |")
  return '\n'.join(lines)

native=[]
for x in raw['native_plane_comparisons']:
  n=x['A_vs_B_native_plane_difference']
  native.append(f"| {x['image']} q{x['qcolor']} | {x['A']['Y']['mse']:.4f} → {x['B']['Y']['mse']:.4f} | {x['A']['U']['mse']:.4f} → {x['B']['U']['mse']:.4f} | {x['A']['V']['mse']:.4f} → {x['B']['V']['mse']:.4f} | {n['Y']['changed_samples']:,} / {n['U']['changed_samples']:,} / {n['V']['changed_samples']:,} |")

allocation=f'''# Existing chroma allocation control results

## Design

A is the unmodified Pass 35 libaom 3.12.1 reference with `aq-mode=0`, `tune=psnr`, `deltaq-mode=0`, and `sb-size=64`. B uses the same binary and settings plus `enable-chroma-deltaq=1`. Both use speed 6, one encoder thread, 8-bit 4:2:0, full range, CICP 1/13/1, and ignored metadata. The low-rate A points and their warm plus three timed runs were reused from Phase B. All other points have one warmup and three timed repetitions per configuration; high-rate runs were interleaved A/B.

Pairs are within 1.30% of file size, so no bounded rate calibration was needed. Scores are measured on the completed AVIF decoded to RGB. SSIMULACRA2 uses the established Pass 39 binary. Timings are medians of the three timed repetitions. A/B output hashes are deterministic across repetitions.

## Completed-file results

Negative RGB MSE change favors B. These are direct near-matched-rate points; each byte delta is shown explicitly.

{table()}

The `+2` chroma qindex offsets measurably affect the stream, but completed-file effects are very small and change direction. B improves MSE slightly at Googlewiki q20 and Cat q35; it worsens MSE at the other four points. The largest MSE change is +0.981% at Coast q90, with B also 1.292% larger. There is no consistent RGB or SSIMULACRA2 advantage. Decode-time shifts are noisy and do not indicate a stable decoder cost; encode changes range from −1.73% to +7.07% in the paired medians.

## Native decoded planes at the low-rate points

Plane MSE is against the source-derived reference planes produced by the pinned libavif input loader. A/B sample differences are decoded native samples that differ between configurations. Luma differences confirm that the option can also change coupled encoder decisions; the control is not a pure post-encode chroma replacement.

| Image / q | Y plane MSE A → B | U plane MSE A → B | V plane MSE A → B | Changed native samples Y / U / V |
|---|---:|---:|---:|---:|
{chr(10).join(native)}

AVIF hashes and decoded RGB hashes differ for all six A/B pairs. The option changes output bytes by −0.466% to +1.292%. Pinned `avifdec` and FFmpeg both decoded all 12 distinct AVIF files at their original dimensions.

## Decision

Gate C passes for verified activity: the option reaches the all-intra quantizer, changes native decoded Y/U/V, and changes the bitstream. It does not provide evidence for a useful rate-distortion improvement. The results do not justify Phase D's source modification: a fixed offset would be another chroma-offset experiment without a consistent allocation signal. No third configuration, rate calibration, source patch, or held-out run was made.

Raw rows, file hashes, decoder checks, native planes, and repeated timings are retained in ignored `outputs/phase_c_results.json`, `outputs/phase_c_point_summary.json`, `outputs/planes/`, and `outputs/benchmark/`.
'''
(PASS/'CHROMA_ALLOCATION_RESULTS.md').write_text(allocation,encoding='utf-8')

results=f'''# Pass 41 results

## Objective and setup

Pass 41 asked whether reallocation between luma and chroma improves completed-file AVIF RGB rate-distortion. It used the pinned Pass 35 static build: libaom 3.12.1 `{raw['sources']['libaom_commit']}` and libavif 1.3.0 `{raw['sources']['libavif_commit']}`. The worktree began at `3723dc14de88c6c3dcbd98dd80a097e019a92a9f` on `research/experiments`. The reference `avifenc` SHA-256 is `{raw['reference_avifenc_sha256']}`. No production code or upstream codec source was modified.

## Source-level findings

`enable-chroma-deltaq` is forwarded by libavif's `aomOptionDefs` to libaom's `AV1E_SET_ENABLE_CHROMA_DELTAQ` control (`libavif/src/codec_aom.c:409–424`). In libaom, `av1_cx_iface.c:1224` copies it into quantizer configuration; all-intra setup calls `av1_set_quantizer` in `av1/encoder/av1_quantize.c:866–961`. In the tested `tune=psnr` path it applies U/V DC and AC qindex offsets of +2; disabled means zero. This is a plane-level offset, distinct from spatial `deltaq-mode`, which was set to 0. The all-intra `tune=iq` 4:2:0 formula instead applies `-clamp((base_qindex / 2) - 14, 0, 16)`, but IQ changes other coding tools and was not tested as an isolated chroma setting.

`av1_init_plane_quantizers` applies the adjusted plane quantizers and QMs (`av1_quantize.c:785–843`). AV1 frame-header quantization syntax is written in `av1/encoder/bitstream.c:2109–2135`. Chroma transforms and CfL are selected through chroma RD paths including `cfl_rd_pick_alpha` (`av1/encoder/intra_mode_search.c:754–961`). Thus the option changes quantization and can alter coupled mode/transform/partition decisions, including luma. Exact source paths and routing are in `CHROMA_CODING_SOURCE_MAP.md`.

## Chroma error budget

The source planes were generated by the pinned libavif PNG/JPEG loaders using the same orientation handling, 4:2:0 conversion, CICP 1/13/1 and full-range settings as encoding. Decoded native Y/U/V were compared with those references. Counterfactual RGB used the file's signaled color properties and the same pinned YUV-to-RGB conversion as `avifdec`; the probe output matched pinned `avifdec` PNG pixels exactly. Restoring source-derived U/V reduced RGB MSE by 14.9% on Googlewiki q5, 10.8% on Cat q35, and 11.0% on Coast q80. Luma restoration reduced it by 49.4%, 85.5%, and 65.0%. These diagnostic improvements are non-additive. Chroma is meaningful but not dominant, so the small existing-control test was warranted.

## Existing control

A/B completed-file results are in `CHROMA_ALLOCATION_RESULTS.md`. B is active and changes output bytes and decoded planes; it is not a no-op. However, its RGB MSE changes from −0.137% to +0.981%, with only two of six points improving. SSIMULACRA2 changes range from −0.1328 to +0.0009. File sizes stay within 1.30% of each A point, and every measured effect is small. There is no consistent cross-content or cross-rate gain. The encode time median moves from −1.73% to +7.07%; these small samples show no meaningful speed win. Decode time is reported per point but treated as noisy subprocess timing.

Pinned `avifdec` and the already-installed FFmpeg decoded all 12 distinct outputs with valid original dimensions. No decoder change is needed for this standard AV1 control.

## Phase D and held-out decision

No source modification was justified. The existing control establishes a small +2 offset in PSNR all-intra mode, but its completed-file results do not identify a stable rate-efficient direction. A second fixed chroma offset would therefore be an unsupported tuning pass. Phase D and its development benchmark were not attempted; no patch hash exists. Held-out images were not used, and Gate E was not attempted.

## Gates and recommendation

- Gate A — pass: source plumbing and effective qindex behavior are mapped.
- Gate B — pass: chroma accounts for a repeatable, non-dominant part of low-rate RGB error.
- Gate C — pass for mechanism activity: output stream, planes, and bytes change. The control does not show a useful RD gain.
- Gate D — not attempted: no credible intervention was supported by Phase C.
- Gate E — not attempted: no frozen candidate met the development threshold.

Close this narrow luma–chroma offset direction. For a later pass, investigate a material mechanism such as AV1's internal prediction-aware RD objective or quantization-matrix selection, with a defined completed-file hypothesis. Do not start that work automatically as Pass 42.

No commit, push, staging, or cleanup was performed. Final worktree status is recorded in the handoff response.
'''
(PASS/'PASS41_RESULTS.md').write_text(results,encoding='utf-8')

summary={
 'pass':41,'objective':'Test whether enabling libaom chroma delta-Q improves completed-file AVIF RGB rate-distortion.',
 'repository':{'branch':'research/experiments','starting_head':'3723dc14de88c6c3dcbd98dd80a097e019a92a9f'},
 'sources':raw['sources'],'build':{'reference_avifenc_sha256':raw['reference_avifenc_sha256'],'reference_avifdec_sha256':hashlib.sha256((ROOT/'research_pass35/outputs/binaries/reference/avifdec.exe').read_bytes()).hexdigest(),'plane_probe_sha256':hashlib.sha256((OUT/'binaries/plane_probe.exe').read_bytes()).hexdigest(),'static_linkage':'Pass 35 pinned reference build'},
 'environment':{'os':'Windows 11','cpu':'Intel i9-12900H','logical_cpus':20,'python':'3.13.5','numpy':'2.2.6','pillow':'11.3.0'},
 'settings':raw['settings'],'input_sha256':phase_b['source_sha256'],
 'configurations':{'A':raw['settings']['configurations']['A'],'B':raw['settings']['configurations']['B'],'quality_settings':{'googlewiki.png':[5,20],'cat.jpg':[35,50],'coast.jpg':[80,90]}},
 'error_budget':{'points':planes['points'] if isinstance(planes,dict) and 'points' in planes else planes,'method':'same pinned libavif input conversion and signaled AVIF color conversion; float64 RGB MSE'},
 'phase_c_points':points,'native_plane_comparisons':raw['native_plane_comparisons'],'decoder_compatibility':raw['decoder_compatibility'],
 'patch':None,'held_out':{'images':['cathedral.png','park.jpg','ducklings.jpg'],'attempted':False,'reason':'No development intervention met the evidence threshold.'},
 'gates':{'A':'pass','B':'pass','C':'pass_mechanism_active_no_consistent_rd_gain','D':'not_attempted','E':'not_attempted'},
 'conclusion':'Existing chroma delta-Q is active but gives tiny, content-dependent RD changes; no Phase D patch or held-out evaluation is justified.'
}
(PASS/'PASS41_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print('wrote allocation, results, summary')
