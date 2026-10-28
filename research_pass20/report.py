import csv
import dataclasses
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass20.common import HERE,INITIAL,ROOT,PBC3Config,meta,sha,write_json
from research_pass17.report import interpolate


def main():
    screen=json.loads((HERE/'screen.json').read_text())['rows']
    costs=json.loads((HERE/'costs.json').read_text())['rows']
    details=json.loads((HERE/'cost_details.json').read_text())
    controls=[]
    for name in ('baseline.json','extra_controls.json','targeted_controls.json'):
        controls.extend(r for r in json.loads((ROOT/'research_pass17'/name).read_text())['rows'] if r['image'] in INITIAL)
    high=json.loads((ROOT/'research_pass18/representation.json').read_text())['googlewiki']['higher_resolution']
    controls.append(dict(image='googlewiki.png',preset='high_quality_500k',bytes=high['bytes'],rgb_mse=high['rgb_mse'],config=high['config'],stream_sha256=high['stream_sha256']))
    old=[{**r,'preset':f"P17_q{r['step']}"} for r in json.loads((ROOT/'research_pass17/hybrid_results.json').read_text())['rows'] if r['image'] in INITIAL]
    bands=[dict(image=r['image'],preset='P17_'+b['kind']+'_q128',**b) for r in costs for b in r['band_repacks']]
    def dominates(a,b):return a['bytes']<=b['bytes'] and a['rgb_mse']<=b['rgb_mse'] and (a['bytes']<b['bytes'] or a['rgb_mse']<b['rgb_mse'])
    comparisons=[]
    for row in screen:
        ordinary=[r for r in controls if r['image']==row['image']]
        frozen=[r for r in old if r['image']==row['image']]
        diagnostic=[r for r in bands if r['image']==row['image']]
        close=[r for r in ordinary if abs(row['bytes']/r['bytes']-1)<=.05]
        best=min(close,key=lambda r:r['rgb_mse']) if close else None
        hq=next(r for r in ordinary if r['preset']=='high_quality')
        baseline128=next(r for r in frozen if r['step']==128)
        bracket=interpolate(ordinary,row['bytes'],'bytes','rgb_mse')
        prior_bracket=interpolate(frozen,row['bytes'],'bytes','rgb_mse')
        comparisons.append(dict(image=row['image'],base_budget=row['base_budget'],setting=row['setting'],bytes=row['bytes'],rgb_mse=row['rgb_mse'],hq_rate_percent=100*(row['bytes']/hq['bytes']-1),hq_mse_percent=100*(row['rgb_mse']/hq['rgb_mse']-1),strictly_dominates_hq=dominates(row,hq),ordinary_dominated_by=[r['preset'] for r in ordinary if dominates(r,row)],dominates_ordinary=[r['preset'] for r in ordinary if dominates(row,r)],frozen_dominated_by=[r['preset'] for r in frozen if dominates(r,row)],band_dominated_by=[r['preset'] for r in diagnostic if dominates(r,row)],p17_q128_rate_percent=100*(row['bytes']/baseline128['bytes']-1),p17_q128_mse_percent=100*(row['rgb_mse']/baseline128['rgb_mse']-1),best_ordinary_within5pct=None if best is None else dict(preset=best['preset'],bytes=best['bytes'],rgb_mse=best['rgb_mse'],signed_rate_gap_percent=100*(row['bytes']/best['bytes']-1),mse_change_percent=100*(row['rgb_mse']/best['rgb_mse']-1)),ordinary_bracket=bracket,frozen_bracket=prior_bracket,ordinary_interpolated_mse_percent=100*(row['rgb_mse']/bracket['value']-1) if bracket else None,frozen_interpolated_mse_percent=100*(row['rgb_mse']/prior_bracket['value']-1) if prior_bracket else None))
    gate=dict(googlewiki_ordinary_success=any(r['image']=='googlewiki.png' and r['dominates_ordinary'] and not r['ordinary_dominated_by'] for r in comparisons),mario_ordinary_success=any(r['image']=='mario64.png' and r['dominates_ordinary'] and not r['ordinary_dominated_by'] for r in comparisons),positive_frozen_improvement=any(r['image']!='cat.jpg' and r['frozen_interpolated_mse_percent'] is not None and r['frozen_interpolated_mse_percent']<=-3 for r in comparisons),cat_frozen_improvement=any(r['image']=='cat.jpg' and r['frozen_interpolated_mse_percent'] is not None and r['frozen_interpolated_mse_percent']<0 for r in comparisons))
    gate['passed']=all(gate.values())
    assert not gate['passed'] and len(screen)==12
    write_json('comparisons.json',dict(method='Actual dominance and strongest ordinary 5%-near control first, retaining signed gaps. Interpolation takes lowest measured bracketing chord; no extrapolation. Frozen full Pass17 and new A-band diagnostics remain separate.',ordinary_controls=controls,frozen_controls=[{k:r[k] for k in ('image','preset','step','bytes','rgb_mse')} for r in old],band_controls=bands,comparisons=comparisons,extension_gate=gate))
    fields=['image','base_budget','setting','bytes','bpp','rgb_mse','base_bytes','coarse_prefix_xz_bytes','fine_incremental_xz_bytes','shared_enhancement_bytes','container_bytes','coarse_raw_bytes','fine_raw_bytes','coarse_only_rgb_mse','encode_seconds','decode_seconds','process_peak_working_set','stream_sha256','pixels_sha256']
    with (HERE/'files.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(screen)
    cost_table='\n'.join(f"| {r['image']} | {r['spans']['dct']*1000:.1f} | {r['spans']['xz_extreme']:.3f} | {r['full_encode_seconds']:.3f} / {r['full_decode_seconds']:.3f} | {r['zero_fraction']*100:.2f}% |" for r in costs)
    entropy_table='\n'.join(f"| {r['image']} | {r['entropy_summary']['extreme']['bytes']:,} / {r['entropy_summary']['extreme']['encode_seconds']:.3f} | {r['entropy_summary']['normal']['bytes']:,} / {r['entropy_summary']['normal']['encode_seconds']:.3f} | {100*(r['entropy_summary']['normal']['bytes']/r['entropy_summary']['extreme']['bytes']-1):+.2f}% |" for r in costs)
    band_table='\n'.join(f"| {r['image']} | {r['full_bytes']:,} / {r['full_mse']:.3f} | {r['band_repacks'][0]['bytes']:,} / {r['band_repacks'][0]['rgb_mse']:.3f} | {r['band_repacks'][1]['bytes']:,} / {r['band_repacks'][1]['rgb_mse']:.3f} |" for r in costs)
    normal_table='\n'.join(f"| {r['image']} | {r['bytes']:,} | {r['rgb_mse']:.3f} | {r['encode_seconds']:.3f} / {r['decode_seconds']:.3f} |" for r in details['normal_full_rows'])
    (HERE/'PBC_RESIDUAL_COST_ANALYSIS.md').write_text(f'''# Pass 20 residual cost analysis

Pass 17 encoding is dominated by its actual preset-0 EXTREME XZ compressor, not the 8x8 DCT. Removing EXTREME saves substantial CPU but incurs a substantial rate penalty. This is an independent tradeoff, not a free production improvement.

## Method and provenance

Start research/experiments at 0a2f1e80d7fd5aaed5acf683eea03aeb1a7b947e; unchanged pbc31_feature_histogram and frozen Pass 17. Googlewiki, Mario64 and Cat, q128/p10. Saved base/coefficient payloads are reused; rebuilt complete original files exactly match the saved Pass 17 SHA. No ordinary RD frontier is rerun for this analysis.

Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0, SciPy 1.17.1, i9-12900H pinned logical P-core 12, numerical threads/FFT workers one. Full warmup precedes measurements. costs.json contains disjoint instrumented stages and separately measured uninstrumented complete calls; decode spans are not added to encode spans. All expanded controls, source/model/input identities and measured source hashes are retained. These profile/full calls are one-pass observations; only the entropy microcomparison has three alternating repetitions.

Stages separately measured: base encode, base decode inside encode, residual construction, color transform, DCT, quantization, coefficient reorder/copy, entropy and container; decoder base, entropy, coefficient view/dequantization, IDCT, color/crop and round/clip. All raw spans are in costs.json.

| Image | DCT ms | Instrumented XZ extreme s | Separate full encode / decode s | Zero q128 coefficients |
| --- | ---: | ---: | ---: | ---: |
{cost_table}

XZ is 81-87% of the sum of the disjoint encode spans. Base encode is about 0.227/0.238/0.544 s; base decode inside the encoder 0.031/0.039/0.192 s. Color/DCT/quantization/copy and final reconstruction remain real costs after entropy is accelerated. Do not add sampled span percentages to earlier speed measurements.

## Exactly two entropy settings

The frozen code uses `lzma.PRESET_EXTREME` alone, meaning **preset 0 plus the EXTREME flag**. Compare conventional **preset 0**, changing only that flag. Preset 6/default and other parameters were not tested; there is no parameter sweep. Identical signed int16 frequency-plane bytes, one XZ frame, three alternating compression/decompression measurements. Every compressed result expands to identical bytes. Table reports medians.

| Image | Extreme bytes / encode s | Normal-0 bytes / encode s | Enhancement size change |
| --- | ---: | ---: | ---: |
{entropy_table}

Normal-0 is about 22-54x faster in compression but increases enhancement bytes by 33.02-42.01%. The total file increase is smaller because the base is unchanged. Entropy decompression medians change from 8.75/7.78/48.92 ms to 11.42/8.68/65.30 ms. It does not improve entropy decode time.

The same setting was also wrapped into actually decodable P17D files, with one warmed **complete** encode/decode call per case. These timings include fresh base encoding, base decoding inside the encoder, transforms, entropy and reconstruction; all independent RGB scoring and I/O are outside the timers. Output pixels exactly match frozen q128.

| Image | Normal-0 complete bytes | RGB MSE | Full encode / decode s |
| --- | ---: | ---: | ---: |
{normal_table}

These are full-call observations, not sums of microbenchmarks or repeated speed claims. Googlewiki still has a completed-file win against measured ordinary HQ and its 500k HQ control; Mario64 is 1.53% larger than ordinary HQ at lower MSE; Cat remains far too large. This entropy-only tradeoff is not a validated universal candidate. No production or frozen Pass 17 source was edited; a scoped hook is used only in a standalone research process and restored afterward.

## Coefficient usefulness and actual repacks

Frequency-plane zero counts do not assign compressed bytes to frequencies. Cat is nearly 99% zero yet still requires a dense roughly 51 MB coefficient byte array. Dense traversal, copying and entropy processing cost work even on zeros. Process peak-working-set observations are cumulative high-water marks, not isolated per-variant allocations or an allocator benchmark.

At p10/q128, low u+v<4 energy fractions are about 19.91% Googlewiki, 79.36% Mario64 and 77.64% Cat. Coefficient counts and original-band energy refer to this p10 residual, not the distinct HQ residual diagnostics from Pass 17.

Two frozen A-band diagnostics omit low or high bands, repack a complete ordinary P17D container with the retained coefficients and zeros elsewhere, and independently decode it. They preserve the full dense coefficient layout and extreme entropy setting. No omitted frequency is assumed free.

| Image | Full q128 bytes / MSE | Low-only bytes / MSE | High-only bytes / MSE |
| --- | ---: | ---: | ---: |
{band_table}

Dropping high frequencies badly damages Googlewiki despite their differing energy/count distributions. Dropping low frequencies damages Mario64 and Cat. The low-only Mario64 and Cat repacks are useful additional diagnostic controls, but they use the old slow entropy setting and are not practical promotions. Counts, energy fractions and completed repack sizes answer different questions; their byte contributions are not additive.

The multiscale experiment uses normal-0 to test the latency path directly. Its rate penalty remains counted in every final file. The next report separates that entropy tradeoff from the loss caused by the coarse/fine frequency partition.
''',encoding='utf-8',newline='\n')
    full_table='\n'.join(f"| {r['image']} | p{r['base_budget']} {r['setting']} | {r['bytes']:,} | {r['rgb_mse']:.3f} | {r['base_bytes']:,} | {r['coarse_prefix_xz_bytes']:,} | {r['fine_incremental_xz_bytes']:,} | {r['encode_seconds']:.3f} / {r['decode_seconds']:.3f} |" for r in screen)
    close_rows=[]
    for c in comparisons:
        if c['base_budget']==10 and c['setting']=='lean' and c['best_ordinary_within5pct']:
            r=c['best_ordinary_within5pct'];close_rows.append(f"| {c['image']} | {r['preset']} | {r['bytes']:,} / {r['rgb_mse']:.3f} | {c['bytes']:,} / {c['rgb_mse']:.3f} | {r['signed_rate_gap_percent']:+.2f}% | {r['mse_change_percent']:+.2f}% |")
    diagnosis_table='\n'.join(f"| {r['image']} | {100*r['omitted_remaining_energy_fraction']:.2f}% | {r['coded_multiscale_rgb_mse']:.3f} | {r['unencoded_all_band_rgb_mse']:.3f} |" for r in details['omitted_band'])
    control_times=json.loads((HERE/'control_times.json').read_text())['rows']
    time_table=[]
    for name in INITIAL:
        c=next(r for r in control_times if r['image']==name)
        r=next(r for r in screen if r['image']==name and r['base_budget']==10 and r['setting']=='lean')
        time_table.append(f"| {name} | {c['encode_seconds']:.3f} / {c['decode_seconds']:.3f} | {r['encode_seconds']:.3f} / {r['decode_seconds']:.3f} | {r['encode_seconds']/c['encode_seconds']:.2f}x | {r['decode_seconds']/c['decode_seconds']:.2f}x |")
    (HERE/'PBC_MULTISCALE_RESIDUAL_RESULTS.md').write_text(f'''# Pass 20 multiscale completed-file results

Stop after the fixed three-image screen. The prototype retains some ordinary-PBC RD wins but fails to improve the frozen Pass 17 frontier. Nine of twelve files are strictly dominated by frozen full-DCT files; the remaining three are dominated by the actual A-band repacks. No six-image extension, repeated candidate timing or activation rule follows.

## Frozen representation

P20M is a research container, not a version or production default. A 40-byte header stores original width/height, base/payload lengths, coarse width/height, two float32 quantizer steps, block size 8, scale 4, fine band floor 4 and filter ID 1. Fixed color matrix and orthonormal DCT are inherited from Pass 17.

The original RGB residual is computed against an independently decoded ordinary Quality base. Analyze it channelwise using signed float32 PIL mode-F bicubic at ceil(width/4), ceil(height/4), reducing_gap=None. DCT/color-transform and quantize the reduced residual. Reconstruct those stored quantized coefficients, inverse-color-transform, then upsample mode-F bicubic to original dimensions. This exact decoder correction is subtracted from the original residual before the fine DCT. There is no double-counting of the same coarse correction.

Fine coding retains u+v>=4 (54/64 coefficients) at original-size 8x8 blocks. Coarse and retained fine coefficients are little-endian int16, frequency-plane ordered, concatenated into **one shared preset-0 XZ stream**. Dimensions determine the raw split; no tile frames or source image/model are needed at decode. Add decoded base + coarse + fine in float32, then np.rint, clip [0,255], uint8.

Two predeclared settings only: lean qc32/qf128, rich qc16/qf64. Two actual Quality base budgets: p10 and p5; p10 bytes match the frozen Pass 17 base, and p5 is verified identical across quantizer settings. Every timed full encode includes fresh base compression with internal MSE enabled. No patch budget is treated as a nested prefix. PLAN.md was fixed before measurements.

## Actual complete files

Every file includes its full base, shared coefficient stream and 40-byte header. Coarse XZ is a diagnostic compression of the coarse prefix alone; fine delta is final shared size minus that prefix size. They sum to actual enhancement bytes but **are not unique compressed-byte ownership**. They are not two separately stored entropy frames. Actual raw coarse/fine bytes and all hashes are in files.csv/screen.json.

| Image | Config | Final bytes | RGB MSE | Base bytes | Coarse XZ prefix | Fine XZ delta | Full encode / decode s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{full_table}

The p10 lean files strictly dominate default ordinary HQ on Googlewiki (-13.85% bytes/-56.34% MSE) and Mario64 (-8.79%/-16.35%). Cat is a tradeoff: +171.29% bytes/-36.80% MSE. This is not sufficient: both positive p10 lean files are themselves dominated by frozen full-DCT q128 (Googlewiki 136,602/222.834; Mario64 78,167/64.788).

Against stronger actually measured 5%-near ordinary controls:

| Image | Ordinary control | Ordinary bytes / MSE | p10 lean bytes / MSE | Signed rate gap | MSE change |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(close_rows)}

Googlewiki p10 lean also strictly dominates the available same-runtime ordinary 500k HQ control (214,902 bytes/MSE 439.912). Other same-runtime 500k controls were not available; no new dense ordinary frontier was generated. Full controls/config provenance, strongest close controls, actual dominance and conservative lowest-bracketing-chord estimates are in comparisons.json. No extrapolation or all-image BD-rate claim.

The smaller p5 base does not dominate p10 in either setting on any image. It saves little on Googlewiki and Cat, with worse quality. Mario64 lean falls to 74,484 bytes/MSE 119.563, near the ordinary-HQ quality, but its rate is 4.71% below frozen q128 with 84.55% higher MSE; this signed near-rate comparison favors the old quality point and is not called equal-rate. The low-only q128 diagnostic is actually smaller and better (62,519/110.787). Cat's p10/p5 lean files, 211-214 KB/MSE ~185, are dominated by the low-only diagnostic 168,749/116.045. All rich files have large rate penalties. All twelve are dominated by at least one actual full-or-band Pass 17 control, with the categories kept separate.

No signal justifies extending to Monet/Eiffel/Flower. Flower is untested in this pass, not assumed fixed. Cat remains the explicit photographic failure. This is not a universal encoder policy or an oracle-selected codec.

## Coarse/fine failure diagnosis

Coarse-only rounded/clipped RGB MSE for p10 lean is 860.308 Googlewiki, 151.006 Mario64 and 209.297 Cat. Adding fine data improves these to 311.886/102.077/185.264, but costs XZ deltas 90,084/22,900/108,908 bytes. Cat's coarse prefix alone is 58,960 bytes, before its 46,135-byte PBC base and header. Total fine cost largely cancels the coarse coding savings.

The coarse analysis/synthesis filter is not complementary to a fixed original 8x8 frequency cutoff. After subtracting the exact quantized coarse reconstruction, substantial **low 8x8-band information remains**:

| Image | Remaining energy in omitted bands | Coded p10 lean RGB MSE | Unencoded all-band reconstruction MSE |
| --- | ---: | ---: | ---: |
{diagnosis_table}

The last column restores the omitted quantized bands as an explanatory reconstruction only. No all-band P20 file was encoded, no rate is assigned, and it is not a third screened setting or a demonstrated RD repair. This isolates a significant frequency-selection loss; merely counting low-band energy as already represented by the coarse layer was wrong. Small/high-frequency text coefficients also matter, as the separate A-band repacks show.

The new format adds only ten header bytes over Pass 17, so metadata is not the meaningful cause of the rate failure. Normal-0 entropy worsens coefficient compression, while fixed-band omission loses quality and dense fine coefficients retain most storage/traversal cost. Scale 4 adds only roughly 1/16 coarse samples but still stores 54/64 full-resolution frequencies, so raw coefficient count falls only about 9% before boundary padding. The fine DCT still computes every frequency before selection. This design is not a substantially sparse transform implementation.

## Practical runtime and memory

One warmed complete call per file, with loading/I/O/hashing/external scoring outside the timers. New HQ timing observations reproduce existing control file hashes; no new RD controls. These batches are **not three-repeat interleaved timing evidence**. No fine-grained speed percentages, universal overhead or release claims follow. Repetition was conditional on the RD gate, which failed.

| Image | Ordinary HQ encode / decode s | p10 lean encode / decode s | Observed encode ratio | Observed decode ratio |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(time_table)}

Fast entropy makes encoding useful on the two small positive images in this screen, but Cat still exceeds 2x ordinary encode time and all decodes exceed the 10-15% overhead target. Coarse inverse DCT + original-size float upsample + fine inverse DCT/copies are additional decoder work. Cat's process high-water working set reaches roughly 1.47 GiB by the rich cases; this is cumulative within the screening process, not an independent per-file peak or fair cross-process memory comparison. There is no evidence that optimizing only Python bookkeeping would remove this architectural decoder work.

## Correctness and disposition

Twelve completed P20 files and nine A entropy/band P17 repacks pass independent size/SHA/pixel/MSE checks. Raw layer readback, signed int16 range +/-32,000, coarse-before-fine reconstruction, final round/clip, four repeated non-block-aligned fixtures, 16 malformed/magic rejection cases and two fresh file-only subprocess decodes with policy loading forbidden pass. Ordinary v0 and frozen P17 decoders reject P20 magic; production source/model remains byte-identical.

Do not promote or extend this prototype. No content activation is investigated because it failed the multi-image RD gate; an inexpensive routing policy is not established, and an offline best-of-files oracle is not reported as deployable. The ordinary practical codec and staged ZIP are unchanged. Strongest follow-up: preserve useful single-scale residual information while reducing dense/sparse coefficient entropy cost, and make any scale partition match its analysis/synthesis filters. The preset test shows large removable encode cost, but normal-0's 33-42% payload penalty is not the solution. No additional preset, scale or coding experiment was launched.
''',encoding='utf-8',newline='\n')
    (HERE/'PBC_PASS20_SUMMARY.md').write_text('''# Pass 20 summary

Stop the frozen multiscale direction after twelve completed files on Googlewiki, Mario64 and Cat. No six-image extension, three-repeat candidate timing, activation classifier, production change or release rebuild.

1. Pass 17 cost: extreme XZ accounts for about 81-87% of measured disjoint encoding spans. Dense zero-heavy coefficient arrays still require copies/scans; base decode and full inverse transforms add work.
2. DCT versus entropy: DCT ~18/32/316 ms versus extreme XZ ~1.98/1.55/6.24 s in the instrumented Googlewiki/Mario/Cat calls. Separately measured complete calls are retained, not synthesized by adding overlapping profiles.
3. Exactly two settings: preset0+EXTREME versus normal preset0. Normal compression is 22-54x faster, but enhancement size rises 33-42%. Full normal P17 files preserve pixels and encode in about 0.366/0.467/1.708 s in one-pass observations; this rate/runtime tradeoff is not a free improvement or production promotion.
4. Coarse layer: reduced scale4 residual coding is compact on text but cannot cover all low/mid 8x8-band information. Cat coarse prefix alone is ~59 KB plus a 46 KB base; it does not solve its low-rate requirement.
5. Fine layer: improves actual MSE but consumes much of the savings. Omitted low bands still hold 16.70%/48.60%/66.44% of remaining residual energy. Restoring them in an unencoded diagnostic greatly improves MSE, with no proven byte cost or RD repair.
6. Versus frozen Pass17: nine files are strictly dominated by full frozen DCT controls; all twelve are dominated when the actual predeclared A-band repacks are included. The latter remain separate diagnostic controls, not relabeled frozen full architecture.
7. Versus ordinary: p10 lean dominates default HQ on Googlewiki and Mario and improves MSE against strong actual near-rate controls. Googlewiki also dominates its available 500k HQ control. These wins do not overcome worse RD than Pass17.
8. Content outcomes: two small positive cases retain ordinary wins, Cat remains expensive. Monet/Eiffel/Flower were not tested because the fixed extension gate failed; no claim that Flower is repaired.
9. Allocation: p5 saves bytes with worse quality; it never dominates p10 here. Budgets are actual fresh encodes and are not nested prefixes. No budget sweep.
10. Runtime: full calls, one-pass only. p10 lean encode/decode ratios versus ordinary HQ are roughly 0.70/2.17 Googlewiki, 1.00/3.64 Mario, 3.44/8.88 Cat. These are unrepeated feasibility observations, not precise benchmark estimates. Decoder targets fail; Cat encode remains >2x.
11. Activation: not evaluated after failed RD gate. No classifier, image-name switch, second-full-encode routing or oracle policy. Dimensions/sparsity alone are not established predictors.
12. Broader validation: not recommended for this representation. All completed files pass decoding/readback/determinism/framing checks. Existing pbc31_feature_histogram, pbc17_dct_detail_v1, v0 model/defaults and staged release are preserved.
13. Next: reduce entropy cost without sacrificing most of the extreme preset's compression, and preserve residual information with a filter-compatible scale split or efficient sparse single-scale representation. No rescue preset/transform/quantizer sweep launched.
14. Research GitHub commit and verified final HEAD are supplied in the chat handoff. Only research_pass20 source, fixed plan, reports and compact evidence are committed; generated streams/coefficient arrays/caches remain ignored.

Reproduce using Python 3.11 and requirements.txt with restored hpt_data and the existing Pass17 saved q128 files (or regenerate those archived artifacts using the prior scripts):

```
research_pass16/.venv/Scripts/python.exe research_pass20/profile_costs.py
research_pass16/.venv/Scripts/python.exe research_pass20/screen.py
research_pass16/.venv/Scripts/python.exe research_pass20/finish_costs.py
research_pass16/.venv/Scripts/python.exe research_pass20/time_controls.py
research_pass16/.venv/Scripts/python.exe research_pass20/validate.py
research_pass16/.venv/Scripts/python.exe research_pass20/report.py
```

Do not run screen.py --extend unless the fixed gate passes in a separately justified future study; it was not run here. Source/model/input/plan hashes accompany measurements, full rows are in files.csv, controls/brackets in comparisons.json. Memory fields are cumulative process peak working sets. Staged ZIP SHA-256 remains 08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591. No HF operation, protected-branch change, release/tag or full 25x4 benchmark.
''',encoding='utf-8',newline='\n')
    provenance=meta();provenance['timing']='Disjoint profile and entropy-only repetitions; complete candidate/normal/HQ calls one-pass, no failed-candidate repetitions.'
    provenance['base_configs']={f'p{budget}':dataclasses.asdict(PBC3Config.quality(patch_count=budget,learned_filler_model_path=str(ROOT/'patch_policy.npz'))) for budget in (5,10)}
    provenance['profile_p10_config']=dataclasses.asdict(PBC3Config.quality(patch_count=10))
    provenance['ordinary_hq_config']=dataclasses.asdict(PBC3Config.high_quality())
    provenance['transform_configs']={'scale':4,'coarse_filter':'PIL float32 F bicubic; reducing_gap=None; ceil dimensions','fine_frequency_rule':'u+v>=4','lean':[32,128],'rich':[16,64],'entropy_preset':0,'header_bytes':40}
    provenance['evidence_sha256']={p.name:sha(p.read_bytes()) for p in HERE.glob('*.json') if p.name!='provenance.json'}
    provenance['staged_zip_sha256_unchanged']='08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591'
    write_json('provenance.json',provenance)
    print('gate',gate,'frozen dominated',sum(bool(r['frozen_dominated_by']) for r in comparisons),'full-or-band dominated',sum(bool(r['frozen_dominated_by'] or r['band_dominated_by']) for r in comparisons),flush=True)


if __name__=='__main__':main()
