import csv
import dataclasses
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass21.common import *
from research_pass21 import sparse_hybrid as codec,zero_runs
from research_pass17.report import interpolate


def main():
    coef=json.loads((HERE/'coefficients.json').read_text())
    files=json.loads((HERE/'files.json').read_text())['rows']
    timings=json.loads((HERE/'timings.json').read_text())
    profile=json.loads((HERE/'decode_profile.json').read_text())['rows']
    validation=json.loads((HERE/'validation.json').read_text())
    ordinary=[]
    for name in ('baseline.json','extra_controls.json','targeted_controls.json'):
        ordinary.extend(json.loads((ROOT/'research_pass17'/name).read_text())['rows'])
    high=json.loads((ROOT/'research_pass18/representation.json').read_text())['googlewiki']['higher_resolution']
    ordinary.append(dict(image='googlewiki.png',preset='high_quality_500k',bytes=high['bytes'],rgb_mse=high['rgb_mse'],config=high['config'],stream_sha256=high['stream_sha256']))
    gates=[]
    for name,step in INITIAL:
        old=next(r for r in coef['summary'] if r['image']==name and r['variant']=='dense_extreme')
        new=next(r for r in coef['summary'] if r['image']==name and r['variant']=='zero_runs')
        gates.append(dict(image=name,enhancement_change_percent=100*(new['payload_bytes']/old['payload_bytes']-1),packing_compression_speedup=old['total_encode_seconds']/new['total_encode_seconds'],unpack_decompression_ratio=new['total_decode_seconds']/old['total_decode_seconds']))
    passed=sum(r['enhancement_change_percent']<=10 for r in gates)>=2 and max(r['enhancement_change_percent'] for r in gates)<=15 and min(r['packing_compression_speedup'] for r in gates)>=5 and max(r['unpack_decompression_ratio'] for r in gates)<=2
    assert passed and len(files)==6 and all(r['exact_coefficients'] and r['exact_rgb'] for r in validation['files'])
    comparisons=[]
    for row in files:
        controls=[r for r in ordinary if r['image']==row['image']]
        hq=next(r for r in controls if r['preset']=='high_quality')
        near=[r for r in controls if abs(row['bytes']/r['bytes']-1)<=.05]
        best=min(near,key=lambda r:r['rgb_mse']) if near else None
        bracket=interpolate(controls,row['bytes'],'bytes','rgb_mse')
        comparisons.append(dict(image=row['image'],step=row['step'],bytes=row['bytes'],rgb_mse=row['rgb_mse'],frozen_file_change_percent=100*(row['bytes']/row['frozen_bytes']-1),hq_bytes=hq['bytes'],hq_mse=hq['rgb_mse'],hq_rate_change_percent=100*(row['bytes']/hq['bytes']-1),hq_mse_change_percent=100*(row['rgb_mse']/hq['rgb_mse']-1),strictly_dominates_hq=row['bytes']<=hq['bytes'] and row['rgb_mse']<hq['rgb_mse'],ordinary_dominated_by=[r['preset'] for r in controls if r['bytes']<=row['bytes'] and r['rgb_mse']<=row['rgb_mse']],best_within5pct=None if best is None else dict(preset=best['preset'],bytes=best['bytes'],rgb_mse=best['rgb_mse'],signed_rate_gap_percent=100*(row['bytes']/best['bytes']-1),mse_change_percent=100*(row['rgb_mse']/best['rgb_mse']-1)),ordinary_bracket=bracket,interpolated_mse_change_percent=100*(row['rgb_mse']/bracket['value']-1) if bracket else None))
    write_json('comparisons.json',dict(coefficient_gate=gates,coefficient_gate_passed=passed,ordinary_controls=ordinary,comparisons=comparisons,method='Exact matching-coefficient pixels first. Actual whole-file size changes, actual dominance and strongest measured ordinary point within 5% with signed gap. Sparse lowest-bracketing-chord estimates separate, no extrapolation. No image-specific quantizer tuning or routing.'))
    with (HERE/'coefficients.csv').open('w',encoding='utf-8',newline='') as handle:
        fields=list(coef['summary'][0]);writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(coef['summary'])
    with (HERE/'files.csv').open('w',encoding='utf-8',newline='') as handle:
        fields=['image','step','bytes','bpp','rgb_mse','base_bytes','enhancement_bytes','container_bytes','enhancement_fraction','sparse_raw_bytes','coefficient_count','nonzero_count','frozen_bytes','dense_normal_bytes','encode_seconds','decode_seconds','process_peak_working_set','stream_sha256','pixels_sha256']
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(files)
    sparse_rows=[r for r in coef['summary'] if r['variant']=='zero_runs']
    raw_table='\n'.join(f"| {r['image']} | {r['coefficient_count']:,} | {r['nonzero_count']:,} | {2*r['coefficient_count']:,} | {r['raw_bytes']:,} | {8*r['nonzero_count']:,} |" for r in sparse_rows)
    (HERE/'PBC_SPARSE_COEFFICIENT_ANALYSIS.md').write_text(f'''# Pass 21 sparse coefficient representation

The primary zero-run layout passes the frozen coefficient gate. It preserves every quantized coefficient while removing most dense zero bytes before conventional XZ. No alternative layout or compressor was needed.

## Preserved information and format

Start research/experiments at 53410c1bc4bef78c77e31bffb36c9a466c923f0d. Primary runtime remains pbc31_feature_histogram; all eight runtime sources and the model are checked against immutable Pass16 source. Pass17 and Pass20 sources/results, v0 decoder, presets, policy and release staging are unchanged.

Exact original order: decorrelated color component, DCT row frequency, DCT column frequency, block row, block column. Quantization, 8x8 DCT, fixed color matrix, base p10 and all coefficient values remain unchanged. This is storage restructuring, not another quantizer, multiscale split or discarded-frequency method.

Raw sparse prefix `<4sQQ>` is 20 bytes: magic R21V, uint64 total coefficient count N, uint64 nonzero count K. For each nonzero, unsigned LEB128 encodes the number of zero slots since the previous nonzero, followed by zigzag-LEB128 for its signed int16 value. Initial previous position is -1. Gaps determine positions without coordinates; trailing zeros follow from N. Counts are mandatory even for all-zero arrays. Zero-valued nonzero symbols and noncanonical/oversized varints are invalid.

Numba traverses the original five-dimensional array directly in the required order. Packing counts nonzeros and reserves at most 8*K temporary bytes, then copies only the used prefix. It does not materialize a second N*2 reordered byte array. No Python loop visits coefficient positions. Unpacking allocates one zero-filled int16 array and writes strictly increasing nonzero positions. A view restores the original five-dimensional block geometry for the unchanged Pass17 inverse transform.

| q128 case | Coefficient slots N | Nonzeros K | Dense raw bytes | Sparse raw bytes incl. prefix | Packing scratch upper bound |
| --- | ---: | ---: | ---: | ---: | ---: |
{raw_table}

Raw size reductions are not compression claims. The final XZ results and full pack/unpack costs are measured separately. Sparse scratch on Cat is about 2.36 MB versus 51.15 MB of dense ordered coefficient bytes; input coefficients still occupy N*2 bytes. The benchmark also retains a dense expected vector for verification outside timers. Its cumulative process high-water mark is not an isolated per-format peak. Actual full decoding still creates a dense coefficient array and large float transform/reconstruction buffers; this work is not removed by sparse storage.

## Reader bounds and correctness

Raw header counts must match container-derived N before allocation. N <=50,331,648; K<=N. Gaps use at most five bytes and unsigned32 range; signed zigzag values use at most three bytes and range 1..65535, mapping exactly to nonzero int16 [-32768,32767]. Positions cannot exceed N or repeat. The body must contain exactly K pairs with no trailing bytes.

XZ is read in two bounded stages: first the 20-byte prefix, then at most 8*K+1 body bytes. EOF, trailing/concatenated streams, exact body consumption and canonical varints are checked. A million-byte trailing expansion with zero declared nonzeros is rejected after the bounded read. Complete files additionally cap source pixels at 16,777,216, total input at 256 MiB, dimensions at uint16 range and coefficient count before decompression.

Six saved sparse files recover all frozen coefficients and RGB pixels exactly. Empty/all-zero arrays, long runs, first/last positions, signed extremes and dense random int16 pass. Four non-block-aligned repeated encodes match bytes and frozen pixels. Thirty-one malformed/resource/magic cases reject. Two new processes decode using only stored files with LearnedFiller.load forced to raise. No original image or model weight access is required by decoding.

The primary succeeds, so the conditional alternative layout and optional Zstandard comparison were not executed. There is one implemented sparse format and one conventional entropy setting. Ordinary decoding gains no new dependency; the research sparse decoder uses the existing Numba/SciPy stack and adds no package.
''',encoding='utf-8',newline='\n')
    entropy_table='\n'.join(f"| {r['image']} | {r['variant']} | {r['raw_bytes']:,} | {r['payload_bytes']:,} | {1000*r['packing_seconds']:.2f} | {1000*r['compression_seconds']:.2f} | {1000*r['decompression_seconds']:.2f} | {1000*r['unpacking_seconds']:.2f} |" for r in coef['summary'])
    gate_table='\n'.join(f"| {r['image']} | {r['enhancement_change_percent']:+.2f}% | {r['packing_compression_speedup']:.1f}x | {r['unpack_decompression_ratio']:.3f}x |" for r in gates)
    (HERE/'PBC_ENTROPY_COMPARISON.md').write_text(f'''# Pass 21 entropy comparison

Zero-run/zigzag + conventional XZ0 approaches the extreme-XZ size while making coefficient packing/compression 39.5-128.7x faster on these three q128 cases. Complete hybrid speedups are smaller and reported separately.

## Controlled coefficient measurement

Same saved coefficient values for all methods. Initial inputs are Googlewiki/Mario64/Cat q128; Pass20 saved NumPy arrays are verified against the corresponding frozen Pass17 enhancement before measurement. Source image/model/runtime identities, input file hashes, raw representation digests, exact coefficient counts and experiment-source hashes accompany the rows.

Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0, SciPy 1.17.1; local i9-12900H, pinned logical P-core12, numerical threads one. Kernels and each actual coefficient path are warmed before three rotated repetitions. Dense extreme is preset0+EXTREME; dense normal and sparse both use preset0. One XZ frame each. No entropy-level sweep or optional compressor comparison.

Every timed method starts from the same 5D int16 coefficients. Dense packing includes reorder/tobytes; sparse packing includes nonzero counting, direct-order run/value generation, prefix and output conversion. Decompression includes bounded sparse prefix/body framing. Unpack/scatter is separately timed. Every result is compared with the exact dense expected vector outside its timers; repeated payload hashes match. Cold JIT/cache startup is outside warmed timings.

Table medians are per-stage; median total times are recorded separately in coefficients.csv/JSON because a sum of stage medians need not equal the median of per-call totals. Packing/unpacking are not omitted from gate speed/ratio calculations.

| Image | Method | Raw bytes | Compressed bytes | Pack ms | Compress ms | Decompress ms | Unpack ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{entropy_table}

Gate comparisons include pack+compress and decompress+unpack:

| Image | Sparse payload change vs extreme | Total coefficient encode speedup | Total coefficient decode ratio |
| --- | ---: | ---: | ---: |
{gate_table}

Frozen gate: at least two cases <=10% payload increase, none >15%; >=5x faster total coefficient encoding on all three; no >2x coefficient decode regression. All three pass. These size differences are far smaller than Pass20 normal-dense's 33-42% payload increase. Mario64 improves payload size as well as runtime. Exact reconstruction holds for all representations.

This is not a claim that whole-image encoding is 40-129x faster. The base compressor, original-resolution residual/color/DCT, inverse transform and image reconstruction still run. Whole files preserve their original RD points with changed byte sizes; whole-call repetitions establish the actual speed benefit.

## Interpretation

XZ0 sees a roughly 81-593 KB stream of meaningful gaps/values rather than 4-51 MB of mostly zero int16 bytes. Explicit zero runs retain the long-range sparsity without requiring expensive extreme-mode matching. There is some cost to varints, count fields and mixed gap/value symbols, but measured final bytes remain close to or below the frozen result. This is measured behavior on the specified cases, not a universal sparse-coding guarantee.

No second format, custom arithmetic coder or compressor dependency is justified by this passing primary test. Process peak-working-set observations are cumulative, with verification buffers outside timers; they are not proof of a particular whole-codec peak-memory reduction. The direct packer's compact scratch requirement is explicit in the implementation.
''',encoding='utf-8',newline='\n')
    file_table='\n'.join(f"| {r['image']} | {r['step']} | {r['frozen_bytes']:,} | {r['dense_normal_bytes']:,} | {r['bytes']:,} | {r['rgb_mse']:.3f} | {r['enhancement_bytes']:,} |" for r in files)
    hq_table='\n'.join(f"| {r['image']} | {r['hq_bytes']:,} / {r['hq_mse']:.3f} | {r['bytes']:,} / {r['rgb_mse']:.3f} | {r['hq_rate_change_percent']:+.2f}% | {r['hq_mse_change_percent']:+.2f}% |" for r in comparisons)
    close_table='\n'.join(f"| {r['image']} | {r['best_within5pct']['preset']} | {r['best_within5pct']['bytes']:,} / {r['best_within5pct']['rgb_mse']:.3f} | {r['best_within5pct']['signed_rate_gap_percent']:+.2f}% | {r['best_within5pct']['mse_change_percent']:+.2f}% |" for r in comparisons if r['best_within5pct'])
    time_table=[]
    for name in ('googlewiki.png','eiffel.png'):
        records=[r for r in timings['summary'] if r['image']==name]
        values={r['variant']:r for r in records}
        time_table.append(f"| {name} | {values['ordinary']['encode_seconds']:.4f} / {values['ordinary']['decode_seconds']:.4f} | {values['dense_extreme']['encode_seconds']:.4f} / {values['dense_extreme']['decode_seconds']:.4f} | {values['dense_normal']['encode_seconds']:.4f} / {values['dense_normal']['decode_seconds']:.4f} | {values['sparse']['encode_seconds']:.4f} / {values['sparse']['decode_seconds']:.4f} |")
    profile_table='\n'.join(f"| {r['image']} | {1000*r['spans']['base_decode_to_float']:.2f} | {1000*r['spans']['enhancement_decompress']:.2f} | {1000*r['spans']['sparse_unpack']:.2f} | {1000*r['spans']['inverse_dct']:.2f} | {1000*(r['enhancement_reconstruction_seconds']-r['spans']['inverse_dct']):.2f} | {100*r['coefficient_recovery_seconds']/r['profile_total_seconds']:.1f}% |" for r in profile)
    other=json.loads((HERE/'other_controls.json').read_text())['rows']
    single_table=[]
    for control in other:
        row=next(r for r in files if r['image']==control['image'])
        single_table.append(f"| {row['image']} | {control['encode_seconds']:.3f} / {control['decode_seconds']:.3f} | {row['encode_seconds']:.3f} / {row['decode_seconds']:.3f} | {row['encode_seconds']/control['encode_seconds']:.2f}x | {row['decode_seconds']/control['decode_seconds']:.2f}x |")
    (HERE/'PBC_SPARSE_HYBRID_RESULTS.md').write_text(f'''# Pass 21 sparse hybrid completed files

Sparse storage retains all four successful Pass17 ordinary-HQ dominance points with identical RGB pixels and much faster encoding. It solves the extreme-XZ encode bottleneck on this bounded screen. Full decoding remains too expensive for a general-purpose replacement; Cat/Flower remain RD failures.

## Completed files and ordinary comparisons

P21S is a research container, not a production version. A 26-byte `<4sIIIIfBB>` header stores magic, original width/height, base/payload lengths, float32 quantization step, block size8 and sparse version1. The XZ-framed R21V coefficient stream stores total/nonzero counts and all gaps/signed values. Original image dimensions determine the full edge-padded coefficient geometry. The fixed color matrix, DCT norm, base p10 and final round/clip are unchanged from Pass17. P17D header was30 bytes; its tile/mode fields are not needed here.

Encoder freshly compresses the same ordinary Quality p10 base, independently decodes it, computes the original residual/color/DCT at the **original measured quantizer**, packs every coefficient and uses conventional XZ0. Decoder recovers the base and exact complete coefficient array, then calls the original inverse/reconstruction routine. No coefficient dropping, multiscale, base reallocation, model training or quantizer sweep.

Initial three complete files preserve the coefficient gate's size behavior and exact pixels, so the declared extension ran at Monet q64, Eiffel q32 and Flower q128. One complete deterministic encode/decode per case establishes RD; only the two promising repeated cases receive the timing protocol below. Frozen files/bases/control rows are reused and byte-checked; no expensive ordinary frontier is regenerated.

| Image | q step | Frozen P17 bytes | Dense fast XZ bytes | Sparse complete bytes | RGB MSE (all three identical) | Sparse enhancement bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{file_table}

All six sparse files match frozen quantized arrays and decoded pixel SHA exactly. The complete file includes base + compressed enhancement + 26-byte header; bpp uses original pixels. Raw sparse bytes, enhancement fraction, full call walls and cumulative memory observations are in files.csv. Four files shrink versus frozen P17; Googlewiki grows 1.00% total, Cat 3.48%. Flower shrinks 9.84% total but remains above ordinary rates.

Actual ordinary HQ comparisons, all prior matching-runtime controls:

| Image | Ordinary HQ bytes / MSE | Sparse bytes / MSE | Signed rate change | MSE change |
| --- | ---: | ---: | ---: | ---: |
{hq_table}

Monet, Googlewiki, Mario64 and Eiffel strictly dominate ordinary HQ. Cat and Flower exchange far more bytes for quality and are not RD wins. Their sparse enhancement payload changes are +4.31%/-13.46% versus frozen, so the storage method does not create a large new photographic penalty; it also does not solve the architecture's existing photo rate problem.

Strongest actually measured ordinary points within5% of sparse file size:

| Image | Ordinary point | Ordinary bytes / MSE | Signed sparse rate gap | Sparse MSE change |
| --- | --- | ---: | ---: | ---: |
{close_table}

Googlewiki and Eiffel have no such actual 5%-near ordinary control; do not call their HQ comparison equal-rate. Googlewiki also strictly dominates the available same-runtime 500k HQ point (214,902 bytes/MSE439.912). comparisons.json retains all ordinary controls, signed close matches and conservative bracket estimates separately; no extrapolation or continuous frontier claim. The original Pass17 controls at the same quantizer remain the cleanest test of the new storage method because quality is identical.

## Repeated full encode/decode calls

Three warmed rotated repetitions per Googlewiki q128 and Eiffel q32, one pinned CPU/numerical thread. Timers include fresh base encode, base decode inside encoder, full residual/DCT, sparse packing/entropy and final decoding. Image loading, hash checks, external scoring and disk I/O are outside timers. Dense-normal uses a scoped research hook restored afterward; frozen source is untouched. Streams and pixels match their recorded controls on every repetition. Ordinary HQ is a different actual RD operating point; frozen/fast/sparse share exactly the same p10 base and quantizer.

| Image | Ordinary HQ encode / decode s | Frozen P17 encode / decode s | Dense-fast encode / decode s | Sparse encode / decode s |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(time_table)}

Sparse full encoding is 6.73x faster than frozen on Googlewiki and 2.75x faster on Eiffel. Against ordinary HQ it takes about 0.61x/0.38x the encode time at the named lower-rate/better-quality points. These do not combine with earlier PBC3.0 or Pass16 speed percentages.

Full sparse decode medians are about 1.71x/1.45x ordinary HQ and broadly similar to frozen P17. Do not claim the coefficient microdecode gains transferred proportionally. Googlewiki sparse decode CV is16.35%; frozen Eiffel decode CV is12.95%, so small full-decode differences are uncertain. Other timing CVs and all raw repeats remain in timings.json. No mean across datasets/protocols is used.

Other four points are **one-pass** full observations against fresh byte-identical ordinary-HQ controls, not repeated speed evidence:

| Image | Ordinary HQ encode / decode s | Sparse encode / decode s | Encode ratio | Decode ratio |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(single_table)}

Cat/Flower remain expensive in original-resolution transform/reconstruction. The one-pass decode observations are around6x ordinary HQ. Their existing low-rate/photo failures are explicit, and no default routing rule is inferred from the six examples.

## Remaining decoder work

One warmed, disjoint stage profile per case; its sums are separate from complete-call medians and are not substituted for those medians. All actual stage walls are in decode_profile.json.

| Image | Base -> float ms | Decompress ms | Unpack ms | IDCT ms | Other reconstruction ms | Recovery share of profile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{profile_table}

Coefficient recovery is only about3-8% of the current profiled decoder. Base decode + dequant/copy + inverse DCT + color + final round/clip total about64/28/589 ms when coefficient recovery is excluded, for Googlewiki/Eiffel/Cat. These are observations of this unchanged implementation, not theoretical lower bounds for every possible decoder. Profile totals differ from full-call medians because measurement/cache/allocation conditions differ; no production latency is predicted by adding component medians.

Eliminating the remaining sparse recovery entirely cannot remove the several hundred milliseconds of large-photo reconstruction. Full original-size coefficient/float/output buffers still exist, even though packing scratch and compressed input are much smaller. Memory fields are cumulative process high-water marks with validation/control buffers and are not isolated per-variant peak comparisons. No major whole-codec memory improvement is claimed without isolation.

## Disposition

Preserve this passing sparse storage prototype for bounded R&D. It substantially reduces entropy overhead without the Pass20 33-42% payload penalty, needs no second sparse format and retains four actual RD successes. It is **not** a practical universal replacement: observed decode overhead misses the10-15% target, and large photos also miss encode/rate targets.

Activation was conditional on completed-file **and runtime** feasibility; the general-purpose runtime gate is not met, so no classifier/routing policy was tested. Source pixel count and downsampling ratio plausibly flag reconstruction cost, but they are not established RD predictors. No image-name switch, dual full-image encode or offline oracle is presented as deployable performance.

The strongest next question is whether nonzero-block-aware inverse reconstruction and fewer full-image float copies can reduce decoder cost while preserving these exact pixels. This is a proposed bounded decode feasibility study, not a new experiment launched here or a promise that entropy changes solve all residual coding. If it cannot approach ordinary latency, keep this as an opt-in research codec and change the general-purpose direction rather than tune more varint formats. Production/v0/model/defaults/staged ZIP are unchanged.
''',encoding='utf-8',newline='\n')
    (HERE/'PBC_PASS21_SUMMARY.md').write_text('''# Pass 21 summary

The primary zero-run/zigzag representation succeeds at the sparse-storage goal. It preserves every coefficient and frozen RGB pixel while approaching or improving extreme-XZ bytes with much lower entropy cost. No alternative layout, Zstandard, multiscale scheme, quantizer sweep or production integration.

1. Before entropy, the three q128 arrays shrink from3.995/5.972/51.151 MB to186,192/81,327/593,435 bytes including counts. Raw size alone is not the win; actual XZ0 sizes remain close to extreme.
2. Format: exact Pass17 frequency-plane order, unsigned zero gaps and signed zigzag LEB128,20-byte N/K header, one XZ0 stream. Small Numba loops directly traverse/scatter; no Python per-slot loop or dense reordered packing array.
3. Extreme / fast-dense / sparse payload bytes: Googlewiki79,288/109,544/80,656; Mario38,340/51,000/34,816; Cat193,688/275,048/202,032. Full tables and packing stages are in coefficients.csv.
4. Sparse payload changes +1.73%/-9.19%/+4.31%, versus fast-dense's33-42% penalty. All coefficient gate conditions pass.
5. Median pack+compress totals16.79/10.45/134.24 ms; decompress+unpack5.49/2.87/20.88 ms. Including all packing, encode is107.8/128.7/39.5x faster than extreme coefficient coding. These are coefficient, not whole-codec, factors.
6. Every quantized int16 coefficient round-trips exactly. Zero runs, endpoints, all-zero/empty arrays, signed extremes and resource bounds pass.
7. Six P21S files decode to exactly the frozen Pass17 RGB images. Four repeated odd-size fixtures match frozen pixels,31 malformed/resource/magic cases reject, two new processes decode files with model loading forbidden.
8. Four ordinary-HQ RD dominance points remain: Monet rate-26.70%/MSE-7.05%; Googlewiki-22.01%/-68.81%; Mario-16.57%/-46.91%; Eiffel-32.29%/-31.42%. Stronger ordinary close-rate and500k controls remain explicit; no universal frontier claim.
9. Cat/Flower remain RD-negative. Cat total bytes grow3.48% versus frozen; Flower shrinks9.84%, but their rates still substantially exceed ordinary HQ. Storage success does not repair the photographic architecture.
10. Three-repeat complete encodes: Googlewiki sparse0.3738s vs frozen2.5143s and HQ0.6135s; Eiffel0.2206s vs frozen0.6068s and HQ0.5851s. Full speedups6.73x/2.75x versus frozen. Other four full calls are one-pass feasibility observations.
11. Repeated sparse decodes0.0829/0.0453s vs HQ0.0485/0.0313s: about1.71x/1.45x. Full decode remains broadly similar to frozen; small differences have appreciable variation. Cat/Flower one-pass decode is around6x ordinary.
12. This is a useful storage improvement, not a practical general-purpose default yet. No claim of major total-memory reduction; decoder still materializes dense transform/output buffers.
13. Main remaining bottleneck is original-resolution inverse transform, color/float copies and final image work. Coefficient recovery is only3-8% of the stage profiles; further entropy tuning cannot erase that work.
14. Continue only with a bounded exact-pixel decoder feasibility question (nonzero-block-aware work/fewer copies). No broad confirmation or activation policy is justified until runtime improves. If it fails, keep the hybrid opt-in or change direction; do not rescue through more sparse-format tweaks.
15. Focused source, tests, compact evidence and reports are committed to research/experiments. Final verified GitHub HEAD/URL is supplied in the chat. Practical pbc31_feature_histogram, frozen Pass17/Pass20, v0, model and staged ZIP remain unchanged.

Reproduce with Python3.11 and requirements.txt, restored hpt_data, the saved Pass17 streams at the six fixed quantizers, and Pass20's three saved q128 NumPy arrays. Frozen hashes/readback are checked before use; generated inputs/outputs are not committed.

```
research_pass16/.venv/Scripts/python.exe research_pass21/compare_coefficients.py
research_pass16/.venv/Scripts/python.exe research_pass21/screen.py
research_pass16/.venv/Scripts/python.exe research_pass21/screen.py --extend
research_pass16/.venv/Scripts/python.exe research_pass21/time_promising.py
research_pass16/.venv/Scripts/python.exe research_pass21/profile_decode.py
research_pass16/.venv/Scripts/python.exe research_pass21/time_other_controls.py
research_pass16/.venv/Scripts/python.exe research_pass21/validate.py
research_pass16/.venv/Scripts/python.exe research_pass21/report.py
```

The extension is conditional on the fixed gate in PLAN.md; it passed here. Only Googlewiki/Eiffel receive repeated full timing. Input/source/model/plan/file hashes, exact configs, resource limits and all compact rows accompany these reports. No full25x4 benchmark, HF deployment, release/tag, protected-branch change or release rebuild. Staged ZIP SHA256 remains08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591.
''',encoding='utf-8',newline='\n')
    result=meta()
    result['base_config']=dataclasses.asdict(PBC3Config.quality(patch_count=10,learned_filler_model_path=str(ROOT/'patch_policy.npz')))
    result['ordinary_timing_config']=dataclasses.asdict(PBC3Config.high_quality())
    result['format']={'container':'P21S','header_bytes':codec.HEADER.size,'coefficient_magic':'R21V','coefficient_header_bytes':zero_runs.HEADER.size,'order':'c,u,v,block-row,block-column','entropy':'XZ/LZMA2 preset0','max_pixels':codec.MAX_PIXELS,'max_coefficients':zero_runs.MAX_COEFFICIENTS,'max_file_bytes':codec.MAX_FILE_BYTES}
    result['frozen_files']={name:{'step':step,'sha256':sha(frozen(name,step)[0])} for name,step in INITIAL+EXTRA}
    result['coefficient_input_npy_sha256']={name:sha((ROOT/'research_pass20/artifacts'/(name+'.q128.npy')).read_bytes()) for name,step in INITIAL}
    result['evidence_sha256']={p.name:sha(p.read_bytes()) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.json','.csv') and p.name!='provenance.json'}
    result['staged_zip_sha256_unchanged']='08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591'
    write_json('provenance.json',result)
    print('coefficient gate passed;',sum(r['strictly_dominates_hq'] for r in comparisons),'ordinary-HQ RD wins; exact coefficients/pixels on six files; general runtime gate not met',flush=True)


if __name__=='__main__':main()
