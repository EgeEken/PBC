import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CHOSEN={'cat.jpg':128,'flower.jpg':128,'monet.jpg':64,'googlewiki.png':128,'mario64.png':128,'eiffel.png':32}


def interpolate(points,value,x,y):
    candidates=[]
    for a in points:
        for b in points:
            if a[x]<=value<=b[x] and a[x]<b[x]:
                weight=(value-a[x])/(b[x]-a[x])
                result=a[y]+weight*(b[y]-a[y])
                candidates.append({'value':result,'a':a.get('preset',a.get('step')),'b':b.get('preset',b.get('step')),'x_a':a[x],'x_b':b[x],'y_a':a[y],'y_b':b[y],'weight':weight})
    exact=[p for p in points if p[x]==value]
    for p in exact: candidates.append({'value':p[y],'a':p.get('preset',p.get('step')),'b':p.get('preset',p.get('step')),'weight':0})
    return min(candidates,key=lambda r:r['value']) if candidates else None


def main():
    inputs={name:json.loads((HERE/name).read_text()) for name in ('baseline.json','hybrid_results.json','tiles_results.json','extra_controls.json','targeted_controls.json','timings.json','validation.json')}
    baseline=inputs['baseline.json']['rows']; full=inputs['hybrid_results.json']['rows']; tiles=inputs['tiles_results.json']['rows']
    controls=baseline+inputs['extra_controls.json']['rows']+inputs['targeted_controls.json']['rows']
    comparisons=[]
    for row in full:
        ordinary=[r for r in controls if r['image']==row['image']]
        bracket=interpolate(ordinary,row['bytes'],'bytes','rgb_mse')
        close=[r for r in ordinary if abs(row['bytes']/r['bytes']-1)<=.05]
        best=min(close,key=lambda r:r['rgb_mse']) if close else None
        hq=next(r for r in baseline if r['image']==row['image'] and r['preset']=='high_quality')
        comparisons.append({'image':row['image'],'step':row['step'],'bytes':row['bytes'],'rgb_mse':row['rgb_mse'],'hq_rate_change_percent':100*(row['bytes']/hq['bytes']-1),'hq_mse_change_percent':100*(row['rgb_mse']/hq['rgb_mse']-1),'strictly_dominates_hq':row['bytes']<=hq['bytes'] and row['rgb_mse']<hq['rgb_mse'],'dominated_by_any_ordinary':any(r['bytes']<=row['bytes'] and r['rgb_mse']<=row['rgb_mse'] for r in ordinary),'ordinary_rate_bracket':bracket,'interpolated_matched_rate_mse_change_percent':100*(row['rgb_mse']/bracket['value']-1) if bracket else None,'best_measured_within5pct':None if best is None else {'preset':best['preset'],'bytes':best['bytes'],'rgb_mse':best['rgb_mse'],'signed_rate_gap_percent':100*(row['bytes']/best['bytes']-1),'mse_change_percent':100*(row['rgb_mse']/best['rgb_mse']-1)}})
    matched_quality=[]
    for row in baseline:
        candidates=[r for r in full if r['image']==row['image']]
        bracket=interpolate(candidates,row['rgb_mse'],'rgb_mse','bytes')
        matched_quality.append({'image':row['image'],'preset':row['preset'],'ordinary_bytes':row['bytes'],'ordinary_mse':row['rgb_mse'],'hybrid_mse_bracket':bracket,'interpolated_rate_change_percent':100*(bracket['value']/row['bytes']-1) if bracket else None})
    result={'comparison_method':'Actual files first. Strongest measured ordinary control within 5% is shown with signed rate gap. Matched-rate interpolation uses the minimum MSE over all valid measured ordinary brackets (a conservative lower-chord envelope); no extrapolation. Bracket endpoints/weights are retained. This is an estimate, not an achievable interpolated file.','comparisons':comparisons,'matched_quality':matched_quality}
    (HERE/'rd_comparisons.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    fieldnames=['image','variant','step','target_preset','bytes','bpp','rgb_mse','base_bytes','enhancement_bytes','container_bytes','enhancement_rate_fraction','encode_seconds','decode_seconds','residual_reconstruction_mse','stream_sha256','pixels_sha256']
    with (HERE/'measurements.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fieldnames,extrasaction='ignore',lineterminator='\n'); writer.writeheader()
        for row in controls: writer.writerow({**row,'variant':row['preset']})
        for row in full+tiles: writer.writerow({**row,'variant':row['architecture']})
    lines=['# Pass 17 completed-file RD results','','The full-image DCT sidecar is promising on four of six images, but is not a production promotion. The current development codec remains pbc31_feature_histogram. Cat and Flower fail the studied rate range. Independent tile selection is rejected in its tested form. No alternative architecture or full-set confirmation was run.','','## Actual completed files','','Every hybrid uses an ordinary Quality base with patch_count=10. The table selects the best-MSE tested full-DCT point not exceeding ordinary HQ bytes; where none fits (Cat/Flower), it shows the smallest tested hybrid and marks the rate excess. Quantization choices are per-image observed operating points from the fixed 32/64/128 screen, not a learned selector or one universal preset. All byte counts include base, XZ-compressed signed coefficients and the 30-byte container.','','| Image | DCT step | Ordinary HQ bytes / MSE | Hybrid bytes / MSE | Rate change | MSE change |','| --- | ---: | ---: | ---: | ---: | ---: |']
    selected=[]
    for name,step in CHOSEN.items():
        row=next(r for r in full if r['image']==name and r['step']==step)
        cmp=next(r for r in comparisons if r['image']==name and r['step']==step)
        hq=next(r for r in baseline if r['image']==name and r['preset']=='high_quality')
        selected.append(row)
        lines.append(f'| {name} | {step} | {hq["bytes"]:,} / {hq["rgb_mse"]:.3f} | {row["bytes"]:,} / {row["rgb_mse"]:.3f} | {cmp["hq_rate_change_percent"]:+.2f}% | {cmp["hq_mse_change_percent"]:+.2f}% |')
    lines+=['','Monet, Googlewiki, Mario64 and Eiffel have measured strict dominance over HQ: fewer final bytes and lower RGB MSE. This is evidence against these ordinary points, not proof against every possible PBC configuration. No measured ordinary point dominates those four selected hybrids. Cat/Flower trade much more bitrate for quality and are not counted as RD wins.','','## Near-rate and bracketed comparisons','','Ordinary controls comprise default Quality/HQ, one fresh HQ p10 point per image, the embedded Quality p10 base, and one additional size-predicted HQ budget for each promising image (Monet p8, Googlewiki p13, Mario64 p13, Eiffel p5). Those four budgets were chosen once from the p10/p20 byte sizes, without quality-based tuning or iterative retries. Requested patch budgets are not nested stream prefixes and rates are not assumed monotone. All measured controls, including dominated ones, remain in the CSV.','','Strongest actually measured ordinary comparator within 5% final bytes for three useful hybrid points:','','| Image / DCT step | Ordinary point | Ordinary bytes / MSE | Hybrid bytes / MSE | Signed rate gap | MSE change |','| --- | --- | ---: | ---: | ---: | ---: |']
    for name,step in (('monet.jpg',64),('mario64.png',128),('eiffel.png',64)):
        r=next(r for r in comparisons if r['image']==name and r['step']==step); a=r['best_measured_within5pct']
        lines.append(f'| {name} / {step} | {a["preset"]} | {a["bytes"]:,} / {a["rgb_mse"]:.3f} | {r["bytes"]:,} / {r["rgb_mse"]:.3f} | {a["signed_rate_gap_percent"]:+.2f}% | {a["mse_change_percent"]:+.2f}% |')
    lines+=['','Googlewiki has no comparator within 5%; its q128 file is 7.34% smaller than the measured HQ p13 file (147,416 bytes, MSE 727.003), with MSE 222.834. Its direct dominance and the measured bracket are reported separately rather than calling the files equal-rate.','','Conservative bracketed estimates at the four selected hybrid rates:','','| Image | Ordinary bracket | Estimated ordinary MSE | Hybrid MSE | Matched-rate MSE change |','| --- | --- | ---: | ---: | ---: |']
    for row in selected:
        cmp=next(r for r in comparisons if r['image']==row['image'] and r['step']==row['step'])
        bracket=cmp['ordinary_rate_bracket']
        if bracket:
            lines.append(f'| {row["image"]} | {bracket["a"]} to {bracket["b"]} | {bracket["value"]:.3f} | {row["rgb_mse"]:.3f} | {cmp["interpolated_matched_rate_mse_change_percent"]:+.2f}% |')
    lines+=['','The estimate takes the lowest interpolated ordinary MSE among all valid measured bracketing pairs, avoiding an artificially weaker curve caused by adding an inefficient intermediate control. It never extrapolates; endpoints/weights and all 18 candidate comparisons are in rd_comparisons.json. The brackets are sparse and not independent confirmation of a continuous frontier.','','At ordinary HQ MSE, valid measured hybrid quality brackets exist for Monet and Eiffel only:','']
    for r in matched_quality:
        if r['preset']=='high_quality' and r['hybrid_mse_bracket']:
            b=r['hybrid_mse_bracket']; lines.append(f'- {r["image"]}: interpolated rate {r["interpolated_rate_change_percent"]:+.2f}%, between DCT steps {b["a"]} and {b["b"]}.')
    lines+=['','Googlewiki/Mario64 already use fewer bytes at better-than-HQ quality, but their tested hybrid qualities do not bracket HQ MSE. No equal-MSE interpolation or extrapolated savings is claimed for them.','','## File accounting and correctness','','Format P17D stores width/height, base length, enhancement length, float32 quantization step, block size, mode and tile count in a fixed 30-byte header. Full mode stores a complete ordinary v0 base followed by one XZ/LZMA2 stream. Coefficients are signed little-endian int16, ordered channel/frequency/block-row/block-column. A fixed orthonormal color transform and 8x8 type-II DCT/IDCT are part of the format definition. The decoder multiplies the stored integers by the stored step, inverts the transform, adds the correction to the decoded base, then rounds/clips final RGB. Residuals are not clipped or wrapped.','','| Image / step | Base bytes | Enhancement bytes | Header bytes | Enhancement share | Residual reconstruction MSE | Final RGB MSE |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in selected:
        lines.append(f'| {r["image"]} / {r["step"]} | {r["base_bytes"]:,} | {r["enhancement_bytes"]:,} | {r["container_bytes"]} | {100*r["enhancement_rate_fraction"]:.2f}% | {r["residual_reconstruction_mse"]:.3f} | {r["rgb_mse"]:.3f} |')
    tr=inputs['hybrid_results.json']['transform_roundtrip']
    lines+=['',f'Unquantized float32 transform round-trip error on signed residuals: max absolute {tr["max_abs"]:.9f}, MSE {tr["mse"]:.3g}. Quantization and final pixel clipping account for the much larger measured residual/final distortions; neither is hidden. All 30 actual full/tile files passed independent size, file-SHA, decoded-pixel-SHA and RGB-MSE checks. Signed int16 payload values spanning -32,000 to +32,000 round-trip exactly. Non-block-aligned images, repeated deterministic encodes, truncated files and research-magic rejection by the ordinary decoder passed. Fresh processes decoded Googlewiki/Eiffel using only the saved file, with policy loading forced to raise. The existing v0 decoder/source remains unchanged.','','## Content-adaptive layer: negative result','','A second mode uses disjoint 32x32 tiles, the same 8x8 transform and fixed step 64. Each tile is ranked by actual quantized/rounded/clipped RGB SSE reduction divided by its actual compressed bytes plus its 8-byte index/length record. Positive-gain tiles are greedily selected under the ordinary Quality or HQ complete-file budget. This is not ranking by residual SSE alone. Tile count is in the fixed header; every tile has its own XZ stream.','','| Image | Quality budget MSE change | HQ budget MSE change | HQ chosen tiles | HQ metadata bytes |','| --- | ---: | ---: | ---: | ---: |']
    for name in CHOSEN:
        q=next(r for r in tiles if r['image']==name and r['target_preset']=='quality'); hq=next(r for r in tiles if r['image']==name and r['target_preset']=='high_quality')
        lines.append(f'| {name} | {q["mse_change_percent"]:+.2f}% | {hq["mse_change_percent"]:+.2f}% | {hq["selected_tiles"]} | {hq["container_bytes"]:,} |')
    lines+=['','All tile files fit their target byte budgets; signed gaps remain in tiles_results.json. Only Googlewiki improved (2/12 points); the other 10 regress. Its HQ tile file is itself dominated by the smaller, lower-MSE full q128 file. This prototype loses cross-tile entropy context and pays independent XZ framing plus index records. Metadata alone is not established as the cause: the tile records, entropy framing/context and changed coefficient selection all contribute. A finer quantizer on a small tile subset also leaves much of the weaker p10 base uncorrected. Neither the rejection of this rule nor the two large-photo failures prove residuals fundamentally unpredictable. No tile-quantizer rescue sweep or alternative architecture followed.','','## Runtime','','Initial per-file encode/decode walls are in measurements.csv. Encode timers include base compression, base decoding, residual/color/DCT construction, quantization, entropy and container construction. Tile timers also include scoring/compressing every considered tile and selection. Decode timers include base decoding plus stored enhancement reconstruction. Image loading, external scoring, hashing, file I/O and reporting are excluded. These are CPU timings, one numerical thread, one pinned P-core.','','Only two promising pairs received three warmed interleaved repetitions. They compare the actual different operating points named below, not equal-byte configurations.','','| Pair | Ordinary encode / decode (s) | Hybrid encode / decode (s) | Encode overhead | Decode overhead |','| --- | ---: | ---: | ---: | ---: |']
    for name,r in inputs['timings.json']['summary'].items():
        a,b=r['ordinary'],r['hybrid']; label='HQ versus hybrid q128' if name=='googlewiki.png' else 'Quality versus hybrid q64'
        lines.append(f'| {name}: {label} | {a["encode_seconds"]:.4f} / {a["decode_seconds"]:.4f} | {b["encode_seconds"]:.4f} / {b["decode_seconds"]:.4f} | {100*(b["encode_seconds"]/a["encode_seconds"]-1):+.1f}% | {100*(b["decode_seconds"]/a["decode_seconds"]-1):+.1f}% |')
    lines+=['','The Googlewiki gain costs about 4x encoding time; this is not an output-preserving speed optimization. The added decoder work is a full-resolution inverse transform and extra buffers. Large-photo full q128 encodes took about 5.0 to 7.6 seconds in the one-pass feasibility screen and decoded in about 0.8 seconds, with no established RD win at the measured rates. No production-performance claim is made. Repeated timing streams reproduced the original file hashes.','','## Disposition and reproduction','','Freeze the full-image prototype as pbc17_dct_detail_v1 for further R&D; reject the tested independently compressed tile rule. This requires a research container/extension, not ordinary v0. Leave pbc31_feature_histogram and the staged ZIP untouched. No version/release decision was made.','','Recommended next benchmark, subject to the user: 8 to 10 additional images spanning text, game renders, art, medium photographs and 8MP photographs; Quality/HQ rate region; the frozen three quantization steps and base p10; at most two size-selected ordinary budget controls per image. Include a few 500k ordinary-resolution controls in that later study so the claim is tested against stronger ordinary configurations, not only 250k budgets. This is a recommendation, not an executed benchmark. The strongest follow-up mechanism is joint base/detail allocation with a multiscale residual representation for large photographs: their lost information is often low/mid-frequency on an 8x8 original-resolution block, and a dense original-size layer is too costly at their very low source-pixel bitrates.','','Reproduce with Python 3.11.16 and the pinned research requirements, restored hpt_data inputs, and no concurrent benchmark workload:', '', '```', 'python research_pass17/analyze_residuals.py', 'python research_pass17/screen_hybrid.py', 'python research_pass17/screen_tiles.py', 'python research_pass17/rate_controls.py', 'python research_pass17/rate_controls.py --targeted', 'python research_pass17/time_promising.py', 'python research_pass17/validate.py', 'python research_pass17/report.py', '```','','SciPy routine definitions: [dctn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html), [idctn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idctn.html). Actual numerical dependency versions and source/model/input hashes are recorded in each measurement JSON. Generated files stay under ignored artifacts/.']
    (HERE/'PBC_RD_PROTOTYPE_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    hypo=HERE/'PBC_ARCHITECTURAL_RD_HYPOTHESIS.md'
    text=hypo.read_text(encoding='utf-8').split('\n## Residual evidence')[0]
    text+='\n## Residual evidence\n\nAll values use independently decoded original-size RGB. DCT high-frequency means u+v>=4 in an 8x8 block; remaining energy is not necessarily recoverable cheaply. The source working-resolution round trip is a diagnostic, not a distortion floor or an additive causal share; baseline.json retains the exact cross term. Spatial concentration counts source pixels only (zero padding of partial tiles).\n\n| Image | Q / HQ MSE | Working-resolution round-trip MSE | HQ high-frequency energy | HQ error on strongest 10% edge pixels | HQ error in highest-SSE 10% tiles |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for name in CHOSEN:
        q=next(r for r in baseline if r['image']==name and r['preset']=='quality'); hq=next(r for r in baseline if r['image']==name and r['preset']=='high_quality'); a=hq['analysis']
        text+=f'| {name} | {q["rgb_mse"]:.2f} / {hq["rgb_mse"]:.2f} | {a["source_roundtrip_mse"]:.2f} | {100*a["dct_high_uplusv_ge4_fraction"]:.1f}% | {100*a["edge_sse_fraction"]:.1f}% | {100*a["top10pct_tiles_error_fraction"]:.1f}% |\n'
    text+='\nGooglewiki/Mario64 retain much high-frequency and edge error. Flower has localized errors and much lower smooth-region MSE, while Cat has more distributed loss. The full smooth-region, SSE and spectral metrics are retained per preset in baseline.json. Cat loses detail at frequencies that can look low-frequency within an original-size 8x8 block because its working image is much smaller. Eiffel is already below the 250k cap: its transform-layer gains therefore also expose correction-grid representation inefficiency, not only information lost by resizing. Unlike earlier transform-patch/AVIF experiments, this prototype codes the actual decoded original-resolution RGB residual over a reduced-bit base and measures complete matched-range files.\n'
    hypo.write_text(text,encoding='utf-8',newline='\n')
    stagesha=hashlib.sha256((Path(r'C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC')/'PBC3_1_Release_Staging/PBC3_1_Release.zip').read_bytes()).hexdigest()
    assert stagesha=='08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591'
    summary='''# PBC Pass 17 summary

The full-image signed DCT residual layer demonstrates useful completed-file RD gains on four of six images. It uses a reduced Quality p10 base, a fixed color transform, 8x8 DCT, three predeclared quantizers and a separately decodable P17D container. Production remains pbc31_feature_histogram; this is an R&D prototype, not a release candidate.

Against measured ordinary HQ, selected hybrid points reduce both bytes and RGB MSE: Monet (q64), Googlewiki (q128), Mario64 (q128), Eiffel (q32). Three actual controls within 5% rate separately show large MSE gains, with signed rate differences retained. Googlewiki has strong measured dominance but no 5%-near ordinary point. Sparse conservative bracket estimates are reported separately; there is no extrapolation or universal frontier claim. Measured hybrid quality brackets support matched-quality interpolation only for Monet and Eiffel; those estimates remain separate from actual files.

Cat and Flower remain too expensive at full-image layer rates. Independent tile coding at fixed total bytes worsened 10/12 points; its HQ Googlewiki result is dominated by the stronger full-image point. No alternative architecture or rescue sweep was needed because the primary architecture already warrants broader research.

All 30 completed experimental files pass actual decoding, SHA/size/MSE checks, signed coefficient readback, boundary/determinism checks and fresh-process file-only decoding with policy loading forbidden. The ordinary v0 decoder is unchanged and rejects P17D. Full mode adds a 30-byte header plus its actual entropy-coded coefficients; tile mode also adds 8 bytes per selected tile, with XZ framing counted inside payload bytes.

The RD gain costs compute: repeated Googlewiki hybrid/ordinary HQ encode is about 4.0x, decode 1.56x; Eiffel q64/ordinary Quality encode is 1.76x, decode 1.53x. Complete calls include all base/detail work and internal base MSE. Runtime is Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0 and SciPy 1.17.1, one i9-12900H P-core, one numerical thread.

The frozen research prototype is pbc17_dct_detail_v1. Recommend a user-approved 8 to 10-image independent follow-up with actual close-rate ordinary controls, including a few larger-working-resolution controls. Do not automatically run the 100-case matrix. Joint base/detail allocation and multiscale detail coding for large photographs are the strongest next hypothesis, based on their rate failure and distributed low/mid-frequency residuals.

No production source/model/default, HF deployment, protected branch, release tag or staging artifact changed. Staged ZIP SHA-256 remains 08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591. Reports and compact per-file rows are in this directory; generated files are ignored. The final research commit is supplied in the chat handoff.
'''
    (HERE/'PBC_PASS17_SUMMARY.md').write_text(summary,encoding='utf-8',newline='\n')
    manifest={'control_commit':inputs['baseline.json']['provenance']['control_commit'],'prototype_id':'pbc17_dct_detail_v1','frozen_measured_source_sha256':inputs['timings.json']['frozen_hybrid_sha256'],'frozen_measured_source_lf_sha256':inputs['timings.json']['frozen_hybrid_lf_sha256'],'production_id':'pbc31_feature_histogram','staged_zip_sha256_unchanged':stagesha,'source_model_runtime_inputs':inputs['baseline.json']['provenance'],'experimental_source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('hybrid.py','analyze_residuals.py','screen_hybrid.py','screen_tiles.py','rate_controls.py','time_promising.py','validate.py','report.py')},'evidence_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in inputs},'completed_experimental_files':len(full)+len(tiles),'ordinary_default_encodes':12,'additional_ordinary_encodes':10,'full_hybrid_encodes':18,'tile_hybrid_encodes':12,'timing_measured_calls':12,'no_full_confirmation':True,'no_alternative_architecture':True}
    (HERE/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Selected comparisons:')
    for row in selected:
        c=next(r for r in comparisons if r['image']==row['image'] and r['step']==row['step'])
        print(row['image'],row['step'],c['hq_rate_change_percent'],c['hq_mse_change_percent'],c['interpolated_matched_rate_mse_change_percent'])

if __name__=='__main__': main()
