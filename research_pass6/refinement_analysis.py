import json
from pathlib import Path
from statistics import mean,stdev

HERE=Path(__file__).resolve().parent


def main():
    screen=json.loads((HERE/'refinement_screen.json').read_text())
    rows=screen['rows']
    raw=[json.loads(line) for line in (HERE/'refinement_raw.jsonl').read_text().splitlines()]
    assert len(raw)==108 and all(r['preview_equal'] for r in raw)
    sources=('integrated','init','iter')
    averages={v:{k:mean(r[k] for r in rows if r['source']==v) for k in ('mse','bpp','encode_seconds','decode_seconds')} for v in sources}
    comparisons=[]
    summary={}
    for source in ('init','iter'):
        group=[r for r in rows if r['source']==source]
        changes={k:100*(averages[source][k]/averages['integrated'][k]-1) for k in averages[source]}
        close=[]
        for r in group:
            b=next(x for x in rows if (x['image'],x['preset'],x['source'])==(r['image'],r['preset'],'integrated'))
            comparison=dict(image=r['image'],preset=r['preset'],variant=source,
                change={k:100*(r[k]/b[k]-1) for k in averages[source]},
                first_geometry_change=r['first_geometry_change'],first_quantized_change=r['first_quantized_change'],
                baseline_patches=b['patches'],candidate_patches=r['patches'])
            comparisons.append(comparison)
            if abs(r['bpp']/b['bpp']-1)<=.05:
                close.append(comparison)
        operations={k:sum(r['stats'].get(k,0) for r in group) for k in ('attempted','accepted','accepted_index_changes','extra_projections','extra_reconstructions','local_sse_reduction','evaluated_original_sse')}
        cv={k:mean(100*stdev(x[k] if k=='decode_seconds' else x['stats'].get('encode_scope_seconds',x[k]) for x in raw if (x['image'],x['preset'],x['source'])==(r['image'],r['preset'],source))/mean(x[k] if k=='decode_seconds' else x['stats'].get('encode_scope_seconds',x[k]) for x in raw if (x['image'],x['preset'],x['source'])==(r['image'],r['preset'],source)) for r in group) for k in ('encode_seconds','decode_seconds')}
        summary[source]=dict(aggregate_change=changes,operations=operations,cv_percent=cv,
            near_5pct_cases=len(close),near_5pct_mse_wins=sum(x['change']['mse']<0 for x in close),
            near_5pct_mean_mse_change=mean(x['change']['mse'] for x in close),
            strict_rd_wins=screen['summary'][source]['rd_wins'],promoted=False)
    decision=dict(promoted=[],full_main_confirmation=False,holdout_evaluated=False,quality500k35_evaluated=False,
        combined_variant_evaluated=False,feedback_strengths_tested=[.5],
        reason='Final RGB mean MSE improves only 0.087%/0.157%. Initialization costs 10.75% mean encode; iterative gains are inconsistent and its best quality change costs 28% bitrate. Stop both variants after the six-image screen.',
        averages=averages,summary=summary,comparisons=comparisons)
    (HERE/'refinement_decision.json').write_text(json.dumps(decision,indent=2)+'\n',newline='\n')
    lines=['# Decoder-aware grid refinement','',
        'Neither screened variant is promoted. The feedback changes real indices and improves local visible working-space SSE, but does not establish a meaningful final rate-distortion improvement. No full main/holdout confirmation, 500k/35 follow-up, combined method, second strength or further optimization was run. Nothing from this experiment is enabled or deployed on HF.','',
        'One forward correction is applied after the ordinary patch has been selected. Its geometry, cell size, grid dimensions, palette mask/bounds/precision and syntax stay fixed. Reconstruct delta R(Q) using the actual signed bicubic/rounded decoder path, calculate e = target − clip(canvas + R(Q), 0, 255), and project e to the stored grid. Propose Q′ by requantizing values(Q) + 0.5 × projected(e) onto the same generated palette. Reconstruct Q′ with the same integer resampler and accept only strictly lower clipped local SSE. int16 additive canvas semantics are retained. LZMA is never called in candidate refinement.','',
        'B-init refines the three selected full-image RGB initialization grids. B-iter tries at most the first three selected iterative patches with cell size >1 and nominal index payload ≥8192 bits. This uses only encoding-time information; geometry, model, proposal count, weights, RNG seed and configuration are unchanged. Both variants allow the ordinary sequential trajectory and stop decision to evolve afterward. The scoped class hooks are restricted to isolated research processes.','',
        'Screen: cat.jpg, flower.jpg, monet.jpg, mario64.png, eiffel.png, googlewiki.png; Quality and High Quality, default 250k working cap; optimized current control, B-init, B-iter; three warmed interleaved repeats (108 timed calls, six warmups). Full candidate encode walls include scope installation/restoration, projections, quantization, extra reconstruction/scoring and all later trajectory work. Independent original-size RGB decode/MSE and complete post-LZMA bytes determine final outcomes. All 108 previews equal normal v0 decoder output and each configuration repeats deterministically.','',
        '| Variant | Mean RGB MSE | Mean bpp | Encode s | Decode s | MSE change | bpp change | Encode change |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for v in sources:
        a=averages[v]
        c={k:100*(a[k]/averages['integrated'][k]-1) for k in a}
        lines.append(f"| {v} | {a['mse']:.6f} | {a['bpp']:.7f} | {a['encode_seconds']:.6f} | {a['decode_seconds']:.6f} | {c['mse']:+.3f}% | {c['bpp']:+.3f}% | {c['encode_seconds']:+.2f}% |")
    lines+=['','MSE/bpp are equal-case arithmetic means; times average the 12 case medians. These fixed-preset aggregates are not matched-rate frontier measurements. Averaging relative per-case MSE changes instead produces −2.35% for init and −0.91% for iter; those different weightings must not replace the near-neutral mean RGB result. Small mean rate savings also do not prove per-image joint dominance.','',
        '| Mechanism | Attempts / accepts | Accepted indices changed | Extra grid projections / reconstructions | Local SSE reduction | Strict RD wins |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for v,s in summary.items():
        o=s['operations']
        lines.append(f"| {v} | {o['attempted']} / {o['accepted']} | {o['accepted_index_changes']} | {o['extra_projections']} / {o['extra_reconstructions']} | {o['local_sse_reduction']} | {s['strict_rd_wins']}/12 |")
    lines+=['','Counts describe one deterministic encode for each of the 12 configurations, not all repeated calls. Local SSE reductions sum accepted same-state comparisons, not a global RGB improvement or a decomposition of final loss. The nominal bit cost of each considered patch is unchanged, but later geometry/counts and entropy context change complete raw and LZMA sizes.','',
        '| Image | Preset | Variant | MSE change | bpp change | Encode change | First changed geometry patch |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in comparisons:
        c=r['change']
        lines.append(f"| {r['image']} | {r['preset']} | {r['variant']} | {c['mse']:+.2f}% | {c['bpp']:+.2f}% | {c['encode_seconds']:+.2f}% | {r['first_geometry_change']} |")
    lines+=['','First geometry position is zero-based in the complete stream; None means no differing common-prefix geometry, not necessarily identical coefficients or patch count. Compact output retains trajectory hashes and first differences; the complete measured trajectories remain locally in ignored refinement_raw.jsonl.','',
        'B-init accepts all 36 corrections, improving summed local SSE by 49,438,754, yet mean original-size RGB MSE improves only 0.087%. Eiffel HQ is a useful individual point (about −19.67% MSE and −6.29% bpp), while Mario Quality worsens MSE 3.30%, flower HQ 2.67%, and googlewiki worsens at both presets. Cat HQ spends about 10.05% more bpp for only 0.30% lower MSE. Quality cat/flower encode increases about 38%/42%; patch counts rise 20→53 and 16→53 including three initialization patches. Their ordinary trajectories now fill the unchanged budget instead of stopping early. Thus the penalty includes real additional search/patch work, not only the three corrections.','',
        'B-iter accepts 27 corrections and changes 9,127 indices. Googlewiki has no eligible coarser-grid patches, so both presets are unchanged. Mean RGB MSE improves only 0.157%; flower HQ improves about 10.97% MSE at 28.27% more bpp, a tradeoff rather than a fixed-rate win. Flower Quality worsens 1.67% MSE and Monet worsens slightly at both presets. There are only 3/12 strict RD wins. Its small cost does not compensate for inconsistent final gains.','']
    for v,s in summary.items():
        lines.append(f"Same-preset actual controls within 5% final rate cover {s['near_5pct_cases']}/12 {v} cases, with {s['near_5pct_mse_wins']} MSE wins and mean relative MSE change {s['near_5pct_mean_mse_change']:+.3f}%. These are unequal-rate files within a tolerance, not exact matches or interpolation. Encode/decode mean repeat CV is {s['cv_percent']['encode_seconds']:.2f}%/{s['cv_percent']['decode_seconds']:.2f}%.")
    lines+=['','The screen fails the intended meaningful final-RD/cost gate. The initialization relative-percentage average is driven by a few cases; its arithmetic RGB mean is almost unchanged and its encode cost is material. The iterative variant is cheap but does not give consistent fixed-rate quality improvement. No dense rate sweep or release comparison is needed to reject promotion at this stage. The immutable release remains available for a future qualified candidate; no four-axis claim can be made from these fixed-preset means.','',
        'The negative result is not “indices never move.” Actual quantized indices move, exact local scoring improves, and normal decoder reconstruction matches. The unresolved issue is transfer through later channel/geometry/stop decisions, color conversion and final resizing to original-size RGB. Those mechanisms are not assigned causal shares here. Another feedback strength or more correction iterations is not justified by this screen.','',
        'A distinct next mechanism is a bounded stop-policy audit: only when the first learned proposal would stop, evaluate one already-ranked alternative proposal and measure whether useful corrections were missed. Freeze the model and proposal generation, cap the extra work, and require actual matched-rate evidence before broader confirmation. This addresses allocation/stop behavior rather than repeating coefficient feedback or training.','']
    (HERE/'PBC_GRID_REFINEMENT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(dict(averages=averages,promoted=[]),indent=2))


if __name__=='__main__':
    main()
