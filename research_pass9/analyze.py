import hashlib
import json
from pathlib import Path
from statistics import mean,median,stdev

HERE=Path(__file__).resolve().parent
SOURCES=('release','dev_release_rng','dev_current_rng')
LABELS=dict(compression='Compression',balanced='Balanced',quality='Quality',high_quality='High Quality',overall='Overall')
METRICS=('encode_seconds','decode_seconds','mse','bpp','patches','initialization_patches','iterative_patches')


def main():
    raw={name:json.loads((HERE/('hf_'+name+'.json')).read_text()) for name in ('smoke','equivalence','comparison')}
    measured=raw['comparison']
    assert len(measured['rows'])==288
    compact=[]
    for image in measured['metadata']['spec']['images']:
        for preset in measured['metadata']['spec']['presets']:
            for source in SOURCES:
                group=[r for r in measured['rows'] if (r['image'],r['preset'],r['source'])==(image,preset,source)]
                assert len(group)==3
                for field in ('stream_sha256','pixels_sha256','trajectory_sha256','mse','bpp','patches'):
                    assert len({r[field] for r in group})==1
                row={k:v for k,v in group[0].items() if k not in ('trajectory','trial','encode_seconds','decode_seconds')}
                assert row['preview_equal']
                for metric in ('encode_seconds','decode_seconds'):
                    samples=[r[metric] for r in group]
                    row[metric]=median(samples)
                    row[metric+'_samples']=samples
                    row[metric+'_cv_percent']=100*stdev(samples)/mean(samples)
                compact.append(row)
    gate_rows=raw['smoke']['rows']+raw['equivalence']['rows']
    pairs=[]
    for image,preset in sorted({(r['image'],r['preset']) for r in gate_rows}):
        a,b=[next(r for r in gate_rows if (r['source'],r['image'],r['preset'])==(s,image,preset)) for s in SOURCES[:2]]
        fields=('stream_sha256','pixels_sha256','trajectory_sha256','mse','bpp','patches','final_bytes','initialization_patches','iterative_patches')
        assert all(a[k]==b[k] for k in fields),(image,preset)
        pairs.append(dict(dataset=a['dataset'],image=image,preset=preset,input_sha256=a['input_sha256'],
            release={k:a[k] for k in fields},dev_release_rng={k:b[k] for k in fields}))
    assert len(pairs)==100
    prior=json.loads((HERE.parent/'research_pass8/comparison_summary.json').read_text())
    continuity={}
    for source,old in (('release','release'),('dev_current_rng','development')):
        rows=[r for r in gate_rows+measured['rows'] if r['source']==source and r['dataset']=='hpt_data']
        cases={(r['image'],r['preset']):r for r in rows}
        for (image,preset),r in cases.items():
            old_row=next(x for x in prior['rows'] if (x['image'],x['preset'],x['source'])==(image,preset,old))
            assert all(r[k]==old_row[k] for k in ('stream_sha256','pixels_sha256','trajectory_sha256','mse','bpp','patches'))
        continuity[source]=len(cases)
    summaries={}
    per_cases=[]
    for image in measured['metadata']['spec']['images']:
        for preset in measured['metadata']['spec']['presets']:
            rows={s:next(r for r in compact if (r['source'],r['image'],r['preset'])==(s,image,preset)) for s in SOURCES}
            per_cases.append(dict(image=image,preset=preset,variants={s:{k:r[k] for k in METRICS} for s,r in rows.items()},
                release_rng_encode_reduction_vs_release=100*(1-rows['dev_release_rng']['encode_seconds']/rows['release']['encode_seconds']),
                release_rng_decode_reduction_vs_release=100*(1-rows['dev_release_rng']['decode_seconds']/rows['release']['decode_seconds']),
                release_rng_encode_reduction_vs_current=100*(1-rows['dev_release_rng']['encode_seconds']/rows['dev_current_rng']['encode_seconds'])))
    for preset in list(LABELS):
        selected=[r for r in compact if preset=='overall' or r['preset']==preset]
        summary={s:{k:mean(r[k] for r in selected if r['source']==s) for k in METRICS} for s in SOURCES}
        summary['reductions']={}
        summary['variation']={}
        for metric in ('encode_seconds','decode_seconds'):
            summary['reductions'][metric]={s+'_vs_'+reference:100*(1-summary[s][metric]/summary[reference][metric])
                for s,reference in (('dev_release_rng','release'),('dev_current_rng','release'),('dev_release_rng','dev_current_rng'))}
            summary['variation'][metric]={s:dict(mean_case_cv_percent=mean(r[metric+'_cv_percent'] for r in selected if r['source']==s)) for s in SOURCES}
            cases=[r for r in per_cases if preset=='overall' or r['preset']==preset]
            for s in SOURCES[1:]:
                reductions=[100*(1-r['variants'][s][metric]/r['variants']['release'][metric]) for r in cases]
                summary['variation'][metric][s].update(wins=sum(x>0 for x in reductions),cases=len(cases),minimum=min(reductions),median=median(reductions),maximum=max(reductions))
        summaries[preset]=summary
    trial_means=[dict(trial=t,source=s,**{m:mean(r[m] for r in measured['rows'] if r['trial']==t and r['source']==s) for m in METRICS[:2]}) for t in range(3) for s in SOURCES]
    quality=[p for p in pairs if p['dataset']=='hpt_data' and p['preset']=='quality']
    release_bpp=mean(p['release']['bpp'] for p in quality)
    restored_bpp=mean(p['dev_release_rng']['bpp'] for p in quality)
    old_current_bpp=prior['summaries']['quality']['development']['bpp']
    restoration=dict(main_quality_cases=len(quality),release_bpp=release_bpp,restored_bpp=restored_bpp,
        pass8_current_bpp=old_current_bpp,pass8_regression_percent=100*(old_current_bpp/release_bpp-1),
        remaining_regression_percent=100*(restored_bpp/release_bpp-1),excess_bpp_removed_percent=100*(old_current_bpp-restored_bpp)/(old_current_bpp-release_bpp))
    metadata={}
    for name,result in raw.items():
        meta=result['metadata']
        metadata[name]={**meta,'warmups':[{k:v for k,v in r.items() if k!='trajectory'} for r in meta['warmups']]}
    output=dict(summaries=summaries,rows=compact,per_image=per_cases,equivalence=pairs,quality_restoration=restoration,
        pass8_output_continuity=continuity,trial_means=trial_means,metadata=metadata,
        raw_sha256={name:hashlib.sha256((HERE/('hf_'+name+'.json')).read_bytes()).hexdigest() for name in raw})
    (HERE/'comparison_summary.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8',newline='\n')
    speed=['| Preset | Release encode | Dev release-RNG encode | Dev current encode | Release decode | Dev release-RNG decode | Dev current decode |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    reductions=['| Preset | Release-RNG encode vs release | Current encode vs release | Release-RNG encode vs current | Release-RNG decode vs release | Current decode vs release | Release-RNG decode vs current |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    rd=['| Preset | MSE release / release-RNG / current | Bpp release / release-RNG / current | Total patches release / release-RNG / current | Iterative patches release / release-RNG / current |','| --- | ---: | ---: | ---: | ---: |']
    for preset,s in summaries.items():
        speed.append('| '+LABELS[preset]+' | '+' | '.join(f'{s[source][metric]:.6f}' for metric in METRICS[:2] for source in SOURCES)+' |')
        reductions.append('| '+LABELS[preset]+' | '+' | '.join(f'{s["reductions"][metric][key]:+.2f}%' for metric in METRICS[:2] for key in ('dev_release_rng_vs_release','dev_current_rng_vs_release','dev_release_rng_vs_dev_current_rng'))+' |')
        rd.append('| '+LABELS[preset]+' | '+' | '.join(' / '.join(f'{s[source][metric]:.6f}' for source in SOURCES) for metric in ('mse','bpp','patches','iterative_patches'))+' |')
    per=['| Image | Preset | Encode seconds release / release-RNG / current | Decode seconds release / release-RNG / current | Release-RNG encode/decode reduction vs release | Release-RNG encode reduction vs current | Patches release / release-RNG / current |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for p in per_cases:
        v=p['variants']
        per.append('| '+p['image']+' | '+LABELS[p['preset']]+' | '+' | '.join(' / '.join(f'{v[s][m]:.6f}' for s in SOURCES) for m in METRICS[:2])+f' | {p["release_rng_encode_reduction_vs_release"]:+.2f}% / {p["release_rng_decode_reduction_vs_release"]:+.2f}% | {p["release_rng_encode_reduction_vs_current"]:+.2f}% | '+' / '.join(str(v[s]['patches']) for s in SOURCES)+' |')
    for name,lines in (('speed_table',speed),('reduction_table',reductions),('rd_table',rd),('per_image_results',per)):
        (HERE/(name+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('\n'.join(speed)); print('\n'.join(reductions)); print(json.dumps(restoration,indent=2))


if __name__=='__main__':
    main()
