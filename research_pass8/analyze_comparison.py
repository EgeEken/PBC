import hashlib
import json
from pathlib import Path
from statistics import mean,median,stdev

HERE=Path(__file__).resolve().parent
METRICS=('encode_seconds','decode_seconds','mse','bpp','patches','initialization_patches','iterative_patches')
LABELS=dict(compression='Compression',balanced='Balanced',quality='Quality',high_quality='High Quality',overall='Overall')


def main():
    path=HERE/'hf_comparison.json'
    result=json.loads(path.read_text())
    spec=result['metadata']['spec']
    assert len(result['rows'])==480 and spec['repeats']==3 and len(spec['images'])==20
    compact=[]
    for image in spec['images']:
        for preset in spec['presets']:
            for source in ('release','development'):
                group=[r for r in result['rows'] if (r['image'],r['preset'],r['source'])==(image,preset,source)]
                assert len(group)==3 and len({r['stream_sha256'] for r in group})==len({r['pixels_sha256'] for r in group})==1
                assert all(r['preview_equal'] and r['input_sha256']==result['metadata']['manifest']['assets']['hpt_data/'+image] for r in group)
                row={k:v for k,v in group[0].items() if k not in ('trial','encode_seconds','decode_seconds')}
                for metric in ('encode_seconds','decode_seconds'):
                    samples=[r[metric] for r in group]
                    row[metric]=median(samples)
                    row[metric+'_samples']=samples
                    row[metric+'_cv_percent']=100*stdev(samples)/mean(samples)
                compact.append(row)
    pairs=[]
    for image in spec['images']:
        for preset in spec['presets']:
            a,b=[next(r for r in compact if (r['image'],r['preset'],r['source'])==(image,preset,s)) for s in ('release','development')]
            pairs.append(dict(image=image,preset=preset,release={k:a[k] for k in METRICS},development={k:b[k] for k in METRICS},
                encode_reduction_percent=100*(1-b['encode_seconds']/a['encode_seconds']),
                decode_reduction_percent=100*(1-b['decode_seconds']/a['decode_seconds']),
                mse_change_percent=100*(b['mse']/a['mse']-1),bpp_change_percent=100*(b['bpp']/a['bpp']-1),
                same_stream=a['stream_sha256']==b['stream_sha256'],same_pixels=a['pixels_sha256']==b['pixels_sha256']))
    summaries={}
    for preset in spec['presets']+['overall']:
        group=[p for p in pairs if preset=='overall' or p['preset']==preset]
        summary={s:{k:mean(p[s][k] for p in group) for k in METRICS} for s in ('release','development')}
        for metric in ('encode','decode'):
            key=metric+'_seconds'
            values=sorted(p[metric+'_reduction_percent'] for p in group)
            summary[metric+'_reduction_percent']=100*(1-summary['development'][key]/summary['release'][key])
            summary[metric+'_paired_distribution']=dict(mean=mean(values),median=median(values),minimum=min(values),maximum=max(values),wins=sum(v>0 for v in values),cases=len(values))
            summary[metric+'_mean_case_cv_percent']={s:mean(r[key+'_cv_percent'] for r in compact if r['source']==s and (preset=='overall' or r['preset']==preset)) for s in ('release','development')}
        for metric in ('mse','bpp'):
            summary[metric+'_change_percent']=100*(summary['development'][metric]/summary['release'][metric]-1)
            summary[metric+'_wins']=sum(p['development'][metric]<p['release'][metric] for p in group)
        summary['strict_rd_wins']=sum(p['development']['mse']<p['release']['mse'] and p['development']['bpp']<=p['release']['bpp'] for p in group)
        summary['strict_rd_losses']=sum(p['development']['mse']>p['release']['mse'] and p['development']['bpp']>=p['release']['bpp'] for p in group)
        summaries[preset]=summary
    summary_result=dict(summaries=summaries,pairs=pairs,rows=compact,metadata=result['metadata'],
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_measured_calls=480,
        trial_means=[dict(trial=t,source=s,**{k:mean(r[k] for r in result['rows'] if r['trial']==t and r['source']==s) for k in ('encode_seconds','decode_seconds')}) for t in range(3) for s in ('release','development')])
    (HERE/'comparison_summary.json').write_text(json.dumps(summary_result,indent=2)+'\n',newline='\n')
    speed=['| Preset | PBC 3.0 encode (s) | Dev encode (s) | Encode reduction | PBC 3.0 decode (s) | Dev decode (s) | Decode reduction |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    rd=['| Preset | Release MSE | Dev MSE | MSE change | Release bpp | Dev bpp | Bpp change | Release patches | Dev patches |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    variation=['| Preset | Encode wins | Paired encode reduction median [min, max] | Decode wins | Paired decode reduction median [min, max] | Encode CV release/dev | Decode CV release/dev |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p,s in summaries.items():
        a,b=s['release'],s['development']
        speed.append(f"| {LABELS[p]} | {a['encode_seconds']:.6f} | {b['encode_seconds']:.6f} | {s['encode_reduction_percent']:+.2f}% | {a['decode_seconds']:.6f} | {b['decode_seconds']:.6f} | {s['decode_reduction_percent']:+.2f}% |")
        rd.append(f"| {LABELS[p]} | {a['mse']:.6f} | {b['mse']:.6f} | {s['mse_change_percent']:+.3f}% | {a['bpp']:.6f} | {b['bpp']:.6f} | {s['bpp_change_percent']:+.3f}% | {a['patches']:.2f} | {b['patches']:.2f} |")
        e,d=s['encode_paired_distribution'],s['decode_paired_distribution']
        ec,dc=s['encode_mean_case_cv_percent'],s['decode_mean_case_cv_percent']
        variation.append(f"| {LABELS[p]} | {e['wins']}/{e['cases']} | {e['median']:+.2f}% [{e['minimum']:+.2f}, {e['maximum']:+.2f}] | {d['wins']}/{d['cases']} | {d['median']:+.2f}% [{d['minimum']:+.2f}, {d['maximum']:+.2f}] | {ec['release']:.1f}/{ec['development']:.1f}% | {dc['release']:.1f}/{dc['development']:.1f}% |")
    (HERE/'speed_table.md').write_text('\n'.join(speed)+'\n',newline='\n')
    (HERE/'rd_table.md').write_text('\n'.join(rd)+'\n',newline='\n')
    (HERE/'variation_table.md').write_text('\n'.join(variation)+'\n',newline='\n')
    per_image=['| Image | Preset | Encode reduction % | Decode reduction % | MSE release → dev | Bpp release → dev | Init release/dev | Iterative release/dev |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in pairs:
        a,b=p['release'],p['development']
        per_image.append(f"| {p['image']} | {LABELS[p['preset']]} | {p['encode_reduction_percent']:+.2f} | {p['decode_reduction_percent']:+.2f} | {a['mse']:.4f} → {b['mse']:.4f} | {a['bpp']:.5f} → {b['bpp']:.5f} | {a['initialization_patches']}/{b['initialization_patches']} | {a['iterative_patches']}/{b['iterative_patches']} |")
    (HERE/'per_image_results.md').write_text('\n'.join(per_image)+'\n',encoding='utf-8',newline='\n')
    print('\n'.join(speed)); print('\n'.join(rd))


if __name__=='__main__':
    main()
