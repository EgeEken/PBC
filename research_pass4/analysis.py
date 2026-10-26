from collections import defaultdict
import argparse
import json
import statistics

from support import OUT, PASS2, PASS3, read, save
from analyze import collapse, summaries, pair, frontier, interpolate
from matched_rate import points


def runtime(name):
    raw = read(name)
    rows = collapse(raw)
    assert all(r['deterministic'] and r['preview_equals_decode'] for r in rows)
    base = {(r['dataset'],r['image'],r['preset']):r for r in rows if r['label']=='baseline'}
    for row in rows:
        control = base[(row['dataset'],row['image'],row['preset'])]
        assert all(row[k]==control[k] for k in ('stream_sha256','pixels_sha256','patch_signatures','mse','bpp','iterative_patches'))
    save(name+'_summary', summaries(raw))
    labels = sorted({r['label'] for r in rows}-{'baseline'})
    comparisons = pair(raw, [(v,'baseline') for v in labels]+([('all','grid_skip'),('all','search')] if 'all' in labels else []), name+'_paired')
    groups = defaultdict(list)
    for r in comparisons:
        groups[(r['dataset'],r['candidate'],r['baseline'])].append(r)
    result=[]
    for (dataset,candidate,baseline),group in groups.items():
        record=dict(dataset=dataset,candidate=candidate,baseline=baseline,cases=len(group),
            exact=sum(r['bytes_equal'] and r['pixels_equal'] and r['patches_equal'] for r in group))
        for metric in ('encode_seconds','decode_seconds'):
            a=statistics.mean(r[metric+'_baseline'] for r in group)
            b=statistics.mean(r[metric+'_candidate'] for r in group)
            record.update({metric+'_baseline':a,metric+'_candidate':b,metric+'_pct':100*(b/a-1),
                metric+'_wins':sum(r[metric+'_candidate']<r[metric+'_baseline'] for r in group)})
        result.append(record)
    save(name+'_aggregate',result)
    print(json.dumps(result,indent=2))


def matched():
    candidates, prior, ordinary = points()
    fresh = read('matched_controls')
    records=[]
    for candidate in candidates:
        same=lambda r:(r['dataset'],r['image'])==(candidate['dataset'],candidate['image'])
        old=[r for r in prior if same(r)]
        new=[r for r in fresh if same(r)]
        ordinary500=[r for r in ordinary if same(r)]
        combined=old+new
        nearest=min(new,key=lambda r:abs(r['bpp']/candidate['bpp']-1))
        fresh_below=[r for r in new if r['bpp']<=candidate['bpp']]
        fresh_above=[r for r in new if r['bpp']>=candidate['bpp']]
        lower_fresh=max(fresh_below,key=lambda r:r['bpp']) if fresh_below else None
        upper_fresh=min(fresh_above,key=lambda r:r['bpp']) if fresh_above else None
        nearest500=min(ordinary500+new,key=lambda r:abs(r['bpp']/candidate['bpp']-1))
        envelope=frontier(combined)
        bracket=interpolate(envelope,candidate['bpp'])
        previous=interpolate(frontier(old),candidate['bpp'])
        conservative=min([m for m in (bracket,previous) if m is not None],key=lambda m:m['value']) if bracket or previous else None
        control500bracket=interpolate(frontier(ordinary500+new),candidate['bpp'])
        strict=lambda a,b:a['mse']<=b['mse'] and a['bpp']<=b['bpp'] and (a['mse']<b['mse'] or a['bpp']<b['bpp'])
        qhq=[r for r in old if r['source']=='release' and r['label'] in ('release_quality','release_high_quality','release_250000','release_hq250')]
        release_defaults=[r for r in old if r.get('kind')=='default_control' and r['source']=='release']
        qlo=next((r for r in release_defaults if r['preset']=='quality'),None)
        qhi=next((r for r in release_defaults if r['preset']=='high_quality'),None)
        qhq_region=bool(qlo and qhi and min(qlo['bpp'],qhi['bpp'])<=candidate['bpp']<=max(qlo['bpp'],qhi['bpp']))
        close=[r for r in combined if abs(r['bpp']/candidate['bpp']-1)<=.05]
        best_close=min(close,key=lambda r:r['mse']) if close else None
        record=dict(dataset=candidate['dataset'],image=candidate['image'],candidate=candidate,
            fresh_points=len(new),nearest_fresh=nearest,nearest500=nearest500,
            fresh_below=lower_fresh,fresh_above=upper_fresh,
            fresh_below_gap_pct=100*(candidate['bpp']-lower_fresh['bpp'])/candidate['bpp'] if lower_fresh else None,
            fresh_above_gap_pct=100*(upper_fresh['bpp']-candidate['bpp'])/candidate['bpp'] if upper_fresh else None,
            rate_gap_pct=100*(nearest['bpp']/candidate['bpp']-1),
            direct_mse_pct=100*(candidate['mse']/nearest['mse']-1),
            direct_strict_win=strict(candidate,nearest),direct_strict_loss=strict(nearest,candidate),
            within_3pct=abs(nearest['bpp']/candidate['bpp']-1)<=.03,
            within_5pct=abs(nearest['bpp']/candidate['bpp']-1)<=.05,
            densified_bracket=bracket,ordinary500_bracket=control500bracket,previous_bracket=previous,
            conservative_bracket=conservative,
            densified_matched_mse_pct=100*(candidate['mse']/bracket['value']-1) if bracket else None,
            strongest_matched_mse_pct=100*(candidate['mse']/conservative['value']-1) if conservative else None,
            ordinary500_matched_mse_pct=100*(candidate['mse']/control500bracket['value']-1) if control500bracket else None,
            previous_matched_mse_pct=100*(candidate['mse']/previous['value']-1) if previous else None,
            dominated_by_measured_control=any(strict(r,candidate) for r in combined),
            qhq_region=qhq_region,
            strongest_close_control=best_close,
            strongest_close_mse_pct=100*(candidate['mse']/best_close['mse']-1) if best_close else None,
            strongest_close_rate_gap_pct=100*(best_close['bpp']/candidate['bpp']-1) if best_close else None,
            strongest_close_strict_win=strict(candidate,best_close) if best_close else None,
            strongest_close_strict_loss=strict(best_close,candidate) if best_close else None,
            nearest_old_distance_pct=min(abs(r['bpp']/candidate['bpp']-1)*100 for r in frontier(old)))
        records.append(record)
    save('matched_rate_comparisons',records)
    summary=[]
    for dataset in ('hpt_data','hpt_data_val'):
        group=[r for r in records if r['dataset']==dataset]
        for subset in ('all','close3','close5','qhq_close5','qhq_bestclose'):
            if subset=='all':
                g=group
            elif subset=='qhq_bestclose':
                g=[r for r in group if r['qhq_region'] and r['strongest_close_control'] is not None]
            else:
                g=[r for r in group if (r['within_3pct'] if subset=='close3' else r['within_5pct']) and (subset!='qhq_close5' or r['qhq_region'])]
            record=dict(dataset=dataset,subset=subset,cases=len(g),
                direct_mse_pct=statistics.mean(r['direct_mse_pct'] for r in g) if g else None,
                direct_mse_wins=sum(r['direct_mse_pct']<0 for r in g),
                direct_strict_wins=sum(r['direct_strict_win'] for r in g),direct_strict_losses=sum(r['direct_strict_loss'] for r in g),
                mean_signed_rate_gap_pct=statistics.mean(r['rate_gap_pct'] for r in g) if g else None,
                mean_abs_rate_gap_pct=statistics.mean(abs(r['rate_gap_pct']) for r in g) if g else None,
                dominated=sum(r['dominated_by_measured_control'] for r in g))
            for name in ('strongest_matched_mse_pct','densified_matched_mse_pct','ordinary500_matched_mse_pct','previous_matched_mse_pct','strongest_close_mse_pct'):
                valid=[r[name] for r in g if r[name] is not None]
                record[name]=statistics.mean(valid) if valid else None
                record[name+'_coverage']=len(valid)
                record[name+'_wins']=sum(v<0 for v in valid)
            close=[r for r in g if r['strongest_close_control'] is not None]
            record.update(strongest_close_strict_wins=sum(r['strongest_close_strict_win'] for r in close),
                strongest_close_strict_losses=sum(r['strongest_close_strict_loss'] for r in close),
                strongest_close_median_pct=statistics.median(r['strongest_close_mse_pct'] for r in close) if close else None,
                fresh_direct_median_pct=statistics.median(r['direct_mse_pct'] for r in g) if g else None)
            summary.append(record)
    save('matched_rate_summary',summary)
    save('matched_rate_points',[{**r,'experiment':'prior'} for r in prior]+[{**r,'experiment':'fresh_control'} for r in fresh]+[{**r,'experiment':'frozen_allocator'} for r in candidates])
    confirmation=[]
    for row in records:
        for label, config, optimization in (('ordinary',row['nearest_fresh']['config'],'none'),('allocator',row['candidate']['config'],'alloc')):
            overrides={k:config[k] for k in ('auto_downsample_max_pixels','patch_count','learned_filler_q')}
            confirmation.append(dict(path=str(OUT.parent/row['dataset']/row['image']),preset='quality',label=label,
                optimization=optimization,overrides=overrides,retain_stream=False))
    save('matched_confirmation_plan',dict(cases=confirmation,repeats=3,selection='Nearest fresh per-image bpp only; frozen allocator; preserve exactly selected controls.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name')
    args=parser.parse_args()
    runtime(args.name) if args.name.startswith('runtime_') else matched()
