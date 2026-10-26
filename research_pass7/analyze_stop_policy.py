import json
from pathlib import Path
from statistics import mean, median, stdev

HERE=Path(__file__).resolve().parent


def main():
    screen=json.loads((HERE/'stop_screen.json').read_text())
    raw=[json.loads(line) for line in (HERE/'stop_screen_raw.jsonl').read_text().splitlines()]
    main_rows=[r for r in screen['rows'] if r['preset'] in ('quality','high_quality')]
    bases={(r['image'],r['preset']):r for r in main_rows if r['mode']=='control'}
    pairs=[]
    for row in main_rows:
        if row['mode']!='S2':
            continue
        base=bases[row['image'],row['preset']]
        changes={k:100*(row[k]/base[k]-1) for k in ('mse','bpp','encode_seconds','decode_seconds')}
        timings={k:median(r['stats'][k] for r in raw if r['mode']=='S2' and (r['image'],r['preset'])==(row['image'],row['preset'])) for k in ('extra_inference_seconds','extra_construction_seconds','extra_total_seconds')}
        pairs.append(dict(image=row['image'],preset=row['preset'],baseline={k:base[k] for k in ('mse','bpp','encode_seconds','decode_seconds','patches')},
            candidate={k:row[k] for k in ('mse','bpp','encode_seconds','decode_seconds','patches')},changes_percent=changes,
            attempts=row['stats']['attempts'],rescued=row['stats']['rescued'],
            extra_timing_medians=timings,
            same_preset_within_5pct_rate=abs(changes['bpp'])<=5,
            strict_rd_win=changes['mse']<0 and changes['bpp']<=0,
            unchanged_stream=row['stream_sha256']==base['stream_sha256']))
    means={mode:{k:mean(r[k] for r in main_rows if r['mode']==mode) for k in ('mse','bpp','encode_seconds','decode_seconds','patches')} for mode in ('control','S1','S2')}
    aggregate={k:100*(means['S2'][k]/means['control'][k]-1) for k in means['control']}
    diagnosis=json.loads((HERE/'diagnosis.json').read_text())
    diagnosed=[r for r in diagnosis['rows'] if r['mode']=='diagnose' and r['preset'] in ('quality','high_quality')]
    for r in screen['rows']:
        if r['mode']=='S1':
            b=next(x for x in screen['rows'] if x['mode']=='control' and (x['image'],x['preset'])==(r['image'],r['preset']))
            assert r['stream_sha256']==b['stream_sha256'] and r['stats']['attempts']==0
    result=dict(decision='Stop: S1 is inapplicable to the deployed model; S2 does not pass RD/CPU screening.',
        cases=len(pairs),means=means,aggregate_change_percent=aggregate,
        mean_case_relative_change_percent={k:mean(p['changes_percent'][k] for p in pairs) for k in pairs[0]['changes_percent']},
        diagnosed=dict(cases=len(diagnosed),budget_exhausted=sum(r['stats']['budget_exhausted'] for r in diagnosed),
            nonpositive=sum(r['stats']['nonpositive'] for r in diagnosed),stop_head=0,no_proposals=sum(r['stats']['no_proposals'] for r in diagnosed)),
        attempts=sum(p['attempts'] for p in pairs),rescued=sum(p['rescued'] for p in pairs),
        changed_cases=sum(not p['unchanged_stream'] for p in pairs),strict_rd_wins=sum(p['strict_rd_win'] for p in pairs),
        worst_mse_change_percent=max(p['changes_percent']['mse'] for p in pairs),
        worst_encode_change_percent=max(p['changes_percent']['encode_seconds'] for p in pairs),
        mean_case_cv_percent={mode:{k:mean(100*stdev(r[k+'_samples'])/mean(r[k+'_samples']) for r in main_rows if r['mode']==mode) for k in ('encode_seconds','decode_seconds')} for mode in ('control','S1','S2')},
        pairs=pairs)
    (HERE/'stop_decision.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
