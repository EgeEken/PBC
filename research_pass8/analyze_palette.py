import json
from pathlib import Path
from statistics import mean,stdev

HERE=Path(__file__).resolve().parent


def main():
    result=json.loads((HERE/'palette_screen.json').read_text())
    rows=result['rows']
    pairs=[]
    for row in rows:
        if row['mode']!='robust':
            continue
        baseline=next(r for r in rows if (r['image'],r['preset'],r['mode'])==(row['image'],row['preset'],'control'))
        pairs.append(dict(image=row['image'],preset=row['preset'],
            control={k:baseline[k] for k in ('mse','bpp','encode_seconds','decode_seconds','patches')},
            robust={k:row[k] for k in ('mse','bpp','encode_seconds','decode_seconds','patches')},
            change_percent={k:100*(row[k]/baseline[k]-1) for k in ('mse','bpp','encode_seconds','decode_seconds')},
            considered=row['stats']['considered'],accepted=row['stats']['accepted'],
            palette_seconds=row['stats']['palette_seconds'],
            strict_rd_win=row['mse']<baseline['mse'] and row['bpp']<=baseline['bpp']))
    means={mode:{k:mean(r[k] for r in rows if r['mode']==mode) for k in ('mse','bpp','encode_seconds','decode_seconds','patches')} for mode in ('control','robust')}
    decision=dict(decision='Reject after six-image screen; no broader confirmation or RD deployment.',means=means,aggregate_change_percent={k:100*(means['robust'][k]/means['control'][k]-1) for k in means['control']},
        mean_case_change_percent={k:mean(p['change_percent'][k] for p in pairs) for k in ('mse','bpp','encode_seconds','decode_seconds')},
        strict_rd_wins=sum(p['strict_rd_win'] for p in pairs),mse_wins=sum(p['change_percent']['mse']<0 for p in pairs),
        worst_mse_change_percent=max(p['change_percent']['mse'] for p in pairs),
        worst_bpp_change_percent=max(p['change_percent']['bpp'] for p in pairs),
        mean_timing_cv_percent={m:{k:mean(100*stdev(r[k+'_samples'])/mean(r[k+'_samples']) for r in rows if r['mode']==m) for k in ('encode_seconds','decode_seconds')} for m in ('control','robust')},
        considered=sum(p['considered'] for p in pairs),accepted=sum(p['accepted'] for p in pairs),pairs=pairs)
    (HERE/'palette_decision.json').write_text(json.dumps(decision,indent=2)+'\n',newline='\n')
    print(json.dumps(decision,indent=2))


if __name__=='__main__':
    main()
