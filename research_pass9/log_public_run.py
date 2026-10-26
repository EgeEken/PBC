import hashlib
import json
from pathlib import Path
import sqlite3
from statistics import mean

HERE=Path(__file__).resolve().parent
RUN='20261026_141310'
DISPLAYED={'PBC3':(.534,.082),'JPEG':(.017,.034),'JPEG2000':(2.792,.899),'WEBP':(.456,.055),'JXL':(1.827,.107),'AVIF':(1.187,.056),'PNG':(3.104,.261)}


def main():
    path=(HERE/'benchmark_stopped.db').resolve()
    con=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    con.row_factory=sqlite3.Row
    assert con.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    rows=[dict(r) for r in con.execute('SELECT * FROM results WHERE run_id=? ORDER BY id',(RUN,))]
    con.close()
    assert len(rows)==996
    first=[r for r in rows if r['trial']==1]
    assert len(first)==980 and len({r['image'] for r in first})==20
    summaries={}
    for codec,display in DISPLAYED.items():
        group=[r for r in first if r['codec']==codec]
        summary=dict(rows=len(group),images=len({r['image'] for r in group}),variants=len({r['variant'] for r in group}),
            encode_seconds=mean(r['encode_seconds'] for r in group),decode_seconds=mean(r['decode_seconds'] for r in group),
            screenshot_encode_seconds=display[0],screenshot_decode_seconds=display[1])
        visible=group[:5] if codec=='PNG' else group
        assert (round(mean(r['encode_seconds'] for r in visible),3),round(mean(r['decode_seconds'] for r in visible),3))==display
        summary['screenshot_matching_first_trial_images']=len({r['image'] for r in visible})
        summaries[codec]=summary
    presets={}
    for preset in ('compression','balanced','quality','high_quality'):
        group=[r for r in first if r['codec']=='PBC3' and r['variant']==preset]
        assert len(group)==20
        presets[preset]={m:mean(r[m] for r in group) for m in ('encode_seconds','decode_seconds','mse','bpp')}
    screenshot=HERE/'hf_runs/ui/user_speed_averages.png'
    result=dict(run_id=RUN,hf_revision='27f6c65225a5496fa065deb586c7e405ecdffc84',completed_rows=996,planned_rows=9800,
        complete_trials=1,partial_second_trial=dict(codec='PBC3',variant='compression',rows=16),
        first_trial=summaries,pbc3_first_trial_presets=presets,
        pbc3_first_trial_rows=[{k:r[k] for k in ('image','variant','mp','encode_seconds','decode_seconds','mse','bpp')} for r in first if r['codec']=='PBC3'],
        database_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),screenshot_sha256=hashlib.sha256(screenshot.read_bytes()).hexdigest(),
        screenshot_provenance='Displayed three-decimal values match first-trial six-codec full coverage and PNG first five images; filters/run selector are not visible in the screenshot.',
        controlled_pass9_comparison=False)
    (HERE/'public_run_log.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['# Existing public speed run log','',
        'At the user\'s request, existing run `20261026_141310` was stopped and its completed rows exported before any restart. It stopped without error at 996/9,800 calls: one complete 980-call trial (20 images, 49 codec/variant points), followed by 16 PBC3 Compression calls in trial two. No Pass 9 HF job, deployment or restart occurred. HF remains `27f6c65225a5496fa065deb586c7e405ecdffc84` on cpu-basic.','',
        '| Codec | Screenshot encode s | Screenshot decode s | Matching image coverage | Complete first-trial encode s | Complete first-trial decode s |',
        '| --- | ---: | ---: | --- | ---: | ---: |']
    for codec,s in summaries.items():
        lines.append(f'| {codec} | {s["screenshot_encode_seconds"]:.3f} | {s["screenshot_decode_seconds"]:.3f} | {s["screenshot_matching_first_trial_images"]}/20 | {s["encode_seconds"]:.6f} | {s["decode_seconds"]:.6f} |')
    lines+=['',
        'The screenshot\'s PNG values match precisely the first five PNG records (cat, cathedral, coast, ducklings, Eiffel), rounded to three decimals. Its other six bars match all 20 first-trial images at that precision. This supports an in-progress PNG snapshot; the screenshot does not show its run selector or filters. The complete first-trial PNG mean is 1.878271/0.164118 s. Do not interpret the partial PNG bar as the completed comparison.','',
        'The completed first-trial codec means above weight every codec/variant/image row equally. Each variant has 20 images, so this also equals the UI mean of variant means without filters. The later 16 PBC3-only calls are excluded to keep complete image coverage. Four PBC3 presets and the competitor quality grids represent different RD operating points; codec averages do not establish matched-rate superiority. There is only one complete trial and no per-encode source/module/RNG manifest from this public runner. These values are an observational public-run log, separate from the unperformed controlled three-way Pass 9 comparison.','',
        'The local ignored backup `benchmark_stopped.db` contains 6,396 total rows, including two historical runs and all 996 rows of this run; SQLite quick_check passed. Its SHA256 is `'+result['database_sha256']+'`. A prior read-only snapshot preserves 945 active-run rows. Neither database nor the user screenshot is committed. `run_stop.json` records the authorized stop and unchanged HF revision; `public_run_log.json` records compact first-trial summaries and the 80 PBC3 per-image rows. Reproduce this analysis without running any codec using `python research_pass9/log_public_run.py` against the retained backup and ignored screenshot.','']
    (HERE/'PBC_EXISTING_SPEED_RUN_LOG.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('\n'.join(lines[:14]))


if __name__=='__main__':
    main()
