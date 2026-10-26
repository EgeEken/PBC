import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from support import ROOT, OUT, PASS2, PASS3, THREAD_ENV, save, read, snapshot, request
from analyze import collapse, frontier, interpolate


def points():
    candidates = [r for r in collapse(json.loads((PASS3/'allocation_full.json').read_text()))
        if r['label'] == 'q500p35_integrated_t0.4']
    prior = json.loads((PASS2/'rd_raw_points.json').read_text())
    ordinary = [r for r in collapse(json.loads((PASS2/'quality_confirmation.json').read_text()))
        if r['label'] in ('q500_p20', 'q500_p35')]
    return candidates, prior, ordinary


def diagnostic():
    candidates, prior, ordinary = points()
    rows = []
    for candidate in candidates:
        group = [r for r in prior if (r['dataset'],r['image']) == (candidate['dataset'],candidate['image'])]
        envelope = frontier(group)
        below = [r for r in envelope if r['bpp'] <= candidate['bpp']]
        above = [r for r in envelope if r['bpp'] >= candidate['bpp']]
        lower = max(below, key=lambda r:r['bpp']) if below else None
        upper = min(above, key=lambda r:r['bpp']) if above else None
        nearest500 = min([r for r in ordinary if (r['dataset'],r['image']) == (candidate['dataset'],candidate['image'])], key=lambda r:abs(r['bpp']/candidate['bpp']-1))
        rows.append(dict(dataset=candidate['dataset'], image=candidate['image'], candidate=candidate,
            lower=lower, upper=upper, nearest_500=nearest500,
            lower_gap_bpp=candidate['bpp']-lower['bpp'] if lower else None,
            upper_gap_bpp=upper['bpp']-candidate['bpp'] if upper else None,
            lower_gap_pct=100*(candidate['bpp']/lower['bpp']-1) if lower else None,
            upper_gap_pct=100*(upper['bpp']/candidate['bpp']-1) if upper else None,
            bracket_span_pct=100*(upper['bpp']-lower['bpp'])/candidate['bpp'] if lower and upper else None,
            measured_bracket=lower is not None and upper is not None,
            already_dominated=any(r['bpp'] <= candidate['bpp'] and r['mse'] <= candidate['mse'] for r in group)))
    save('prior_rate_overlap', rows)


def controls():
    before = snapshot()
    candidates, prior, ordinary = points()
    log = (OUT/'matched_controls_current_stderr.log').open('w')
    process = subprocess.Popen([sys.executable, str(OUT/'runner.py'), '--worker', 'current'], cwd=ROOT,
        env={**os.environ, **THREAD_ENV, 'PYTHONPATH':''}, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    rows, decisions, warmups = [], [], []
    metadata = request(process, {'op':'metadata'})
    warmed = set()
    try:
        with (OUT/'matched_controls.jsonl').open('w', encoding='utf-8') as output:
            for candidate in candidates:
                key = candidate['dataset'], candidate['image']
                known = [r for r in ordinary if (r['dataset'],r['image']) == key]
                fresh = []
                for step in range(6):
                    q = 0.8
                    tested = {int(r['config']['patch_count']) for r in known+fresh if r['config']['learned_filler_q'] == q}
                    if step < 2:
                        budget = (10,25)[step]
                    else:
                        nearest = min(fresh, key=lambda r:abs(r['bpp']/candidate['bpp']-1))
                        if abs(nearest['bpp']/candidate['bpp']-1) <= 0.03:
                            break
                        available = [r for r in known+fresh if r['config']['learned_filler_q'] == q]
                        below = sorted([r for r in available if r['bpp'] <= candidate['bpp']], key=lambda r:-r['bpp'])
                        above = sorted([r for r in available if r['bpp'] >= candidate['bpp']], key=lambda r:r['bpp'])
                        budget = None
                        if below and above:
                            lo, hi = below[0], above[0]
                            p0,p1 = lo['config']['patch_count'],hi['config']['patch_count']
                            possibilities = [p for p in range(min(p0,p1)+1,max(p0,p1)) if p not in tested]
                            if possibilities:
                                estimate = p0+(candidate['bpp']-lo['bpp'])/(hi['bpp']-lo['bpp'])*(p1-p0) if hi['bpp'] != lo['bpp'] else (p0+p1)/2
                                budget = min(possibilities, key=lambda p:abs(p-estimate))
                        elif not below:
                            budget = 3 if 3 not in tested else None
                        elif not above:
                            budget = 50 if 50 not in tested else None
                        if budget is None:
                            choices = [v for v in (0.7,0.9) if not any(r['config']['learned_filler_q']==v for r in fresh)]
                            if not choices:
                                break
                            q,budget = choices[0],35
                    overrides = dict(auto_downsample_max_pixels=500000, patch_count=budget, learned_filler_q=q)
                    if q not in warmed:
                        warmups.append(request(process, dict(op='encode',path=str(ROOT/'hpt_data'/'googlewiki.png'),
                            preset='quality',overrides=overrides,optimization='none',warmup=True)))
                        warmed.add(q)
                    label = f'ordinary_p{budget}_q{q}'
                    row = request(process, dict(op='encode',path=str(ROOT/key[0]/key[1]),preset='quality',
                        overrides=overrides,optimization='none',label=label,trial=0,save=True,experiment='matched_controls'))
                    rows.append(row)
                    fresh.append(row)
                    output.write(json.dumps(row)+'\n')
                    output.flush()
                    decisions.append(dict(dataset=key[0],image=key[1],step=step,budget=budget,q=q,
                        target_bpp=candidate['bpp'],actual_bpp=row['bpp'],rate_gap_pct=100*(row['bpp']/candidate['bpp']-1)))
                print('matched',key,len(fresh),'nearest gap',min(abs(r['bpp']/candidate['bpp']-1)*100 for r in fresh),flush=True)
    finally:
        request(process, {'op':'quit'})
        process.wait(timeout=30)
        log.close()
        save('matched_controls_provenance',dict(before=before,after=snapshot(),metadata=metadata,warmups=warmups,
            method='Fresh ordinary encodes; at most six settings/image; selection based on bpp only; no prefixes. One full warmup per learned-q value, shared compiled kernels; individual exploratory timings are not confirmation speed claims.'))
    save('matched_controls',rows)
    save('matched_control_decisions',decisions)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('diagnostic','controls'))
    args=parser.parse_args()
    globals()[args.mode]()
