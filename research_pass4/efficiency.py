import argparse
import json
import statistics
from collections import defaultdict

from support import ROOT, OUT, SCREEN, images, save, snapshot, read
from analyze import collapse, summaries, pair


def plan(stage):
    paths=[ROOT/'hpt_data'/n for n in SCREEN] if stage in ('profile','screen','exactness') else images()
    modes=[('ordinary','none'),('allocator','alloc')]
    if stage in ('screen','exactness'):
        modes += [('projection','alloc_opt_projection'),('before_sse','alloc_opt_before'),('optimized','alloc_opt_both')]
        if stage=='exactness':
            modes=modes[1:]
    elif stage=='full':
        modes += [('optimized','alloc_opt_both')]
    elif stage=='fast':
        modes=[('fast_ordinary','all'),('fast_allocator','all_alloc'),('fast_optimized','all_alloc_opt_both')]
    cases=[dict(path=str(p),preset='quality',label=label,optimization=mode,
        overrides=dict(auto_downsample_max_pixels=500000,patch_count=35,compute_final_mse=stage!='fast'),
        profile=stage=='profile',retain_stream=stage=='screen') for p in paths for label,mode in modes]
    save('efficiency_'+stage+'_plan',dict(decision=snapshot(),cases=cases,repeats=1 if stage=='exactness' else 2 if stage=='profile' else 3))


def analyze(name):
    raw=read(name)
    rows=collapse(raw)
    for row in rows:
        if row['label'] in ('projection','before_sse','optimized','fast_optimized'):
            base_label='fast_allocator' if row['label']=='fast_optimized' else 'allocator'
            other=next(r for r in rows if (r['image'],r['dataset'],r['label'])==(row['image'],row['dataset'],base_label))
            assert all(row[k]==other[k] for k in ('stream_sha256','pixels_sha256','patch_signatures','mse','bpp'))
            for key in ('policy_actions','extra_representations','original_raw_bits','selected_raw_bits',
                'original_local_sse_gain','selected_local_sse_gain','representations_changed','policy_stops_preserved'):
                assert row['experimental_stats'].get(key)==other['experimental_stats'].get(key)
        if 'alloc' in row['optimization'] and not row['optimization'].startswith('all_'):
            assert 'rectangle_updates' not in row['experimental_stats']
    save(name+'_summary',summaries(raw))
    labels={r['label'] for r in rows}
    if name.endswith('fast'):
        pairs=[('fast_allocator','fast_ordinary'),('fast_optimized','fast_ordinary'),('fast_optimized','fast_allocator')]
    else:
        pairs=[('allocator','ordinary')]+[(v,'allocator') for v in ('projection','before_sse','optimized') if v in labels]
    comparisons=pair(raw,pairs,name+'_paired')
    result=[]
    for candidate,baseline in pairs:
        for dataset in ('hpt_data','hpt_data_val'):
            group=[r for r in comparisons if (r['candidate'],r['baseline'],r['dataset'])==(candidate,baseline,dataset)]
            if not group:
                continue
            a=statistics.mean(r['encode_seconds_baseline'] for r in group)
            b=statistics.mean(r['encode_seconds_candidate'] for r in group)
            result.append(dict(dataset=dataset,candidate=candidate,baseline=baseline,cases=len(group),
                baseline_seconds=a,candidate_seconds=b,encode_pct=100*(b/a-1),
                wins=sum(r['encode_seconds_candidate']<r['encode_seconds_baseline'] for r in group),
                bytes_equal=sum(r['bytes_equal'] for r in group),pixels_equal=sum(r['pixels_equal'] for r in group),patches_equal=sum(r['patches_equal'] for r in group)))
    save(name+'_aggregate',result)
    if name.endswith('profile'):
        groups=defaultdict(list)
        for row in raw:
            for function in row['profile']:
                groups[(row['label'],function['file'],function['function'])].append(function)
        profile=[dict(label=label,file=file,function=fn,observations=len(group),
            self_seconds=statistics.mean(r['self_seconds'] for r in group),
            cumulative_seconds=statistics.mean(r['cumulative_seconds'] for r in group),
            calls=statistics.mean(r['calls'] for r in group)) for (label,file,fn),group in groups.items()]
        save('allocator_profile_functions',profile)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode')
    args=parser.parse_args()
    analyze(args.mode) if args.mode.startswith('efficiency_') else plan(args.mode)
