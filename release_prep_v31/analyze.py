from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics as st
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'
MODES=('original','published','candidate')
PRESETS=('compression','balanced','quality','high_quality')
rows=[json.loads(s) for s in (OUT/'measurements.jsonl').read_text().splitlines()]
assert len(rows)==900
groups=defaultdict(list)
for row in rows:
    assert row['error'] is None
    groups[row['image'],row['preset'],row['variant']].append(row)
assert len(groups)==300 and all(len(rs)==3 for rs in groups.values())
exact_keys=('stream_sha256','pixel_sha256','bytes','bpp','rgb_mse','internal_mse','patch_count','mode','size')
cases=[]
for image,preset in sorted({(r['image'],r['preset']) for r in rows}):
    case=dict(image=image,preset=preset,variants={})
    for mode in MODES:
        rs=groups[image,preset,mode]
        assert sorted(r['repeat'] for r in rs)==[0,1,2]
        for key in exact_keys:
            assert all(r[key]==rs[0][key] for r in rs),('nondeterministic',image,preset,mode,key)
        case['variants'][mode]={key:rs[0][key] for key in exact_keys}
        for field in ('encode_seconds','decode_seconds'):
            values=[r[field] for r in rs]
            case['variants'][mode][field]=dict(median=st.median(values),mean=st.mean(values),
                minimum=min(values),maximum=max(values),cv_percent=100*st.pstdev(values)/st.mean(values))
    case['differences']=[key for key in exact_keys if any(case['variants'][m][key]!=case['variants']['original'][key] for m in MODES[1:])]
    cases.append(case)
def aggregate(subset):
    result={}
    for field in ('encode_seconds','decode_seconds'):
        means={m:st.mean(c['variants'][m][field]['median'] for c in subset) for m in MODES}
        comparisons={}
        for old,new in (('original','published'),('published','candidate'),('original','candidate')):
            reductions=[dict(image=c['image'],preset=c['preset'],
                reduction_percent=100*(1-c['variants'][new][field]['median']/c['variants'][old][field]['median']))
                for c in subset]
            comparisons[new+'_vs_'+old]=dict(reduction_percent=100*(1-means[new]/means[old]),
                improved=sum(c['reduction_percent']>0 for c in reductions),
                regressed=sum(c['reduction_percent']<0 for c in reductions),
                largest_regressions=sorted(reductions,key=lambda c:c['reduction_percent'])[:5])
        result[field]=dict(mean_case_medians=means,comparisons=comparisons,
            mean_cv_percent={m:st.mean(c['variants'][m][field]['cv_percent'] for c in subset) for m in MODES},
            max_cv_percent={m:max(c['variants'][m][field]['cv_percent'] for c in subset) for m in MODES})
    result['quality']={m:{k:st.mean(c['variants'][m][k] for c in subset) for k in ('rgb_mse','bpp','bytes')} for m in MODES}
    return result
summary=dict(measured_pairs=len(rows),case_count=len(cases),exact_cases=sum(not c['differences'] for c in cases),
    overall=aggregate(cases),presets={p:aggregate([c for c in cases if c['preset']==p]) for p in PRESETS},
    datasets={d:aggregate([c for c in cases if c['image'].split('/')[0]==d]) for d in ('hpt_data','hpt_data_val')},
    images={i:aggregate([c for c in cases if c['image']==i]) for i in sorted({c['image'] for c in cases})},
    cases=cases,raw_sha256=hashlib.sha256((OUT/'measurements.jsonl').read_bytes()).hexdigest())
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# V3.1 definitive local benchmark','',
'25 manifest-verified images × four presets × three frozen implementations × three measured repetitions: 900 complete encode and 900 complete decode calls. Additional full-call warmups precede every image/preset/variant. Timed calls include final MSE and entropy coding. Image normalization/loading, independent scoring, hashing, IPC and I/O are excluded. Persistent source-isolated Python 3.11.16 workers share pinned dependency versions and logical CPU 10 on the i9-12900H; numerical/Numba threads are one. Variant order rotates every repetition and case. No build or codec validation workload ran concurrently.',
'',
'Windows topology identifies CPU 10 as an SMT P-core (sibling 11, efficiency class 1). CPU 12 used in older reports is an E-core here. Topology is retained in provenance.json. These are local timings; no HF or cross-machine claim.',
'',
'Times below average the three-repetition median for each image/preset case, weighting all 100 cases equally. Reduction = 100 × (1 − new / old). Encode/decode columns report seconds.',
'',
'| Scope | 3.0 enc | Published 3.1 enc | Final enc | 3.0 dec | Published 3.1 dec | Final dec |',
'| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for scope,data in [('Overall',summary['overall']),*summary['presets'].items()]:
    v=[data[f]['mean_case_medians'][m] for f in ('encode_seconds','decode_seconds') for m in MODES]
    lines.append('| '+scope+' | '+' | '.join(f'{x:.6f}' for x in v)+' |')
lines+=['','| Comparison | Encode time reduction | Decode time reduction | Encode wins/regressions | Decode wins/regressions |',
'| --- | ---: | ---: | ---: | ---: |']
for key,label in [('published_vs_original','Published 3.1 versus 3.0'),('candidate_vs_published','Final versus published 3.1'),('candidate_vs_original','Final versus 3.0')]:
    e=summary['overall']['encode_seconds']['comparisons'][key];d=summary['overall']['decode_seconds']['comparisons'][key]
    lines.append(f"| {label} | {e['reduction_percent']:.2f}% | {d['reduction_percent']:.2f}% | {e['improved']}/{e['regressed']} | {d['improved']}/{d['regressed']} |")
lines+=['',f"All {summary['exact_cases']}/100 cases have exact three-way streams, pixels, dimensions, mode, size, bpp, patch count and internal/external MSE. All 300 implementation/cases are deterministic across repetitions. Original release RNG/configuration semantics were retained; no equality was manufactured.",
'', '| Image | 3.0 enc | Published enc | Final enc | 3.0 dec | Published dec | Final dec |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for image,data in summary['images'].items():
    v=[data[f]['mean_case_medians'][m] for f in ('encode_seconds','decode_seconds') for m in MODES]
    lines.append('| '+image+' | '+' | '.join(f'{x:.6f}' for x in v)+' |')
lines+=['','Within-case CV (population standard deviation / mean), averaged over cases:']
for field in ('encode_seconds','decode_seconds'):
    data=summary['overall'][field]
    lines.append('- '+field+': '+', '.join(f"{m} {data['mean_cv_percent'][m]:.2f}% (maximum {data['max_cv_percent'][m]:.2f}%)" for m in MODES)+'.')
    for key in ('published_vs_original','candidate_vs_published','candidate_vs_original'):
        worst=data['comparisons'][key]['largest_regressions']
        lines.append('  '+key+' lowest reductions: '+', '.join(f"{r['image']}/{r['preset']} {r['reduction_percent']:.2f}%" for r in worst)+'.')
lines+=['','Detailed case medians, per-preset comparisons, dataset aggregates, variability and regressions: artifacts/summary.json. Full repetition-level configs/hashes/statistics: artifacts/measurements.jsonl (SHA-256 '+summary['raw_sha256']+'). Warmups and worker module origins are retained separately. Negative reductions are timing regressions; three repetitions do not establish statistical significance. Sources: provenance.json and prepare.py.']
(HERE/'BENCHMARK.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:summary['overall'][k] for k in ('encode_seconds','decode_seconds')},indent=2))
