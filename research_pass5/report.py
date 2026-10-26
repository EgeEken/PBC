import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median, stdev
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def main():
    run=json.loads((HERE/'hf_representative.json').read_text())
    rows=run['rows']
    assert run['exact'] and len(rows)==288
    with (ROOT/'research_pass4'/'runtime_full.csv').open(newline='',encoding='utf-8') as f:
        prior={(r['image'],r['preset']):r for r in csv.DictReader(f) if r['label']=='baseline' and r['dataset']=='hpt_data'}
    reference_checks=[]
    for r in rows:
        if r['variant']=='baseline' and r['trial']==0:
            old=prior[(r['image'],r['preset'])]
            reference_checks.append(dict(image=r['image'],preset=r['preset'],
                stream_equal=r['stream_sha256']==old['stream_sha256'],
                pixels_equal=r['pixels_sha256']==old['pixels_sha256'],
                mse_equal=r['mse']==float(old['mse'])))
    baseline_prior_matches=sum(r['stream_equal'] and r['pixels_equal'] and r['mse_equal'] for r in reference_checks)
    (HERE/'prior_baseline_equivalence.json').write_text(json.dumps(reference_checks,indent=2)+'\n')
    keys=sorted({(r['image'],r['preset']) for r in rows})
    cases=[]
    for image,preset in keys:
        selected={v:[r for r in rows if (r['image'],r['preset'],r['variant'])==(image,preset,v)] for v in ('baseline','all_mse','all')}
        baseline=selected['baseline'][0]
        for group in selected.values():
            assert len(group)==3
            for r in group:
                for k in ('stream_sha256','pixels_sha256','mse','bpp','final_bytes','initialization_patches','iterative_patches'):
                    assert r[k]==baseline[k],(image,preset,k)
        assert all(r['internal_mse']==baseline['internal_mse'] for r in selected['all_mse'])
        for variant,group in selected.items():
            item=dict(image=image,preset=preset,variant=variant,mse=baseline['mse'],bpp=baseline['bpp'],final_bytes=baseline['final_bytes'])
            for metric in ('encode_seconds','decode_seconds'):
                values=[r[metric] for r in group]
                control=median(r[metric] for r in selected['baseline'])
                item[metric]=median(values)
                item[metric+'_baseline']=control
                item[metric+'_change_percent']=100*(median(values)/control-1)
                item[metric+'_cv_percent']=100*stdev(values)/mean(values)
            cases.append(item)
    summary={}
    for scope in ['all','compression','balanced','quality','high_quality']:
        summary[scope]={}
        for variant in ('baseline','all_mse','all'):
            group=[r for r in cases if r['variant']==variant and (scope=='all' or r['preset']==scope)]
            item=dict(cases=len(group))
            for metric in ('encode_seconds','decode_seconds'):
                value=mean(r[metric] for r in group)
                base=mean(r[metric+'_baseline'] for r in group)
                item[metric]=value
                item[metric+'_baseline']=base
                item[metric+'_change_percent']=100*(value/base-1)
                item[metric+'_wins']=sum(r[metric]<r[metric+'_baseline'] for r in group)
                item[metric+'_mean_cv_percent']=mean(r[metric+'_cv_percent'] for r in group)
                item[metric+'_worst_change_percent']=max(r[metric+'_change_percent'] for r in group)
                calls=[r for r in rows if r['variant']==variant and (scope=='all' or r['preset']==scope)]
                controls=[r for r in rows if r['variant']=='baseline' and (scope=='all' or r['preset']==scope)]
                item[metric+'_all_calls_mean']=mean(r[metric] for r in calls)
                item[metric+'_all_calls_mean_change_percent']=100*(mean(r[metric] for r in calls)/mean(r[metric] for r in controls)-1)
                item[metric+'_trial_mean_change_percent']={str(t):100*(mean(r[metric] for r in calls if r['trial']==t)/mean(r[metric] for r in controls if r['trial']==t)-1) for t in range(3)}
            summary[scope][variant]=item
    (HERE/'timing_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (HERE/'paired_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    for name,values in [('paired_cases',cases),('hf_rows',rows)]:
        with (HERE/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,dict) else v for k,v in r.items()} for r in values)
    deployment=json.loads((HERE/'deployment.json').read_text())
    versions=run['metadata']['versions']
    peak=max(r['peak_process_rss_kib'] for r in rows)/1024
    allsummary=summary['all']
    lines=['# Existing HF Space runtime benchmark results','',
        'Outcome: Validated and ready to promote through a separate production integration. The optimizations ran on the actual existing HF cpu-basic deployment. Public codec defaults and UI remain unchanged. The current deployment exposes the optimized implementation only through the authenticated isolated benchmark worker.','',
        f"Deployed revision: `{deployment['commit']}` on [EgeEken/PBC](https://huggingface.co/spaces/EgeEken/PBC). Both HF repository metadata and the running runtime revision were confirmed before and after measurement; health passed. Parent/rollback baseline: `{deployment['base_revision']}`.",'',
        '## Measured result','',
        'The table uses the mean of 32 image/preset medians, each from three interleaved repeats. Negative percentages mean less time. These are complete server-side compress/decompress call walls, including search state creation, controlled-source compilation, updates and cleanup. File loading, independent scoring, IPC and HTTP are outside the timers.','',
        '| Variant | Encode seconds | Encode change | Decode seconds | Decode change | Encode/decode wins |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    labels=dict(baseline='Unchanged production baseline',all_mse='Grid + search, internal MSE retained',all='Grid + search, internal MSE skipped')
    for v,s in allsummary.items():
        lines.append(f"| {labels[v]} | {s['encode_seconds']:.6f} | {s['encode_seconds_change_percent']:+.2f}% | {s['decode_seconds']:.6f} | {s['decode_seconds_change_percent']:+.2f}% | {s['encode_seconds_wins']}/{s['decode_seconds_wins']} of 32 |")
    lines += ['', 'Public-API-compatible acceleration is the retained-MSE row. The optional skipped-MSE row is suitable only for callers that do not need that internal metric. Both are directly measured; separate speed percentages are not added.','',
        '| Preset | Retained MSE encode change | Skipped MSE encode change | Retained/skipped decode change |',
        '| --- | ---: | ---: | ---: |']
    for p in ('compression','balanced','quality','high_quality'):
        a,b=summary[p]['all_mse'],summary[p]['all']
        lines.append(f"| {p} | {a['encode_seconds_change_percent']:+.2f}% | {b['encode_seconds_change_percent']:+.2f}% | {a['decode_seconds_change_percent']:+.2f}% / {b['decode_seconds_change_percent']:+.2f}% |")
    lines += ['', 'Mean within-case repeat coefficient of variation:', '', '| Variant | Encode CV | Decode CV |', '| --- | ---: | ---: |']
    for v,s in allsummary.items():
        lines.append(f"| {v} | {s['encode_seconds_mean_cv_percent']:.2f}% | {s['decode_seconds_mean_cv_percent']:.2f}% |")
    lines += ['', 'Variation is substantial (roughly 17% encode and 18–20% decode CV), so the percentages should be treated as this bounded run’s estimates. The gain direction is consistent: both optimized modes win all 32 case medians in both metrics and all three complete-trial means.', '',
        '| Variant | All-call encode mean s / change | All-call decode mean s / change | Per-trial mean encode changes |',
        '| --- | ---: | ---: | --- |']
    for v,s in allsummary.items():
        trials=', '.join(f"{x:+.2f}%" for x in s['encode_seconds_trial_mean_change_percent'].values())
        lines.append(f"| {v} | {s['encode_seconds_all_calls_mean']:.6f} / {s['encode_seconds_all_calls_mean_change_percent']:+.2f}% | {s['decode_seconds_all_calls_mean']:.6f} / {s['decode_seconds_all_calls_mean_change_percent']:+.2f}% | {trials} |")
    lines += ['', 'Largest encode regressions or smallest improvements:', '']
    for v in ('all_mse','all'):
        worst=sorted((r for r in cases if r['variant']==v),key=lambda r:r['encode_seconds_change_percent'],reverse=True)[:3]
        lines.append('- '+v+': '+', '.join(f"{r['image']} / {r['preset']} {r['encode_seconds_change_percent']:+.2f}%" for r in worst)+'.')
    lines += ['', '## Correctness and service behavior','',
        'All 288 representative rows and the six initial HF smoke rows pass exact bitstream, decoded RGB pixels, independently measured original-size MSE, bpp and patch-count checks within the actual HF runtime. Cross-repeat determinism is also checked in the final analysis. Retained internal MSE values match baseline; skipped internal values are None. Final compressed bytes include the full post-LZMA file, not raw stream size. No quality/rate tradeoff is introduced by either optimization.','',
        f"A separate cross-environment comparison matches {baseline_prior_matches}/32 HF baseline streams/pixels/MSE against Pass 4 Windows baseline records; {32-baseline_prior_matches} differ. These are not mismatches between baseline and optimized variants on HF. Source/model/input hashes pass. Pass 1 already documented a current-code Windows-versus-HF trajectory difference on googlewiki Quality (PBC_BENCHMARK_VALIDATION.md, deployment_comparison.json), so cross-runtime identity was not an established baseline guarantee. The rebuild resolved unpinned NumPy/Numba versions as well. The exact cause of each new mismatch is not isolated; platform/library numerical and ordering behavior are hypotheses. Do not attribute it to the optimizations or claim it is harmless numerical noise. Do not compare local and HF files as an optimization pair. See prior_baseline_equivalence.json; the within-HF paired result remains exact.",'',
        'The public /api/compress check used the existing deployed googlewiki.png and Balanced preset. That input is opaque RGBA, and the existing public handler preserves its alpha channel; benchmark inputs deliberately convert to RGB. The returned RGBA stream matches both a local encode with the unchanged public configuration and the actual predeployment Pass 1 HF Balanced stream (SHA256 2f7ef0a8dcdd22444a65ba03f3e2955dde57844d36db7fa269567ca38bcd9414, 51,493 bytes). Its PNG preview equals independently decoded RGBA pixels, and original dimensions, rounded compressed file size and displayed four-channel MSE are correct. Public compression handler/configs are unchanged. This is a service check, not an HTTP-based speed measurement. No new input was introduced. The optimized remote timing set is RGB only; the public RGBA service check exercises unchanged production code, while the prior local RGBA optimization checks are reused.','',
        'Two public requests were made. Initial local verification incorrectly interpreted the preview data-URL prefix, then incorrectly compared the public RGBA stream with the RGB benchmark stream. The checker was corrected and the saved second response was fully verified without another request. Remote codec output was correct; no deployed-source fix was needed. These local-client failures lost their in-memory random benchmark keys and led to two extra secret rotations/restarts. The successful HF smoke is retained and not rerun. client_correction.json records this operational issue.','',
        'Remote access checks returned 403 for absent/wrong research tokens and 400 for an unallowlisted path. The valid static variants were selectable and ran. The research start branch is disabled without its Space secret. No arbitrary paths, configs, code or modules can be supplied. Existing read-only status and ordinary benchmark access policies are retained.','',
        '## Configuration and limits','',
        'Eight predetermined existing main-set images: cat.jpg, monet.jpg, flower.jpg, cathedral.png, googlewiki.png, stockpoint.jpg, mario64.png, eiffel.png. All four unchanged presets; RNG seed 2003; same production model and image bytes, verified by manifest before execution. No allocation, RD, entropy or training experiment.','',
        f"Representative measurement: one worker PID {run['metadata']['worker_pid']}, pinned affinity {run['metadata']['affinity']}, numerical thread variables all 1. Python {run['metadata']['python'].splitlines()[0]}; NumPy {versions['numpy']}, Numba {versions['numba']}, Pillow {versions['Pillow']}. HF reports cpu-basic and one replica. The exact physical CPU model is not exposed by platform.processor on this run; hardware-tier provenance is HF metadata.",'',
        'Each variant/preset receives one complete encode/decode warmup on googlewiki.png in the same worker (12 full warmups). Measured order rotates per image, preset and trial so every case visits each variant position across three repeats. The initial separate one-image Balanced smoke has six measured calls plus three warmups. No complete previous local suite was repeated: assembled-candidate local smoke has six rows plus three warmups, compile checks, route/auth/allowlist checks and source equivalence checks.','',
        f"Observed whole-worker cumulative peak RSS reaches {peak:.1f} MiB. ru_maxrss is a process high-water mark shared across sequential variants; it cannot identify variant-specific incremental memory. Previous tracked vector grid temporary sizes remain relevant; this run shows no OOM or worker failure, but does not prove equal production peak memory. The worker's first warmup includes cache-loading/first-call effects; this is a warmed throughput/latency comparison, not a pristine cold-start benchmark.",'',
        'Existing benchmark locking excludes simultaneous benchmark/sweep/training jobs; before/after status snapshots are retained. The public Space remains online, so unobserved user traffic and shared-host scheduling are possible sources of variation. Three repeats and interleaving limit this issue; they do not create dedicated hardware. No significant compatibility or reliability failure was observed in the tested calls.','',
        '## Decision and promotion boundary','',
        'The measured HF result supports promoting exact grid I/O and search-state reuse while retaining internal MSE for the public API. Metric skipping remains an explicit private/file-only option. This task does not switch the production default: the user explicitly required public defaults and behavior to remain unchanged.','',
        'The benchmark uses scoped monkeypatches in an isolated process. That installation mechanism must not be copied into a concurrent public request handler: a production integration should use per-encode owned search state and an explicit grid implementation path. The underlying exact runtime optimizations are validated; safe public integration remains a small separate reviewed code change. No allocator or PBC 3.1 all-metric release claim follows from this runtime result.','',
        '## Deployment, rollback and repository preservation','',
        'Exactly six reviewed files were committed: server.py, pbc3_benchmark.py, research_pass5/hf_runtime_job.py, research_pass5/hf_candidate_grid.py, research_pass5/hf_candidate_search.py, research_pass5/hf_runtime_manifest.json. Candidate grid/search modules are byte-identical to Pass 4; the worker extends only the required static modes/image/preset limits and adds RSS/PID reporting. The endpoint/backend reuse the prepared Pass 4 proposal, fix clearing the stop event before its running check, and return a bounded worker error for diagnosis. No dependency, model, dataset, default, UI, production codec or DB file is in the commit.','',
        'The final private benchmark token is a domain-separated HMAC derived in memory from the existing HF credential and deployment revision. Its value is installed through the Space secret API and is never printed, saved locally or committed; the HF credential itself is never sent to the app. An authorized local client can reproduce the scoped token after a retry without unnecessary rotations. The secret remains enabled; rotate/remove it to revoke/disable research starts. Secret configuration and the push restart/rebuild the Space; health and revision were verified afterward.','',
        f"Rollback: create a clean branch based on current HF main, run `git revert {deployment['commit']}`, then push that revert to HF main. Do not force-push or reset user work. Optionally delete PBC_RESEARCH_BENCHMARK_TOKEN to disable research access. The previous source is parent {deployment['base_revision']}; no persistent benchmark DB changes need reversal.",'',
        'The deployment commit was constructed from HF main with an isolated index and recorded on local codex/pass5-hf-runtime, then pushed to HF main. The original multistage_pbc31 branch and its index stay in place; its unrelated experimental commits were not pushed. The two authorized backend edits remain visible in the original working checkout relative to its original HEAD. Unrelated .gitignore and user files remain untouched. Final status and preservation checks are supplied separately.','',
        '## Evidence','',
        '- hf_representative.json / hf_rows.csv: all raw measured rows, configs, timestamps in job records, source/model/input hashes, dependency versions, warmups and worker provenance.',
        '- hf_smoke.json, public_api_smoke.json, remote_auth_checks.json: bounded remote smoke, public API and access-control checks.',
        '- timing_summary.json, paired_cases.json / .csv: aggregate, per-preset, per-case medians, repeat variation and outliers.',
        '- deployment.json, deployment.patch, deployment_files.json, push.json: exact six-file commit, source diff, target and rollback.',
        '- hf_before.json, hf_deployed.json, hf_after.json, deployment_observations.json, health_deployed.json, service_before.json, service_after.json: revision, hardware and service state.',
        '- DEPLOYMENT_REVIEW.md, candidate_checks.json, local_smoke.json: reviewed authorization scope and focused local verification.',
        '- changed_files.txt, artifact_manifest.json, final_checks.json, git_status_after.txt: inventory and preservation evidence.','']
    (HERE/'PBC_HF_BENCHMARK_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(allsummary,indent=2))


if __name__=='__main__':
    main()
