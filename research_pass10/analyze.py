import hashlib
import json
from pathlib import Path
from statistics import mean,stdev
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
PRESETS=('compression','balanced','quality','high_quality')
LABELS=dict(compression='Compression',balanced='Balanced',quality='Quality',high_quality='High Quality',overall='Overall')


def main():
    path=HERE/'public_benchmark.json'
    data=json.loads(path.read_text())
    rows=data['rows']; assert len(rows)==240
    selected=data['selected_run']; run=next(r for r in data['runs'] if r['run_id']==selected)
    meta=run['metadata']
    assert run['coverage_complete'] and run['status']=='complete'
    assert meta['baseline_id']=='pbc31_release_rng_baseline' and meta['rng_mode']=='release'
    assert meta['n_trials']==3 and meta['codecs']==['PBC3'] and len(meta['images'])==20
    keys={(r['trial'],r['variant'],r['image']) for r in rows}
    assert keys=={(t,p,i) for t in range(1,4) for p in PRESETS for i in meta['images']}
    assert len(keys)==len(rows) and {r['codec'] for r in rows}=={'PBC3'}
    frozen=json.loads((ROOT/'research_pass9/comparison_summary.json').read_text())
    reference={(r['image'],r['preset']):r['release'] for r in frozen['equivalence'] if r['dataset']=='hpt_data'}
    cases=[]
    for image in meta['images']:
        for preset in PRESETS:
            group=[r for r in rows if (r['image'],r['variant'])==(image,preset)]
            expected=reference[image,preset]
            assert len(group)==3
            assert all(r['stream_sha256']==expected['stream_sha256'] and r['patches']==expected['patches'] for r in group)
            assert all(abs(r['bpp']-expected['bpp'])<1e-12 and abs(r['mse']-expected['mse'])<max(1e-4,expected['mse']*1e-6) for r in group)
            row=dict(image=image,preset=preset,stream_sha256=expected['stream_sha256'],patches=expected['patches'],mse=group[0]['mse'],bpp=group[0]['bpp'])
            for metric in ('encode_seconds','decode_seconds'):
                samples=[r[metric] for r in group]
                row[metric]=mean(samples); row[metric+'_samples']=samples
                row[metric+'_cv_percent']=100*stdev(samples)/mean(samples)
            cases.append(row)
    metrics=('encode_seconds','decode_seconds','mse','bpp','patches')
    summaries={preset:{metric:mean(r[metric] for r in rows if preset=='overall' or r['variant']==preset) for metric in metrics} for preset in PRESETS+('overall',)}
    result=dict(run_id=selected,hf_revision=json.loads((HERE/'hf_deployment.json').read_text())['commit'],metadata=meta,
        rows=240,distinct_cases=80,release_identical_files=240,coverage_complete=True,only_pbc=True,summaries=summaries,cases=cases,
        timings='Ordinary public runner arithmetic means over all three trials and 20 images; full calls, including Image.fromarray in encode; input dataset loading/warmup and external scoring/hash/patch parsing excluded. Internal MSE enabled. No CPU affinity or numerical-thread constraint, no per-preset interleaved source comparison.',
        source_response_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (HERE/'benchmark_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    source_revision='8006f30debfc4a57289df06df21636aeb52baffd'
    for name,value in meta['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',source_revision+':'+name])).hexdigest()==value,name
    variants=json.loads((HERE/'variant_checks.json').read_text())
    for row in variants['rows']:
        previous=next(x for x in frozen['rows'] if (x['source'],x['image'],x['preset'])==(row['source'],row['image'],row['preset']))
        assert all(row[k]==previous[k] for k in ('stream_sha256','pixels_sha256','mse','bpp','patches'))
    workers=variants['workers']
    baseline=dict(baseline_id='pbc31_release_rng_baseline',github_source_revision=source_revision,
        github_source_url='https://github.com/EgeEken/PBC/commit/'+source_revision,hf_revision=result['hf_revision'],
        immutable_release_revision='2e59f1b06980aa467da0cde94fa33c3f9a0d4d51',model_sha256=meta['model_sha256'],
        rng='numpy.random.default_rng(config.random_seed)',default_seed=2003,owned_rng_per_encode=True,
        historical_custom_control=dict(id='pbc3_custom_rng_historical',rng_mode='current',worker_variant='dev_current_rng'),
        presets=workers['dev_release_rng']['presets'],environment=workers['dev_release_rng']['versions'],
        working_resolution='Aspect-preserving automatic working image capped at 250000 pixels; smaller images remain native; reconstruction returns original size.',
        stream_version=0,decoder_unchanged=True,internal_mse_default=True,source_sha256=meta['source_sha256'],
        runtime_optimizations=['vectorized grid I/O','per-encode exact search state','selected-delta reuse','contiguous float32 PIL resampling buffers','model-weight caching'],
        prior_equivalence=dict(github_revision='e6c8eca726b3048a92514bd09d31fca0b3e00fd7',hf_revision='4237d954f4e1a04951c822c738113a53bbd58e9b',cases=100),
        controlled_pass9_time_reductions_percent=dict(encode=28.83,decode=64.38),
        public_benchmark=dict(run_id=selected,rows=240,release_identical_files=240,summaries=summaries),
        official_release=False)
    (HERE/'baseline_manifest.json').write_text(json.dumps(baseline,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['| Preset | Mean encode s | Mean decode s | RGB MSE | Final bpp | Mean patches |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for p,s in summaries.items():
        lines.append('| '+LABELS[p]+' | '+' | '.join(f'{s[m]:.6f}' for m in metrics)+' |')
    (HERE/'speed_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Run',selected,': all 240 files match the 80 frozen HF release cases; complete PBC-only coverage.')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
