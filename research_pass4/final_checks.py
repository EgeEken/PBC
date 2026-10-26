import json
from pathlib import Path
import py_compile
import subprocess

from support import ROOT, OUT, PASS3, read, save, sha, snapshot
from analyze import collapse


def main():
    before=read('initial_snapshot')
    after=snapshot()
    assert before['hashes']==after['hashes']
    assert before['commit']==after['commit']
    assert before['status']==after['status']
    prior=read('previous_artifacts_manifest')
    assert all(sha((ROOT/p).read_bytes())==value for p,value in prior.items())
    expected={'runtime_exactness':200,'runtime_screen':288,'runtime_full':1200,
        'matched_controls':106,'matched_confirmation':150,'efficiency_profile':24,
        'efficiency_exactness':24,'efficiency_screen':90}
    promotion=read('efficiency_promotion')
    if promotion['full']:
        expected['efficiency_full']=225
    if promotion['fast']:
        expected['efficiency_fast']=225
    frozen={(r['dataset'],r['image']):r for r in json.loads((PASS3/'allocation_full.json').read_text())
        if r['trial']==0 and r['label']=='q500p35_integrated_t0.4'}
    counts={}
    streams={}
    for name,count in expected.items():
        rows=read(name)
        assert len(rows)==count,(name,len(rows),count)
        counts[name]=len(rows)
        for r in rows:
            assert r['preview_equals_decode']
            assert r['bpp']==8*r['final_bytes']/(r['width']*r['height'])
            assert r['patch_count']==r['initialization_patches']+r['iterative_patches']
            assert (r['internal_mse'] is None)==(not r['config']['compute_final_mse'])
            assert r['model_sha256']==before['hashes']['patch_policy.npz']
            assert r['input_sha256']==before['hashes'][str(Path(r['dataset'])/r['image'])]
            if r.get('stream_path'):
                path=OUT/r['stream_path']
                assert sha(path.read_bytes())==r['stream_sha256']
                assert path.stat().st_size==r['final_bytes']
                streams[str(path.relative_to(OUT))]=r['stream_sha256']
            if 'alloc' in r['optimization']:
                if not r['optimization'].startswith('all_'):
                    assert 'rectangle_updates' not in r['experimental_stats']
                control=frozen[(r['dataset'],r['image'])]
                assert all(r[k]==control[k] for k in ('stream_sha256','pixels_sha256','patch_signatures','mse','bpp'))
                for k in ('policy_actions','extra_representations','original_raw_bits','selected_raw_bits',
                    'original_local_sse_gain','selected_local_sse_gain','representations_changed','policy_stops_preserved'):
                    assert r['experimental_stats'].get(k)==control['experimental_stats'].get(k),(name,r['image'],k)
        assert all(r['deterministic'] for r in collapse(rows))
        metadata_path=OUT/(name+'_metadata.json')
        if metadata_path.exists():
            workers=json.loads(metadata_path.read_text())['workers']
            assert all(len(m['affinity'])==1 and all(v=='1' for v in m['threads'].values()) for m in workers.values())
    for name in ('runtime_exactness','runtime_screen','runtime_full'):
        rows=read(name)
        baseline={(r['dataset'],r['image'],r['preset'],r['trial']):r for r in rows if r['label']=='baseline'}
        for r in rows:
            other=baseline[(r['dataset'],r['image'],r['preset'],r['trial'])]
            assert all(r[k]==other[k] for k in ('stream_sha256','pixels_sha256','patch_signatures','mse','bpp'))
            assert r['frozen_control_equal']
    assert len(read('runtime_edges')['rows'])==16
    assert all(r['byte_identical_to_frozen'] for r in read('regression_diagnostics'))
    assert max(r['fresh_points'] for r in read('matched_rate_comparisons'))<=6
    assert len(read('matched_rate_comparisons'))==25
    assert all(r['optimization']=='none' for r in read('matched_controls'))
    assert read('local_hf_packaging_smoke')['exact']
    assert read('local_hf_packaging_smoke')['metadata']['label']=='LOCAL_PACKAGING_SMOKE'
    candidate_manifest=read('hf_runtime_manifest')
    assert all(sha((ROOT/p).read_text(encoding='utf-8').encode())==value for p,value in candidate_manifest['source_lf'].items())
    subprocess.run(['git','apply','--check',str(OUT/'proposed_hf_benchmark.patch')],cwd=ROOT,check=True,capture_output=True)
    for p in OUT.rglob('*.py'):
        py_compile.compile(str(p),doraise=True)
    for name in ('PBC_HF_RUNTIME_VALIDATION.md','PBC_ALLOCATOR_MATCHED_RATE.md','PBC_ALLOCATOR_EFFICIENCY.md','PBC_PASS4_SUMMARY.md'):
        assert (OUT/name).stat().st_size>1000
    save('final_snapshot',after)
    (OUT/'git_status_after.txt').write_text(after['status'])
    validation=dict(production_model_datasets_unchanged=True,previous_artifacts_verified=len(prior),
        previous_artifacts_unchanged=True,head_unchanged=after['commit'],preexisting_status_preserved=True,
        measured_rows=counts,saved_stream_files_verified=len(streams),frozen_allocator_bytes_decisions_preserved=True,
        integration_patch_applied=False,integration_patch_check_passed=True,remote_codec_benchmarks=0,
        remote_activity='Read-only public health/schema/status/source GETs; no POSTs or uploads',
        production_changes=0,commits=0,pushes=0,deployments=0,restarts=0,uploads=0,training=0)
    save('final_validation',validation)
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.mpl-cache' not in p.parts)
    inventory=sorted({str(p.relative_to(ROOT)) for p in paths}|{'research_pass4\\changed_files.txt','research_pass4\\artifact_manifest.json'})
    (OUT/'changed_files.txt').write_text('\n'.join(inventory)+'\n')
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.mpl-cache' not in p.parts and p.name!='artifact_manifest.json')
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in paths}
    save('artifact_manifest',manifest)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
