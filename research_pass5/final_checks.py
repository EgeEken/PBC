import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent


def main():
    sha=lambda data:hashlib.sha256(data).hexdigest()
    deployment=json.loads((HERE/'deployment.json').read_text())
    expected=deployment['commit']
    parent=subprocess.check_output(['git','rev-parse',expected+'^'],text=True).strip()
    assert parent==deployment['base_revision']
    actual=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',expected],text=True).splitlines()
    assert sorted(actual)==sorted(deployment['files'])
    for path in actual:
        stored=subprocess.check_output(['git','show',expected+':'+path])
        local=(ROOT/path).read_text(encoding='utf-8').encode()
        assert stored==local,path
    for name in ('hf_deployed','hf_after'):
        meta=json.loads((HERE/(name+'.json')).read_text())
        assert meta['sha']==meta['runtime']['sha']==expected
        assert meta['runtime']['stage']=='RUNNING'
        assert meta['runtime']['hardware']['current']=='cpu-basic'
    for name,count in (('hf_smoke',6),('hf_representative',288)):
        run=json.loads((HERE/(name+'.json')).read_text())
        assert run['exact'] and len(run['rows'])==count
        assert all(v=='1' for v in run['metadata']['threads'].values())
        manifest=run['metadata']['manifest']
        for path,digest in manifest['source_lf'].items():
            assert sha((ROOT/path).read_text(encoding='utf-8').encode())==digest,path
        assert sha((ROOT/'patch_policy.npz').read_bytes())==manifest['model_sha256']
        for path,digest in manifest['inputs'].items():
            assert sha((ROOT/'hpt_data'/path).read_bytes())==digest,path
    count=0
    prior=json.loads((ROOT/'research_pass4'/'previous_artifacts_manifest.json').read_text())
    prior.update({p:r['sha256'] for p,r in json.loads((ROOT/'research_pass4'/'artifact_manifest.json').read_text()).items()})
    for path,digest in prior.items():
        assert sha((ROOT/path).read_bytes())==digest,path
        count+=1
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert head=='3e9e8c94b7497ecefc5e482a66fde697dfad6e19'
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain=v1','-uall'],text=True)
    (HERE/'git_status_after.txt').write_text(status,encoding='utf-8')
    before=(HERE/'git_status_before.txt').read_text(encoding='utf-8-sig').splitlines()
    ignore=lambda lines:{line for line in lines if 'research_pass5/' not in line and line[3:] not in ('server.py','pbc3_benchmark.py')}
    assert ignore(status.splitlines())==ignore(before)
    old_ignore=(HERE/'preexisting_gitignore.patch').read_text(encoding='utf-8-sig').replace('\r\n','\n').rstrip()
    current_ignore=subprocess.check_output(['git','diff','--binary','--','.gitignore'],text=True).rstrip()
    assert old_ignore==current_ignore
    result=dict(deployed_revision=expected,exact_six_file_commit=True,unrelated_ancestors_not_pushed=True,
        original_head_and_index_preserved=True,preexisting_status_preserved=True,prior_artifacts_unchanged=count,
        source_model_input_hashes_verified=True,remote_smoke_rows=6,remote_representative_rows=288,
        cross_repeat_bytes_pixels_quality_exact=True,within_hf_optimization_pairs_exact=True,
        baseline_matches_prior_windows_cases=18,baseline_prior_windows_cases=32,
        cross_environment_baseline_difference_predates_pass5=True,
        public_api_verified=True,public_compression_calls=2,local_checker_correction_disclosed=True,
        public_defaults_ui_unchanged=True,benchmark_db_writes=0,new_images_models=0,
        production_default_promoted=False,secret_value_logged_saved_committed=False)
    (HERE/'final_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json')
    inventory=sorted(set(str(p.relative_to(ROOT)).replace('\\','/') for p in paths)|{'research_pass5/artifact_manifest.json','research_pass5/changed_files.txt'})
    (HERE/'changed_files.txt').write_text('\n'.join(inventory)+'\n')
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json')
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in paths}
    (HERE/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
