import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
RELEASE='2e59f1b06980aa467da0cde94fa33c3f9a0d4d51'
BASE='cd0d5691647a4f02cf765454a2d3b0abfde8f3b7'


def main():
    release_files=sorted((ROOT/'research_pass1/release').glob('*.py'))
    assert len(release_files)==6
    verified={}
    for path in release_files:
        original=subprocess.check_output(['git','show',RELEASE+':'+path.name])
        assert original.decode().replace('\r\n','\n')==path.read_text(encoding='utf-8'),path.name
        verified[path.name]=dict(git_blob=subprocess.check_output(['git','rev-parse',RELEASE+':'+path.name],text=True).strip(),sha256=hashlib.sha256(original).hexdigest())
    model=subprocess.check_output(['git','show',RELEASE+':patch_policy.npz'])
    assert model==(ROOT/'patch_policy.npz').read_bytes()
    core=['PBC3.py','pbc3_ops.py','pbc3_heads.py','learned_filler.py','pbc3_stream.py','pbc3_features.py','pbc3_types.py','pbc3_trace.py']
    for name in core:
        assert subprocess.check_output(['git','show',BASE+':'+name]).decode().replace('\r\n','\n')==(ROOT/name).read_text(encoding='utf-8'),name
    files=['server.py','pbc3_benchmark.py','static/benchmark.js','research_pass8/worker.py','research_pass8/hf_release_job.py']
    files += [p.relative_to(ROOT).as_posix() for p in release_files]
    sources=core+files
    assets=json.loads((ROOT/'research/assets.json').read_text())
    manifest=dict(release_revision=RELEASE,development_codec_revision=BASE,
        source_lf={p:hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in sources},
        assets={p:v['sha256'] for p,v in assets['inputs'].items() if p.startswith('hpt_data/')})
    manifest['assets']['patch_policy.npz']=hashlib.sha256(model).hexdigest()
    for name,expected in manifest['assets'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    (HERE/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',newline='\n')
    old=ROOT/'research_pass6/runtime_manifest.json'
    prior=json.loads(old.read_text())
    for p in ('server.py','pbc3_benchmark.py'):
        prior['source_lf'][p]=manifest['source_lf'][p]
    old.write_text(json.dumps(prior,indent=2)+'\n',newline='\n')
    files += ['research_pass8/runtime_manifest.json','research_pass6/runtime_manifest.json']
    env=dict(os.environ,GIT_INDEX_FILE=str(HERE/'hf-deployment.index'))
    subprocess.run(['git','read-tree',BASE],env=env,check=True)
    for name in files:
        data=(ROOT/name).read_text(encoding='utf-8').encode()
        oid=subprocess.check_output(['git','hash-object','--no-filters','-w','--stdin'],input=data).decode().strip()
        subprocess.run(['git','update-index','--index-info'],input=f'100644 {oid}\t{name}\n'.encode(),env=env,check=True)
    tree=subprocess.check_output(['git','write-tree'],env=env,text=True).strip()
    commit=subprocess.check_output(['git','commit-tree',tree,'-p',BASE],input='Add authenticated immutable-release benchmark and Speed Benchmark run selector\n\nKeep production codec/defaults unchanged. Bound comparison to 20 existing main images, four presets, two fixed sources and three trials. No RD experiment or data/model payload added.\n',text=True).strip()
    result=dict(base=BASE,commit=commit,files=files,verified_release_sources=verified,model_sha256=manifest['assets']['patch_policy.npz'],
        production_codec_unchanged=True,rollback='Non-destructive revert of this benchmark-only HF commit; never reset or force-push.')
    (HERE/'hf_deployment.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
