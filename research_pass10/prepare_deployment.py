import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
BASE='4237d954f4e1a04951c822c738113a53bbd58e9b'


def main():
    checks=json.loads((HERE/'local_checks.json').read_text())
    assert checks['release_identical_cases']==4 and checks['custom_preserved_cases']==2
    before=subprocess.check_output(['git','show',BASE+':PBC3.py']).decode().replace('\r\n','\n')
    assert before.replace('rng_mode="current"','rng_mode="release"')==(ROOT/'PBC3.py').read_text(encoding='utf-8')
    path=ROOT/'research_pass9/runtime_manifest.json'
    manifest=json.loads(path.read_text())
    for name in manifest['source_lf']:
        if name not in ('PBC3.py','pbc3_benchmark.py','static/benchmark.js','research_pass9/worker.py'):
            assert subprocess.check_output(['git','show',BASE+':'+name]).decode().replace('\r\n','\n')==(ROOT/name).read_text(encoding='utf-8'),name
    for name,value in manifest['assets'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name
    manifest.update(base_hf_revision=BASE,public_rng_default='release',development_baseline='pbc31_release_rng_baseline')
    manifest['source_lf']={name:hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest() for name in manifest['source_lf']}
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    files=['PBC3.py','pbc3_benchmark.py','static/benchmark.js','research_pass9/worker.py','research_pass9/runtime_manifest.json']
    env=dict(os.environ,GIT_INDEX_FILE=str(HERE/'hf-deployment.index'))
    subprocess.run(['git','read-tree',BASE],env=env,check=True)
    for name in files:
        data=(ROOT/name).read_text(encoding='utf-8').encode()
        oid=subprocess.check_output(['git','hash-object','--no-filters','-w','--stdin'],input=data).decode().strip()
        subprocess.run(['git','update-index','--index-info'],input=f'100644 {oid}\t{name}\n'.encode(),env=env,check=True)
    tree=subprocess.check_output(['git','write-tree'],env=env,text=True).strip()
    commit=subprocess.check_output(['git','commit-tree',tree,'-p',BASE],input='Promote release-compatible RNG as development default\n\nPreserve explicit custom RNG, public presets/model/v0 decoder and runtime optimizations. Record new benchmark provenance, patches and file hashes without relabeling history.\n',text=True).strip()
    record=dict(base=BASE,commit=commit,files=files,baseline_id='pbc31_release_rng_baseline',public_rng_default='release',stream_version=0,
        codec_change='Only compress/compress_stream default selection; previous custom generator remains explicit.',model_sha256=manifest['assets']['patch_policy.npz'])
    (HERE/'hf_deployment.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
