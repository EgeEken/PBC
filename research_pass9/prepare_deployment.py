import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
BASE='27f6c65225a5496fa065deb586c7e405ecdffc84'
RELEASE='2e59f1b06980aa467da0cde94fa33c3f9a0d4d51'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--commit',action='store_true')
    args=parser.parse_args()
    prior=json.loads((ROOT/'research_pass8/runtime_manifest.json').read_text())
    sources=[p for p in prior['source_lf'] if not p.startswith('research_pass8/')]
    sources+=['research_pass9/worker.py','research_pass9/hf_release_job.py']
    for p in sources:
        if p.startswith('research_pass1/release/'):
            original=subprocess.check_output(['git','show',RELEASE+':'+Path(p).name]).decode().replace('\r\n','\n')
            assert original==(ROOT/p).read_text(encoding='utf-8'),p
        elif p not in ('PBC3.py','server.py','pbc3_benchmark.py') and not p.startswith('research_pass9/'):
            original=subprocess.check_output(['git','show',BASE+':'+p]).decode().replace('\r\n','\n')
            assert original==(ROOT/p).read_text(encoding='utf-8'),p
    model=subprocess.check_output(['git','show',RELEASE+':patch_policy.npz'])
    assert model==(ROOT/'patch_policy.npz').read_bytes()
    assets=json.loads((ROOT/'research/assets.json').read_text())
    manifest=dict(release_revision=RELEASE,base_hf_revision=BASE,public_rng_default='current',
        source_lf={p:hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in sources},
        assets={p:v['sha256'] for p,v in assets['inputs'].items()})
    manifest['assets']['patch_policy.npz']=hashlib.sha256(model).hexdigest()
    for p,value in manifest['assets'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==value,p
    (HERE/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    if not args.commit:
        print('Immutable release/model, unchanged support modules, and 25 inputs verified; manifest written.')
        return
    checks=json.loads((HERE/'local_equivalence.json').read_text())
    assert checks['cases']==100 and checks['equivalent']
    files=['PBC3.py','server.py','pbc3_benchmark.py','research_pass9/worker.py','research_pass9/hf_release_job.py','research_pass9/runtime_manifest.json']
    env=dict(os.environ,GIT_INDEX_FILE=str(HERE/'hf-deployment.index'))
    subprocess.run(['git','read-tree',BASE],env=env,check=True)
    for p in files:
        data=(ROOT/p).read_text(encoding='utf-8').encode()
        oid=subprocess.check_output(['git','hash-object','--no-filters','-w','--stdin'],input=data).decode().strip()
        subprocess.run(['git','update-index','--index-info'],input=f'100644 {oid}\t{p}\n'.encode(),env=env,check=True)
    tree=subprocess.check_output(['git','write-tree'],env=env,text=True).strip()
    commit=subprocess.check_output(['git','commit-tree',tree,'-p',BASE],input='Add opt-in per-encode release RNG and bounded authenticated three-way research comparison\n\nKeep public encoding defaults, presets, MSE, decoder, model, and UI unchanged.\n',text=True).strip()
    result=dict(base=BASE,commit=commit,files=files,public_rng_default='current',codec_change='Explicit per-encode RNG selection only; no decoder or other algorithm change.')
    (HERE/'hf_deployment.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
