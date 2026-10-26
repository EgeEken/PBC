import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent


def main():
    checks=json.loads((HERE/'integration_checks.json').read_text())
    assert checks['all_exact'] and checks['frozen_cases']==100
    code=['PBC3.py','pbc3_ops.py','pbc3_heads.py','learned_filler.py','pbc3_stream.py','server.py','pbc3_benchmark.py']
    code+=['research_pass6/'+p for p in ('worker.py','hf_runtime_job.py')]
    code+=[p.relative_to(ROOT).as_posix() for p in sorted((HERE/'baseline').glob('*.py'))]
    manifest=dict(source_lf={p:hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in code},
        assets={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('patch_policy.npz','hpt_data/cat.jpg','hpt_data/flower.jpg','hpt_data/googlewiki.png')})
    (HERE/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',newline='\n')
    files=code+['research_pass6/runtime_manifest.json']
    base=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
    assert base=='eee83b8f10f6f3404bca1c2c178e27fc4e8a1aa0',base
    env=dict(os.environ,GIT_INDEX_FILE=str(HERE/'hf-deployment.index'))
    subprocess.run(['git','read-tree',base],env=env,check=True)
    for name in files:
        data=(ROOT/name).read_text(encoding='utf-8').encode()
        oid=subprocess.check_output(['git','hash-object','--no-filters','-w','--stdin'],input=data).decode().strip()
        subprocess.run(['git','update-index','--index-info'],input=f'100644 {oid}\t{name}\n'.encode(),env=env,check=True)
    tree=subprocess.check_output(['git','write-tree'],env=env,text=True).strip()
    commit=subprocess.check_output(['git','commit-tree',tree,'-p',base],input='Integrate exact grid I/O and per-encode search reuse\n\nKeep public presets, stream v0 and internal MSE unchanged. Preserve the prior codec in an isolated bounded authenticated smoke worker. No RD refinement is enabled or deployed.\n',text=True).strip()
    (HERE/'hf_deployment.json').write_text(json.dumps(dict(base=base,commit=commit,files=files,
        rollback='Create and push a non-destructive revert of '+commit+' to HF main; preserve unrelated work.'),indent=2)+'\n')
    print(commit)


if __name__=='__main__':
    main()
