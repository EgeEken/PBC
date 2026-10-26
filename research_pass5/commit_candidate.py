import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent


def main():
    files=json.loads((HERE/'deployment_files.json').read_text())
    base=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
    assert base=='7faa6365186d201f44b842f100645b60273ffd4c'
    assert json.loads((HERE/'candidate_checks.json').read_text())['exact']
    env=dict(os.environ,GIT_INDEX_FILE=str(HERE/'deployment.index'))
    subprocess.run(['git','read-tree',base],env=env,check=True)
    subprocess.run(['git','add','--']+files,env=env,check=True)
    changed=subprocess.check_output(['git','diff','--cached','--name-only',base],env=env,text=True).splitlines()
    assert sorted(changed)==sorted(files),(changed,files)
    subprocess.run(['git','diff','--cached','--check'],env=env,check=True)
    patch=subprocess.check_output(['git','diff','--cached','--binary',base],env=env)
    (HERE/'deployment.patch').write_bytes(patch)
    tree=subprocess.check_output(['git','write-tree'],env=env,text=True).strip()
    message='Add authenticated bounded HF runtime validation\n\nKeep public codec defaults and UI unchanged. Run baseline, exact grid/search, and optional metric-free modes in an isolated worker with static input bounds and source/model verification.\n'
    commit=subprocess.check_output(['git','commit-tree',tree,'-p',base],input=message,text=True).strip()
    subprocess.run(['git','update-ref','refs/heads/codex/pass5-hf-runtime',commit,'0'*40],check=True)
    result=dict(base_revision=base,commit=commit,local_branch='codex/pass5-hf-runtime',
        target='https://huggingface.co/spaces/EgeEken/PBC',target_ref='refs/heads/main',files=files,
        method='Isolated index from HF main; current checkout/index/branch and unrelated commits are preserved.',
        rollback='git revert '+commit+'; push the revert to HF main. Delete PBC_RESEARCH_BENCHMARK_TOKEN to disable research runs.')
    (HERE/'deployment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
