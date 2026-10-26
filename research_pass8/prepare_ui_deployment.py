import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent


def main():
    base=json.loads((HERE/'hf_deployment.json').read_text())['commit']
    path=HERE/'runtime_manifest.json'
    manifest=json.loads(path.read_text())
    manifest['source_lf']['static/benchmark.js']=hashlib.sha256((ROOT/'static/benchmark.js').read_text(encoding='utf-8').encode()).hexdigest()
    path.write_text(json.dumps(manifest,indent=2)+'\n',newline='\n')
    files=['static/benchmark.js','research_pass8/runtime_manifest.json']
    env=dict(os.environ,GIT_INDEX_FILE=str(HERE/'hf-ui-deployment.index'))
    subprocess.run(['git','read-tree',base],env=env,check=True)
    for name in files:
        data=(ROOT/name).read_text(encoding='utf-8').encode()
        oid=subprocess.check_output(['git','hash-object','--no-filters','-w','--stdin'],input=data).decode().strip()
        subprocess.run(['git','update-index','--index-info'],input=f'100644 {oid}\t{name}\n'.encode(),env=env,check=True)
    tree=subprocess.check_output(['git','write-tree'],env=env,text=True).strip()
    commit=subprocess.check_output(['git','commit-tree',tree,'-p',base],input='Keep benchmark run selection stable during asynchronous polling\n\nIgnore stale result responses and preserve unchanged selector options. Codec and authenticated comparison worker unchanged.\n',text=True).strip()
    (HERE/'hf_ui_deployment.json').write_text(json.dumps(dict(base=base,commit=commit,files=files,production_codec_unchanged=True),indent=2)+'\n',newline='\n')
    print(commit)


if __name__=='__main__':
    main()
