import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE='https://huggingface.co/spaces/EgeEken/PBC/resolve/'
HOST='https://egeeken-pbc.hf.space'


def main():
    metadata=json.load(urllib.request.urlopen('https://huggingface.co/api/spaces/EgeEken/PBC'))
    revision=metadata['sha']
    assert revision==metadata['runtime']['sha']=='cd0d5691647a4f02cf765454a2d3b0abfde8f3b7'
    paths=['PBC3.py','pbc3_ops.py','pbc3_heads.py','pbc3_stream.py','learned_filler.py','pbc3_types.py','server.py','pbc3_benchmark.py','Dockerfile']
    sources={}
    for path in paths:
        source=urllib.request.urlopen(BASE+revision+'/'+path).read().decode().replace('\r\n','\n')
        sources[path]=dict(deployed_lf_sha256=hashlib.sha256(source.encode()).hexdigest(),
            before_sha256=hashlib.sha256(__import__('subprocess').check_output(['git','show','3e5a725:'+path]).decode().replace('\r\n','\n').encode()).hexdigest())
        assert sources[path]['deployed_lf_sha256']==sources[path]['before_sha256'],path
    result=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha=revision,runtime=metadata['runtime'],
        source_hashes=sources,health=json.load(urllib.request.urlopen(HOST+'/api/health')),
        storage=metadata.get('storage'),live_dataset=json.load(urllib.request.urlopen(HOST+'/api/benchmark/dataset')))
    (HERE/'active_path_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(sha=revision,stage=metadata['runtime']['stage'],source_files_verified=len(sources),images=result['live_dataset']['count'],storage=result['storage'])))


if __name__=='__main__':
    main()
