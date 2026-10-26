import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'research'/'snapshot_manifest.json').read_text())
    for name, record in manifest['files'].items():
        data=(root/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('Snapshot hash mismatch: '+name)
        if name.endswith('.py'):
            ast.parse(data.decode('utf-8'),filename=name)
    print(json.dumps(dict(files_verified=len(manifest['files']),python_syntax=True,codec_imports=0,benchmarks=0)))


if __name__=='__main__':
    main()
