import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description='Restore declared local dataset inputs without encoding or network access')
    parser.add_argument('--source', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'research'/'assets.json').read_text())
    pending = []
    for name, record in manifest['inputs'].items():
        source = args.source.resolve()/name
        target = root/name
        if hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Source hash mismatch: '+name)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Refusing to overwrite differing input: '+name)
        else:
            pending.append((source,target))
    for source,target in pending:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    print(json.dumps(dict(copied_inputs=len(pending),declared_inputs=len(manifest['inputs']),network=False,encodes=0)))


if __name__=='__main__':
    main()
