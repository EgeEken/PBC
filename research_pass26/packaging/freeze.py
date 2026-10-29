import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

PASS=Path(__file__).resolve().parents[1]
DIGEST='edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede'

def freeze():
    archive=PASS/'artifacts/released.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve('https://github.com/EgeEken/PBC/releases/download/v3.1/PBC3_1_Release.zip',archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=DIGEST:
        raise ValueError('published release ZIP hash differs from frozen baseline')
    source=archive.parent/'published'
    expected=json.loads((PASS/'baseline.json').read_text())['files']
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            path=Path(entry.filename)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe ZIP path')
        z.extractall(source)
    for row in expected:
        if hashlib.sha256((source/row['file']).read_bytes()).hexdigest()!=row['published_sha256']:
            raise ValueError('source hash mismatch: '+row['file'])
    return source

if __name__=='__main__':
    print(freeze())
