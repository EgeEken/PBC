import difflib
import hashlib
import json
from pathlib import Path

from benchmark import ROOT, OUT


if __name__ == '__main__':
    provenance = json.loads((OUT / 'github_provenance.json').read_text())
    tree = json.loads(provenance['github_release_tree']['content'])['tree']
    entries = []
    for item in tree:
        name = item['path']
        release = OUT / 'release' / name
        if not release.exists():
            continue
        data = release.read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == item['sha'], name
        entry = {'file': name, 'github_blob': item['sha'], 'states': {}}
        for source, path in (('release', release), ('current', ROOT / name), ('old', ROOT / 'old_pbc' / name)):
            if not path.exists():
                continue
            data = path.read_bytes()
            entry['states'][source] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            if source != 'release' and name.endswith('.py'):
                lines = difflib.unified_diff(release.read_text().splitlines(True), path.read_text().splitlines(True),
                                             fromfile='release/' + name, tofile=source + '/' + name)
                diff = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n' for line in lines)
                (OUT / (source + '_' + name + '.diff')).write_text(diff)
        entries.append(entry)
    (OUT / 'implementation_hashes.json').write_text(json.dumps(entries, indent=2))
    files = list((OUT / 'release').glob('*.py')) + list((ROOT / 'old_pbc').glob('*.py')) + list((ROOT / 'old_pbc' / 'training_space').glob('*.py'))
    files += [ROOT / name for name in ('PBC3.py', 'PBC3_animation.py', 'pbc3_ops.py', 'pbc3_heads.py', 'pbc3_types.py',
                                       'pbc3_features.py', 'pbc3_stream.py', 'pbc3_trace.py', 'learned_filler.py', 'patch_features.py',
                                       'server.py', 'Dockerfile', 'requirements_docker.txt', 'pbc3_quick_rd.py', 'pbc3_benchmark.py', 'pbc3_sweep.py')]
    (OUT / 'runtime_source_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2))
    print('Release blobs verified; source hashes and exact diffs recorded')
