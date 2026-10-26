import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PASS1, PASS2, PASS3 = [ROOT / ('research_pass' + str(n)) for n in (1, 2, 3)]
sys.path.append(str(PASS2))
import common as shared
shared.OUT = OUT
os.environ.update(shared.THREAD_ENV)
from common import PRESETS, SCREEN, RELEASE, THREAD_ENV, sha, save, snapshot, images, run_plan, request


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(name):
    return json.loads((OUT / (name + '.json')).read_text())


def initial():
    if (OUT / 'initial_snapshot.json').exists():
        raise RuntimeError('Initial snapshot exists')
    value = snapshot()
    save('initial_snapshot', value)
    (OUT / 'git_status_before.txt').write_text(value['status'])
    paths = [p for d in (PASS1, PASS2, PASS3) for p in d.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and '.mpl-cache' not in p.parts and p.suffix not in ('.nbc', '.nbi')]
    paths += [ROOT / n for n in ('PBC_BASELINE_AUDIT.md', 'PBC_BENCHMARK_VALIDATION.md', 'PBC_FIRST_PASS_RESULTS.md')]
    manifest = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths}
    for d in (PASS2, PASS3):
        prior = json.loads((d / 'artifact_manifest.json').read_text())
        assert all(sha((ROOT / p).read_bytes()) == r['sha256'] for p, r in prior.items())
    save('previous_artifacts_manifest', manifest)
    save('initial_integrity', dict(artifacts=len(manifest), verified=True))
