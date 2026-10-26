import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PASS2 = ROOT / 'research_pass2'
sys.path.append(str(PASS2))
import common as shared

shared.OUT = OUT
os.environ.update(shared.THREAD_ENV)
from common import PRESETS, SCREEN, RELEASE, PASS1, THREAD_ENV, sha, save, snapshot, images, run_plan


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initial():
    if (OUT / 'initial_snapshot.json').exists():
        raise RuntimeError('Initial snapshot already exists')
    value = snapshot()
    save('initial_snapshot', value)
    (OUT / 'git_status_before.txt').write_text(value['status'])
    paths = [p for directory in (PASS1, PASS2) for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.nbc', '.nbi')]
    paths += [ROOT / n for n in ('PBC_BASELINE_AUDIT.md', 'PBC_BENCHMARK_VALIDATION.md', 'PBC_FIRST_PASS_RESULTS.md')]
    manifest = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths}
    prior = json.loads((PASS2 / 'artifact_manifest.json').read_text())
    assert all(sha((ROOT / p).read_bytes()) == r['sha256'] for p, r in prior.items())
    save('previous_artifacts_manifest', manifest)


def read(name):
    path = OUT / (name + '.json')
    return json.loads(path.read_text()) if path.exists() else []
