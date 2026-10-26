import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark import OUT, ROOT, THREAD_ENV


def snapshot():
    metadata = json.loads((OUT / 'baseline_metadata.json').read_text())
    paths = list(json.loads((OUT / 'runtime_source_manifest.json').read_text()))
    paths += ['patch_policy.npz', 'old_pbc/patch_policy.npz', 'research_pass1/release/patch_policy.npz']
    return {'observed_unix_time': time.time(), 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_model_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
            'dataset': [{'path': item['path'], 'sha256': hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()} for item in metadata['dataset']],
            'thread_env': THREAD_ENV, 'affinity_in_child': [0], 'python': sys.version,
            'dependencies': metadata['sources']['release']['versions'],
            'default_presets': {source: value['presets'] for source, value in metadata['sources'].items()}}


if __name__ == '__main__':
    records = []
    jobs = [('stream_compare', ['experiments.py', 'compare']), ('rng_attribution', ['rng_diagnostic.py']),
            ('A_resolution', ['experiments.py', 'floor']), ('B_profile', ['experiments.py', 'profile']),
            ('C_D_E_screen', ['experiments.py', 'screen'])]
    for name, arguments in jobs:
        before = snapshot()
        record = {'job': name, 'command': [sys.executable, str(OUT / arguments[0]), *arguments[1:]], 'before': before}
        records.append(record)
        (OUT / 'experiment_provenance.json').write_text(json.dumps(records, indent=2))
        print('START', name, flush=True)
        with open(OUT / (name + '_progress.log'), 'w') as log:
            completed = subprocess.run(record['command'], cwd=ROOT, env={**os.environ, **THREAD_ENV}, stdout=log, stderr=subprocess.STDOUT)
        record.update(after=snapshot(), exit_code=completed.returncode)
        assert record['before']['source_model_sha256'] == record['after']['source_model_sha256']
        assert record['before']['dataset'] == record['after']['dataset']
        assert record['before']['commit'] == record['after']['commit']
        (OUT / 'experiment_provenance.json').write_text(json.dumps(records, indent=2))
        print('DONE', name, completed.returncode, flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)
