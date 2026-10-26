import hashlib
import json
from pathlib import Path
import py_compile
import subprocess

from benchmark import ROOT, OUT


if __name__ == '__main__':
    manifest = json.loads((OUT / 'runtime_source_manifest.json').read_text())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in manifest.items())
    metadata = json.loads((OUT / 'baseline_metadata.json').read_text())
    assert metadata['commit'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert all(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'] for item in metadata['dataset'])
    rows = [json.loads(line) for line in (OUT / 'baseline.jsonl').read_text().splitlines()]
    assert len(rows) == 900 and all(r['preview_equals_decode'] for r in rows)
    for row in (r for r in rows if r['trial'] == 0):
        path = OUT / 'streams' / row['source'] / row['dataset'] / (row['image'] + '.' + row['preset'] + '.pbc3')
        data = path.read_bytes()
        assert len(data) == row['final_bytes']
        assert hashlib.sha256(data).hexdigest() == row['stream_sha256']
        assert row['bpp'] == 8 * len(data) / (row['width'] * row['height'])
    assert len(json.loads((OUT / 'stream_comparison.json').read_text())) == 200
    assert len(json.loads((OUT / 'resolution_floor.json').read_text())) == 400
    assert len(json.loads((OUT / 'backend_validation.json').read_text())) == 4
    assert all(r['current_with_numpy_rng_equals_release'] for r in json.loads((OUT / 'rng_attribution.json').read_text()))
    for path in OUT.glob('*.py'):
        py_compile.compile(str(path), doraise=True)
    status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    (OUT / 'git_status_after.txt').write_text(status)
    before = set((OUT / 'git_status_before.txt').read_text().splitlines())
    after = set(status.splitlines())
    added = sorted(after - before)
    removed = sorted(before - after)
    assert not removed, removed
    assert set(added) == {'?? PBC_BASELINE_AUDIT.md', '?? PBC_BENCHMARK_VALIDATION.md', '?? PBC_FIRST_PASS_RESULTS.md', '?? research_pass1/'}, added
    deployment = json.loads((OUT / 'deployment_comparison.json').read_text()) if (OUT / 'deployment_comparison.json').exists() else []
    if deployment:
        assert len(deployment) == 4
        assert all(r['remote_preview_equals_decoded_rgb'] and r['remote_displayed_mse_matches_rgba'] and r['remote_displayed_kb_matches_bytes'] for r in deployment)
    record = {'commit': metadata['commit'], 'baseline_rows': len(rows), 'saved_baseline_streams': 300,
              'production_sources_unchanged': True, 'dataset_hashes_unchanged': True,
              'existing_user_git_status_preserved': True, 'added_status_entries': added,
              'remote_compression_upload_performed': bool(deployment), 'remote_compression_checks': len(deployment),
              'remote_byte_identical_checks': sum(r['remote_bytes_equal_local'] for r in deployment),
              'commit_push_deploy_performed': False}
    (OUT / 'final_validation.json').write_text(json.dumps(record, indent=2))
    reports = [ROOT / name for name in ('PBC_BASELINE_AUDIT.md', 'PBC_BENCHMARK_VALIDATION.md', 'PBC_FIRST_PASS_RESULTS.md')]
    files = reports + [p for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('artifact_manifest.json', 'generated_files.txt')]
    records = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    (OUT / 'artifact_manifest.json').write_text(json.dumps(records, indent=2))
    (OUT / 'generated_files.txt').write_text('\n'.join(r['path'] for r in records) + '\nresearch_pass1/artifact_manifest.json\nresearch_pass1/generated_files.txt\n')
    print(json.dumps(record, indent=2))
