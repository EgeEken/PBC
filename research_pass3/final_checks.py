import json
from pathlib import Path
import py_compile
import subprocess

from support import ROOT, OUT, PASS1, PASS2, snapshot, save, read, sha
from analyze import collapse


def check():
    initial = read('initial_snapshot')
    final = snapshot()
    assert initial['commit'] == final['commit']
    assert initial['hashes'] == final['hashes']
    previous = read('previous_artifacts_manifest')
    assert all(sha((ROOT / p).read_bytes()) == value for p, value in previous.items()), 'Previous artifact mutation'
    patch = read('integration_patch_provenance')
    assert sha((ROOT / 'pbc3_stream.py').read_bytes()) == patch['production_source_sha256']
    assert sha((OUT / 'integration_candidate' / 'pbc3_stream.py').read_bytes()) == patch['candidate_sha256']
    subprocess.run(['git', 'apply', '--check', str(OUT / 'proposed_grid_io.patch')], cwd=ROOT, check=True, capture_output=True)
    expected = {'performance_screen': 288, 'fixed_identity': 36, 'fixed_screen': 144,
        'integrated_screen': 144, 'search_exactness': 48, 'performance_full': 1200,
        'allocation_full': 675, 'search_screen': 216, 'search_setup_screen': 144, 'search_full': 600, 'search_profile': 18}
    counts = {}
    checked_files = 0
    for name, count in expected.items():
        rows = read(name)
        assert len(rows) == count, (name, len(rows), count)
        counts[name] = count
        assert all(r['preview_equals_decode'] for r in rows)
        assert all(r['bpp'] == 8 * r['final_bytes'] / (r['width'] * r['height']) for r in rows)
        assert all(r['patch_count'] == r['initialization_patches'] + r['iterative_patches'] for r in rows)
        assert all(r['deterministic'] for r in collapse(rows))
        for row in rows:
            assert row['model_sha256'] == initial['hashes']['patch_policy.npz']
            assert (row['internal_mse'] is None) == (not row['config'].get('compute_final_mse', True))
            assert row['input_sha256'] == initial['hashes'][str(Path(row['dataset']) / row['image'])]
            a = row['accounting']
            assert a['raw_bytes'] * 8 == a['wrapper_bits'] + a['header_bits'] + a['padding_bits'] + sum(p['total_bits'] for p in a['patches'])
            if row.get('stream_path'):
                data = (OUT / row['stream_path']).read_bytes()
                assert sha(data) == row['stream_sha256'] and len(data) == row['final_bytes']
                checked_files += 1
            if row['source'] == 'release':
                assert all(Path(row['modules'][n]).is_relative_to(PASS1 / 'release') for n in ('PBC3', 'pbc3_types', 'pbc3_ops', 'pbc3_heads', 'pbc3_features', 'learned_filler'))
    for name in ('performance_full_paired', 'search_full_paired'):
        rows = read(name)
        assert rows and all(r['bytes_equal'] and r['pixels_equal'] and r['patches_equal'] and r['mse_pct'] == 0 and r['bpp_pct'] == 0 for r in rows)
    assert all(r['frozen_control_equal'] for r in read('performance_full') + read('search_full') + read('search_exactness'))
    assert all(r['initialization_equal'] for r in read('allocation_trajectories'))
    assert all(r['same_geometry_order_count'] for r in read('allocation_trajectories') if '_fixed_' in r['label'])
    for name in ('performance_correctness', 'integration_checks', 'search_compatibility'):
        assert read(name)
    assert len(read('patch_ledger')) > 500 and len(read('representation_study')) > 300
    for row in read('patch_ledger'):
        assert row['global_working_sse_before'] - row['global_working_sse_after'] == row['local_sse_gain']
    import numpy as np
    caches = {}
    for row in read('patch_ledger'):
        file = row['quantized_values_file']
        if file not in caches:
            caches[file] = np.load(OUT / file)
        values = caches[file][row['quantized_values_key']]
        assert sha(values.tobytes()) == row['quantized_values_sha256']
    for cache in caches.values():
        cache.close()
    smoke = json.loads((OUT / 'hf_plan_smoke' / 'equivalence.json').read_text())
    assert smoke['current_variants_exact'] and smoke['requests'] == 0
    for script in OUT.rglob('*.py'):
        if '__pycache__' not in script.parts:
            py_compile.compile(str(script), doraise=True)
    reports = ('PBC_PERFORMANCE_INTEGRATION.md', 'PBC_RATE_AWARE_ALLOCATION.md', 'PBC_SEARCH_REUSE.md', 'PBC_PASS3_SUMMARY.md')
    assert all((OUT / name).stat().st_size > 500 for name in reports)
    assert initial['status'] == final['status']
    (OUT / 'git_status_after.txt').write_text(final['status'])
    save('final_snapshot', final)
    save('final_validation', dict(production_model_datasets_unchanged=True, previous_artifacts_verified=len(previous),
        previous_artifacts_unchanged=True, head_unchanged=final['commit'], preexisting_git_status_preserved=True,
        measured_rows=counts, saved_stream_files_verified=checked_files, grid_layout_checks=len(read('performance_correctness')['bit_layout']),
        integration_patch_applied=False, integration_patch_check_passed=True,
        remote_requests=0, uploads=0, commits=0, pushes=0, deployments=0, training=0))
    files = [p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.nbi', '.nbc')]
    names = {str(p.relative_to(ROOT)) for p in files} | {'research_pass3\\changed_files.txt', 'research_pass3\\artifact_manifest.json'}
    (OUT / 'changed_files.txt').write_text('\n'.join(sorted(names)) + '\n')
    files = [p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.nbi', '.nbc') and p.name != 'artifact_manifest.json']
    save('artifact_manifest', {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in sorted(files)})
    print(json.dumps(read('final_validation'), indent=2))


if __name__ == '__main__':
    check()
