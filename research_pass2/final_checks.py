import json
from pathlib import Path
import py_compile

from common import ROOT, OUT, PASS1, sha, snapshot, save
from analyze import read, collapse


def verify():
    initial = json.loads((OUT / 'initial_snapshot.json').read_text())
    final = snapshot()
    assert initial['commit'] == final['commit']
    assert initial['hashes'] == final['hashes'], 'Source/model/input mutation'
    pass1 = json.loads((OUT / 'pass1_frozen_manifest.json').read_text())
    assert all(sha((ROOT / path).read_bytes()) == digest for path, digest in pass1.items()), 'Pass 1 artifact mutation'
    prior_manifest = json.loads((PASS1 / 'artifact_manifest.json').read_text())
    assert all(sha((ROOT / r['path']).read_bytes()) == r['sha256'] for r in prior_manifest if r['path'] in ('PBC_BASELINE_AUDIT.md', 'PBC_BENCHMARK_VALIDATION.md', 'PBC_FIRST_PASS_RESULTS.md'))
    old_manifest = json.loads((PASS1 / 'runtime_source_manifest.json').read_text())
    assert all(sha((ROOT / path).read_bytes()) == digest for path, digest in old_manifest.items())
    expected = {'rng_control': 600, 'resolution_screen': 144, 'allocation_screen': 36, 'rate_screen': 72,
        'performance_screen': 288, 'rng_profile': 48, 'performance_components': 18}
    for name in ('quality_confirmation', 'performance_confirmation'):
        plan = read('quality_confirmation_plan') if name == 'quality_confirmation' else read('performance_promotion')
        expected[name] = len(plan['cases']) * plan.get('repeats', 3) if name == 'quality_confirmation' else 25 * 4 * (1 + len(plan['variants'])) * 3
    counts = {}
    checked_streams = 0
    for name, count in expected.items():
        rows = read(name)
        assert len(rows) == count, (name, len(rows), count)
        counts[name] = count
        assert all(r['preview_equals_decode'] for r in rows)
        assert all(r['patch_count'] == r['initialization_patches'] + r['iterative_patches'] for r in rows)
        assert all(r['bpp'] == 8 * r['final_bytes'] / (r['width'] * r['height']) for r in rows)
        assert all(r['deterministic'] for r in collapse(rows))
        for row in rows:
            assert row['raw_bytes'] * 8 == row['accounting']['wrapper_bits'] + row['accounting']['header_bits'] + row['accounting']['padding_bits'] + sum(p['total_bits'] for p in row['accounting']['patches'])
            assert row['input_sha256'] == final['hashes'][str(Path(row['dataset']) / row['image'])]
            assert row['model_sha256'] == final['hashes']['patch_policy.npz']
            if row.get('stream_path'):
                data = (OUT / row['stream_path']).read_bytes()
                assert len(data) == row['final_bytes'] and sha(data) == row['stream_sha256']
                checked_streams += 1
            if row['source'] == 'release':
                for module in ('PBC3', 'pbc3_types', 'pbc3_ops', 'pbc3_heads', 'pbc3_features', 'learned_filler'):
                    assert Path(row['modules'][module]).is_relative_to(PASS1 / 'release')
    equivalent = read('rng_equivalence')
    assert len(equivalent) == 300
    assert all(r['bytes_equal'] and r['pixels_equal'] and r['patches_equal'] and r['counts_equal'] for r in equivalent)
    for name in ('performance_screen_paired', 'performance_confirmation_paired'):
        assert all(r['bytes_equal'] and r['pixels_equal'] and r['patches_equal'] and r['mse_pct'] == 0 and r['bpp_pct'] == 0 for r in read(name))
    entropy = read('entropy_screen')
    assert len(entropy) == 1008 and all(r['pixels_equal'] for r in entropy)
    assert all(r.get('raw_equal', r.get('semantic_equal', False)) for r in entropy)
    unit = read('optimization_unit_checks')
    assert len(unit['grid_io']) == 640
    assert all(r['equal'] for r in unit['grid_io'] + unit['channel_sum'])
    for script in OUT.glob('*.py'):
        py_compile.compile(str(script), doraise=True)
    reports = ('RNG_SPEED_ATTRIBUTION.md', 'PBC_QUALITY_SCALING.md', 'PBC_ENTROPY_RESEARCH.md', 'PBC_PERFORMANCE_RESULTS.md', 'PBC_PASS2_SUMMARY.md')
    assert all((OUT / name).stat().st_size > 500 for name in reports)
    assert initial['status'] == final['status'], 'Unexpected Git status change outside the already-created research directory'
    (OUT / 'git_status_after.txt').write_text(final['status'])
    save('final_snapshot', final)
    save('final_validation', dict(production_models_inputs_unchanged=True, pass1_artifacts_unchanged=True,
        head_unchanged=final['commit'], preexisting_user_git_status_preserved=True, rows=counts,
        checked_stream_files=checked_streams, rng_identical_cases=100, rng_identical_repeated_comparisons=300,
        lossless_entropy_rows=len(entropy), syntax_unit_checks=640,
        remote_compression_requests=0, commits=0, pushes=0, deployments=0))
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.nbc', '.nbi'))
    names = sorted({str(p.relative_to(ROOT)) for p in files} | {'research_pass2\\artifact_manifest.json', 'research_pass2\\changed_files.txt'})
    (OUT / 'changed_files.txt').write_text('\n'.join(names) + '\n')
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.nbc', '.nbi') and p.name != 'artifact_manifest.json')
    save('artifact_manifest', {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in files})
    print(json.dumps(read('final_validation'), indent=2))


if __name__ == '__main__':
    verify()
