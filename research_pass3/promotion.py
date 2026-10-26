from support import ROOT, PRESETS, images, save, snapshot, read


def allocation():
    variants = {
        'release_q250_none': ('release', 'quality', {}, 'none'),
        'release_hq250_none': ('release', 'high_quality', {}, 'none'),
        'q250_none': ('current', 'quality', {}, 'none'),
        'hq250_none': ('current', 'high_quality', {}, 'none'),
        'q500p35_none': ('current', 'quality', {'auto_downsample_max_pixels': 500000, 'patch_count': 35}, 'none'),
    }
    for method in ('fixed', 'integrated'):
        for weight in (.4, .8):
            variants[f'q500p35_{method}_t{weight}'] = ('current', 'quality', {'auto_downsample_max_pixels': 500000, 'patch_count': 35}, f'{method}_t{weight}')
    cases = [dict(path=str(path), preset=preset, source=source, label=label, overrides=overrides,
        optimization=method, retain_stream=method != 'none')
        for path in images() for label, (source, preset, overrides, method) in variants.items()]
    save('allocation_full_plan', dict(decision=snapshot(), cases=cases, repeats=3,
        reason='Freeze two integrated settings, 500k/35 t=0.4 and t=0.8, before new holdout results. Include identical fixed-geometry settings as diagnostic ablations, five ordinary controls. The integrated 0.4 screen improves five of six matched-rate cases and all six lie inside release Q-HQ brackets; 0.8 brackets a lower rate. HQ250 gains fail the stronger existing-current frontier control; do not promote them. No new rate weights or content routing.'))


if __name__ == '__main__':
    allocation()
