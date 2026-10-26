import argparse

from support import ROOT, SCREEN, save, snapshot, read, run_plan

CONFIGS = {
    'q250': ('quality', {}),
    'hq250': ('high_quality', {}),
    'q500p35': ('quality', {'auto_downsample_max_pixels': 500000, 'patch_count': 35}),
}


def plan(stage):
    if stage == 'fixed_identity':
        methods = ('none', 'fixed_identity')
        repeats = 1
    elif stage == 'fixed_screen':
        methods = ('none', 'fixed_t0.1', 'fixed_t0.4', 'fixed_t0.8')
        repeats = 2
    elif stage == 'integrated_screen':
        assert read('integrated_decision')['proceed']
        methods = ('none', 'integrated_t0.1', 'integrated_t0.4', 'integrated_t0.8')
        repeats = 2
    else:
        raise ValueError(stage)
    cases = [dict(path=str(ROOT / 'hpt_data' / image), preset=preset, source='current',
        label=family + '_' + method, overrides=overrides, optimization=method)
        for image in SCREEN for family, (preset, overrides) in CONFIGS.items() for method in methods]
    save(stage + '_plan', dict(decision=snapshot(), cases=cases, repeats=repeats,
        decision_basis='44/262 cheaper static alternatives retain >=95% local gain at <=85% raw bits, but zero meet 98%/75%; no local dominance. Three normalized penalty brackets 0.1/0.4/0.8; no LZMA or extra proposals/model calls.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('fixed_identity', 'fixed_screen', 'integrated_screen'))
    args = parser.parse_args()
    plan(args.stage)
