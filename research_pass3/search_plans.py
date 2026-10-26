import sys

from support import ROOT, SCREEN, PRESETS, save, snapshot

stage = sys.argv[1]
methods = ('scores_verify', 'rectangles_verify') if stage == 'search_exactness' else ('baseline', 'scores', 'rectangles')
cases = [dict(path=str(ROOT / 'hpt_data' / name), preset=preset, source='current', label=method,
    retain_stream=False,
    optimization='none' if method == 'baseline' else 'search_' + method)
    for name in SCREEN for preset in PRESETS for method in methods]
save(stage + '_plan', dict(decision=snapshot(), cases=cases, repeats=1 if stage == 'search_exactness' else 3))
