import json
from pathlib import Path

from common import OUT, PASS1, ROOT, SCREEN, PRESETS, run_plan, save


def equivalence():
    rows = json.loads((OUT / 'rng_control.json').read_text())
    pairs = []
    for row in [r for r in rows if r['source'] == 'numpy_root']:
        other = next(r for r in rows if (r['source'], r['dataset'], r['image'], r['preset'], r['trial']) == ('release', row['dataset'], row['image'], row['preset'], row['trial']))
        pairs.append({k: row[k] for k in ('dataset', 'image', 'preset', 'trial')} | {
            'bytes_equal': row['stream_sha256'] == other['stream_sha256'],
            'pixels_equal': row['pixels_sha256'] == other['pixels_sha256'],
            'patches_equal': row['patch_signatures'] == other['patch_signatures'],
            'counts_equal': row['patch_count'] == other['patch_count'],
            'first_patch_difference': next((i for i, (a, b) in enumerate(zip(row['patch_signatures'], other['patch_signatures'])) if a != b), None),
            'release_encode': other['encode_seconds'], 'numpy_root_encode': row['encode_seconds'],
            'release_decode': other['decode_seconds'], 'numpy_root_decode': row['decode_seconds']})
    save('rng_equivalence', pairs)
    print('Equal cases', sum(p['bytes_equal'] for p in pairs), '/', len(pairs))


def profile():
    cases = [dict(path=str(ROOT / 'hpt_data' / name), preset=p, source=s, label=s, profile=True)
        for name in ('flower.jpg', 'monet.jpg', 'googlewiki.png') for p in PRESETS for s in ('release', 'numpy_root')]
    run_plan('rng_profile', cases, 2)


def discrepancy():
    import os
    import sys
    from common import THREAD_ENV
    os.environ.update(THREAD_ENV)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import psutil
    from PIL import Image, ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    from stream_tools import parse, serialize, patch_signature
    import pbc3_ops as ops
    from learned_filler import LearnedFiller
    import learned_filler

    psutil.Process().cpu_affinity([0])
    source = ImageOps.exif_transpose(Image.open(ROOT / 'hpt_data' / 'googlewiki.png')).convert('RGBA')
    remote = (PASS1 / 'space_googlewiki.quality.pbc3').read_bytes()
    local = (PASS1 / 'local_rgba_googlewiki.quality.pbc3').read_bytes()
    rh, rp, _ = parse(remote)
    lh, lp, _ = parse(local)
    first = next(i for i, (a, b) in enumerate(zip(rp, lp)) if patch_signature(a) != patch_signature(b))
    assert rh == lh
    assert serialize(rh, rp[:first]) == serialize(lh, lp[:first])
    prep = PBC3.prepare(source, PBC3Config.quality())
    target = prep['target']
    canvas = PBC3._decode_to_canvas(serialize(lh, lp[:first]))[0]
    residuals = [float(np.sum(np.abs(target[:, :, c] - np.clip(canvas[:, :, c], 0, 255)))) for c in range(4)]
    cache = LearnedFiller._forward
    rng = ops.PBC3Rng
    propose = learned_filler.propose_boxes
    proposal_observation = {}
    def observed_propose(target, canvas, config, generator, channel, step, trace=None):
        if step == first - lh[6] + 1 and not proposal_observation:
            visible = np.clip(canvas[:, :, channel], 0, 255).astype(np.int32)
            error = np.abs(target[:, :, channel] - visible)
            anchors = ops.top_anchors(error, config.top_k, config.anchor_block_size, channel)
            scores, _, _ = ops._anchor_block_scores_kernel(np.ascontiguousarray(error), config.anchor_block_size)
            proposal_observation.update(step=step, rng_state=generator.state, anchors=anchors,
                top_anchor_score_tie_count=int(np.sum(scores == np.max(scores))))
        boxes = propose(target, canvas, config, generator, channel, step, trace=trace)
        if step == first - lh[6] + 1 and 'ranked_boxes' not in proposal_observation:
            proposal_observation['ranked_boxes'] = boxes
        return boxes
    results = []
    for variant in ('default_warm_cache', 'numpy_rng', 'float64_mlp'):
        if variant == 'default_warm_cache':
            learned_filler.propose_boxes = observed_propose
        if variant == 'numpy_rng':
            ops.PBC3Rng = np.random.default_rng
        if variant == 'float64_mlp':
            def forward(self, x, trace=None):
                h = x.astype(np.float64)
                for weight, bias in self.layers:
                    h = h @ weight.astype(np.float64).T + bias.astype(np.float64)
                    h = h / (1 + np.exp(-h))
                return h @ self.Wa.astype(np.float64).T + self.ba, h @ self.Wv.astype(np.float64).T + self.bv[None, :]
            LearnedFiller._forward = forward
        try:
            result = PBC3.compress(source, config=PBC3Config.quality(learned_filler_model_path=str(ROOT / 'patch_policy.npz')))
        finally:
            ops.PBC3Rng = rng
            LearnedFiller._forward = cache
            learned_filler.propose_boxes = propose
        _, patches, _ = parse(result.data)
        results.append({'variant': variant, 'remote_equal': result.data == remote, 'local_equal': result.data == local,
            'first_difference_from_remote': next((i for i, (a, b) in enumerate(zip(patches, rp)) if patch_signature(a) != patch_signature(b)), None)})
    save('hf_discrepancy', {'headers_equal': rh == lh, 'first_differing_patch': first,
        'all_earlier_quantized_semantics_equal': True, 'channel_errors_before_difference': residuals,
        'remote_patch_geometry': {k: rp[first][k] for k in ('channel', 'x', 'y', 'w', 'h', 'cell_size')},
        'local_patch_geometry': {k: lp[first][k] for k in ('channel', 'x', 'y', 'w', 'h', 'cell_size')}, 'variants': results,
        'local_proposal_observation': proposal_observation,
        'remote_requests': 0, 'versions_available': 'Only installed local stack; no dependency installation or remote introspection'})


def cpu_variant():
    import os
    import sys
    from common import THREAD_ENV, sha
    os.environ.update(THREAD_ENV)
    sys.path.insert(0, str(ROOT))
    import psutil
    from PIL import Image, ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    from stream_tools import parse, patch_signature
    psutil.Process().cpu_affinity([0])
    image = ImageOps.exif_transpose(Image.open(ROOT / 'hpt_data' / 'googlewiki.png')).convert('RGBA')
    result = PBC3.compress(image, config=PBC3Config.quality(learned_filler_model_path=str(ROOT / 'patch_policy.npz')))
    remote = (PASS1 / 'space_googlewiki.quality.pbc3').read_bytes()
    local = (PASS1 / 'local_rgba_googlewiki.quality.pbc3').read_bytes()
    _, patches, _ = parse(result.data)
    _, reference, _ = parse(remote)
    save('hf_cpu_variant', dict(disabled_numpy_features=os.environ.get('NPY_DISABLE_CPU_FEATURES'),
        remote_equal=result.data == remote, local_equal=result.data == local, stream_sha256=sha(result.data),
        first_difference_from_remote=next((i for i, (a, b) in enumerate(zip(patches, reference)) if patch_signature(a) != patch_signature(b)), None)))


def tie_analysis():
    import os
    import sys
    from common import THREAD_ENV
    os.environ.update(THREAD_ENV)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from PIL import Image, ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    from stream_tools import parse, serialize
    import pbc3_ops as ops
    diagnostic = json.loads((OUT / 'hf_discrepancy.json').read_text())
    header, patches, _ = parse((PASS1 / 'local_rgba_googlewiki.quality.pbc3').read_bytes())
    image = ImageOps.exif_transpose(Image.open(ROOT / 'hpt_data' / 'googlewiki.png')).convert('RGBA')
    config = PBC3Config.quality()
    target = PBC3.prepare(image, config)['target']
    canvas = PBC3._decode_to_canvas(serialize(header, patches[:diagnostic['first_differing_patch']]))[0]
    error = np.abs(target[:, :, 1] - np.clip(canvas[:, :, 1], 0, 255).astype(np.int32))
    integral = ops.integral(error)
    scores, _, _ = ops._anchor_block_scores_kernel(np.ascontiguousarray(error), config.anchor_block_size)
    anchors = ops.top_anchors(error, config.top_k, config.anchor_block_size, 1)
    rng = ops.PBC3Rng(config.random_seed)
    rng.state = diagnostic['local_proposal_observation']['rng_state']
    boxes, sums, areas = [], [], []
    for i in range(config.search_depth):
        c, x, y, w, h, _, _ = ops.sample_box(rng, anchors[i % len(anchors)], header[3], header[4], config)
        value = integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]
        if value > 0:
            boxes.append((c, x, y, w, h))
            sums.append(float(value))
            areas.append(float(w * h))
    q = ops.interp(config.search_q_start, config.search_q_end, 11, config.patch_count)
    prescores = q * ops.norm(sums) - (1 - q) * ops.norm(areas)
    order = sorted(ops.select_top_indices(prescores, config.proposal_depth), key=lambda i: prescores[i], reverse=True)
    ranked = [boxes[i] for i in order]
    assert [list(b) for b in ranked] == diagnostic['local_proposal_observation']['ranked_boxes']
    remote = diagnostic['remote_patch_geometry']
    box = tuple(remote[k] for k in ('channel', 'x', 'y', 'w', 'h'))
    largest = np.sort(scores)[-config.top_k:]
    save('hf_tie_analysis', dict(remote_box_in_local_200_pool=box in boxes,
        remote_box_rank_in_local_top50=ranked.index(box) if box in ranked else None,
        remote_width_height_in_local_pool=any((b[3], b[4]) == (box[3], box[4]) for b in boxes),
        top20_anchor_tied_scores=[float(v) for v in np.unique(largest) if np.sum(scores == v) > 1],
        cutoff_anchor_score_multiplicity=int(np.sum(scores == largest[0])),
        top_proposal_score_tie_count=int(np.sum(prescores == np.max(prescores))),
        first_second_proposal_score_gap=float(prescores[order[0]] - prescores[order[1]]),
        source='Replay only retained common prefix/local recorded RNG state; no encode and no remote request'))


if __name__ == '__main__':
    import sys
    from common import snapshot
    mode = sys.argv[1]
    before = snapshot() if mode in ('discrepancy', 'cpu_variant', 'tie_analysis') else None
    globals()[mode]()
    if before:
        save('hf_' + mode + '_provenance', dict(before=before, after=snapshot()))
