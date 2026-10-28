import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_pass18.diagnose import HERE, PBC3Config, describe, lf, np, ops, provenance, setup, write_json
from research_pass18.proposals import content_boxes, learned_action


def compare(row, selected):
    gain = selected['gain']
    row['gain_fraction'] = row['gain'] / gain if gain > 0 else None
    row['cost_fraction'] = row['raw_bits'] / selected['raw_bits']
    row['strict_dominance'] = row['gain'] >= gain and row['raw_bits'] <= selected['raw_bits'] and (row['gain'] > gain or row['raw_bits'] < selected['raw_bits'])
    row['near_gain_cheaper'] = gain > 0 and row['gain'] >= .98 * gain and row['raw_bits'] <= .75 * selected['raw_bits']
    row['useful_tradeoff'] = gain > 0 and row['gain'] >= .9 * gain and row['raw_bits'] <= .6 * selected['raw_bits']
    return row


def main():
    setup()
    meta = provenance()
    meta['timing'] = 'One warmed component measurement per frozen state; these are not complete-codec timings.'
    diag = json.loads((HERE / 'diagnostics.json').read_text())
    rows = []
    first = diag['states'][0]
    warm = np.load(HERE / first['file'])
    warm_config = PBC3Config(**first['config'])
    warm_rng = np.random.default_rng()
    warm_rng.bit_generator.state = copy.deepcopy(first['rng_before'])
    lf.propose_boxes(warm['target'], warm['canvas'], warm_config, warm_rng, first['selected']['box'][0], first['selected']['step'])
    content_boxes(warm['target'], warm['canvas'], first['selected']['box'][0], warm_config)
    for state in diag['states']:
        arrays = np.load(HERE / state['file'])
        target, canvas = arrays['target'], arrays['canvas']
        selected = state['selected']
        box = tuple(selected['box'])
        config = PBC3Config(**state['config'])
        channel_bits = state['channel_bits']
        model = lf.LearnedFiller.load(config.learned_filler_model_path, q_override=config.learned_filler_q)
        built = lf.build_action(target, canvas, box, config, channel_bits, selected['action'])
        assert describe(built, channel_bits) == {k: selected[k] for k in describe(built, channel_bits)}
        start = time.perf_counter()
        same = []
        pairs = [(selected['cell'], 1), (max(1, selected['cell'] // 2), 1), (selected['cell'], 2)]
        for cell, bits in dict.fromkeys(pairs):
            candidate = lf._build_action(target, canvas, box, config, channel_bits, cell, bits)
            row = compare(describe(candidate, channel_bits), selected)
            row['exactly_same_patch'] = candidate[0]['cell_size'] == built[0]['cell_size'] and candidate[0]['max_bitcount'] == built[0]['max_bitcount'] and np.array_equal(candidate[0]['indices'], built[0]['indices'])
            same.append(row)
        action_seconds = time.perf_counter() - start
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state['rng_before'])
        start = time.perf_counter()
        stochastic = lf.propose_boxes(target, canvas, config, rng, box[0], selected['step'])
        stochastic_seconds = time.perf_counter() - start
        assert stochastic == [tuple(b) for b in state['boxes']]
        assert rng.bit_generator.state == state['rng_after']
        start = time.perf_counter()
        content, proxy = content_boxes(target, canvas, box[0], config)
        content_seconds = time.perf_counter() - start
        spatial = []
        start = time.perf_counter()
        for b, score in zip(content, proxy):
            action = learned_action(model, target, canvas, b, config, selected['step'])
            learned = lf.build_action(target, canvas, b, config, channel_bits, action)
            binary = lf._build_action(target, canvas, b, config, channel_bits, 2, 1)
            for kind, candidate in [('model', learned), ('binary_cell2', binary)]:
                row = compare(describe(candidate, channel_bits), selected)
                row.update(kind=kind, action=action if kind == 'model' else None, proxy=score)
                spatial.append(row)
        exact_seconds = time.perf_counter() - start
        overlaps = []
        for b in content:
            _, x, y, w, h = b
            best_iou = 0.
            for original in stochastic:
                _, ox, oy, ow, oh = original
                intersection = max(0, min(x + w, ox + ow) - max(x, ox)) * max(0, min(y + h, oy + oh) - max(y, oy))
                best_iou = max(best_iou, intersection / (w * h + ow * oh - intersection))
            overlaps.append(best_iou)
        rows.append(dict(id=state['id'], image=selected['image'], preset=selected['preset'], step=selected['step'], selected=selected, same_region=same, action_alternative_seconds=action_seconds, spatial=spatial, existing_proposals=len(stochastic), existing_exact_evaluations=1, content_proposals=len(content), content_exact_evaluations=len(spatial), stochastic_seconds=stochastic_seconds, content_seconds=content_seconds, content_exact_seconds=exact_seconds, max_iou_to_existing=overlaps))
        print(state['id'], 'same', [(a['requested_bits'], a['cell'], round(a['gain_fraction'], 3), round(a['cost_fraction'], 3)) for a in same], 'content', len(content), flush=True)
    write_json('frozen_results.json', dict(provenance=meta, states=len(rows), rows=rows))
    provenance()


if __name__ == '__main__':
    main()
