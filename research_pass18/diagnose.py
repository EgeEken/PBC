import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_pass17.analyze_residuals import IMAGES, PRESETS, PBC3, PBC3Config, load, measure, np, provenance, setup, sha
import learned_filler as lf
import pbc3_ops as ops
from pbc3_heads import DownsampleInitHead
from pbc3_types import BitWriter

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / 'artifacts'
STEPS = (3, 8, 13, 18)


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')


def describe(built, channel_bits):
    patch, values, delta, gain, bits = built
    writer = BitWriter()
    PBC3._write_patch(writer, patch, channel_bits)
    assert len(writer.data) * 8 + writer.nbits == bits
    return dict(box=[patch[k] for k in ('channel', 'x', 'y', 'w', 'h')], cell=patch['cell_size'], requested_bits=patch['max_bitcount'], effective_bits=patch['bitcount'], grid_cells=int(patch['indices'].size), grid_bits=int(patch['indices'].size * patch['bitcount']), raw_bits=int(bits), gain=float(gain), gain_per_bit=float(gain / bits), palette=ops.palette_generator(patch['mask'], patch['max_bitcount'], patch['neg'], patch['pos']).tolist())


def main():
    setup()
    ARTIFACTS.mkdir(exist_ok=True)
    meta = provenance()
    meta['research_start_commit'] = '2e7d03e47f9db4d501b1f4d2e2bf1b46a1b17735'
    meta['timing'] = 'Instrumented decision capture only; encode walls are not used for performance claims. Exact completed streams and independent pixels checked against Pass 17.'
    context = {}
    rows, states, ledger = [], [], []
    original_init = DownsampleInitHead.select
    original_select = lf.LearnedFiller.select_patch
    original_propose = lf.propose_boxes
    original_forward = lf.LearnedFiller._forward
    original_build = lf.build_action

    def init(self, channel, target, canvas, w, h, config, channel_bits, trace=None):
        before = target - np.clip(canvas, 0, 255)
        built = original_init(self, channel, target, canvas, w, h, config, channel_bits, trace)
        patch, values, delta, cell, bits = built
        gain, cost, _ = lf._score_patch(target, canvas, (channel, 0, 0, w, h), channel_bits, patch, values)
        row = describe((patch, values, delta, gain, cost), channel_bits)
        row.update(kind='initialization', step=0, accepted=True, canvas_sse=float(np.sum(before.astype(np.int64) ** 2)), **context['identity'])
        context['ledger'].append(row)
        return built

    def select(self, target, canvas, config, rng, channel_bits, step, current_channel, trace=None, abs_error=None):
        context.update(step=step, rng_before=copy.deepcopy(rng.bit_generator.state), rng=rng)
        return original_select(self, target, canvas, config, rng, channel_bits, step, current_channel, trace, abs_error)

    def propose(*args, **kwargs):
        start = time.perf_counter()
        boxes = original_propose(*args, **kwargs)
        context.update(boxes=boxes, proposal_seconds=time.perf_counter() - start)
        return boxes

    def forward(self, features, trace=None):
        result = original_forward(self, features, trace)
        context['logits'] = result[0][0].copy()
        return result

    def build(target, canvas, box, config, channel_bits, action, trace=None):
        built = original_build(target, canvas, box, config, channel_bits, action, trace)
        row = describe(built, channel_bits)
        before = target - np.clip(canvas, 0, 255)
        row.update(kind='iterative', step=context['step'], action=action, accepted=built[3] > 0, canvas_sse=float(np.sum(before.astype(np.int64) ** 2)), action_logits=context['logits'].tolist(), proposal_rank=context['boxes'].index(box) + 1, **context['identity'])
        context['ledger'].append(row)
        if context['step'] in STEPS:
            name = f"{context['identity']['image']}.{context['identity']['preset']}.{context['step']}"
            np.savez_compressed(ARTIFACTS / (name + '.npz'), target=target, canvas=canvas)
            states.append(dict(id=name, file='artifacts/' + name + '.npz', selected=row, config=dataclasses.asdict(config), channel_bits=channel_bits, boxes=context['boxes'], proposal_seconds=context['proposal_seconds'], rng_before=context['rng_before'], rng_after=copy.deepcopy(context['rng'].bit_generator.state)))
        return built

    lf.LearnedFiller.select_patch = select
    lf.LearnedFiller._forward = forward
    lf.propose_boxes = propose
    lf.build_action = build
    DownsampleInitHead.select = init
    prior = json.loads((ROOT / 'research_pass17/baseline.json').read_text())['rows']
    try:
        for name in IMAGES:
            image = load(name)
            for preset in PRESETS:
                config = getattr(PBC3Config, preset)()
                context.clear()
                context.update(identity=dict(image=name, preset=preset), ledger=[])
                encoded = PBC3.compress(image, config=config)
                decoded = PBC3.decompress(encoded.data).image
                previous = next(r for r in prior if r['image'] == name and r['preset'] == preset)
                assert sha(encoded.data) == previous['stream_sha256']
                assert sha(np.asarray(decoded).tobytes()) == previous['pixels_sha256']
                (ARTIFACTS / (name + '.' + preset + '.pbc')).write_bytes(encoded.data)
                _, raw = PBC3._open_body(encoded.data)
                row = dict(image=name, preset=preset, bytes=len(encoded.data), raw_body_bytes=len(raw), rgb_mse=previous['rgb_mse'], stream_sha256=sha(encoded.data), pixels_sha256=previous['pixels_sha256'], config=dataclasses.asdict(config), patches=sum(r['accepted'] for r in context['ledger']), working_size=list(PBC3.prepare(image, config)['target'].shape[:2][::-1]))
                rows.append(row)
                ledger.extend(context['ledger'])
                print(name, preset, row['patches'], row['bytes'], flush=True)
    finally:
        lf.LearnedFiller.select_patch = original_select
        lf.LearnedFiller._forward = original_forward
        lf.propose_boxes = original_propose
        lf.build_action = original_build
        DownsampleInitHead.select = original_init
    (HERE / 'decisions_raw.json').write_text(json.dumps(ledger) + '\n')
    write_json('diagnostics.json', dict(provenance=meta, rows=rows, states=states, raw_ledger_sha256=sha((HERE / 'decisions_raw.json').read_bytes())))
    import csv
    columns = ['image', 'preset', 'kind', 'step', 'accepted', 'action', 'requested_bits', 'effective_bits', 'cell', 'grid_cells', 'grid_bits', 'raw_bits', 'gain', 'gain_per_bit', 'canvas_sse', 'proposal_rank', 'box']
    with (HERE / 'decisions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(ledger)
    provenance()


if __name__ == '__main__':
    main()
