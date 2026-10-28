import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_pass18.diagnose import HERE, PBC3, PBC3Config, describe, lf, load, np, ops, provenance, setup, sha, write_json
from research_pass18.proposals import content_boxes


def fit_map(target, canvas, channel=0):
    size = 32
    h, w = target.shape[:2]
    hh, ww = h // size * size, w // size * size
    residual = (target[:hh, :ww, channel] - canvas[:hh, :ww, channel]).astype(np.float32)
    tiles = residual.reshape(hh // size, size, ww // size, size).transpose(0, 2, 1, 3)
    upper = tiles > tiles.mean(axis=(2, 3), keepdims=True)
    n = upper.sum(axis=(2, 3))
    sums = tiles.sum(axis=(2, 3))
    high = (tiles * upper).sum(axis=(2, 3))
    low = sums - high
    m1, m0 = high / np.maximum(n, 1), low / np.maximum(size * size - n, 1)
    total = (tiles * tiles).sum(axis=(2, 3))
    variance = total - sums * sums / (size * size)
    fit = 1 - (total - high * m1 - low * m0) / np.maximum(variance, 1)
    eligible = (m1 - m0 >= 24) & (n >= size * size * .05) & (n <= size * size * .95)
    return dict(tiles=int(fit.size), contrast_eligible=int(eligible.sum()), binary_tiles=int(((fit >= .8) & eligible).sum()), fit_median=float(np.median(fit[eligible])) if eligible.any() else None, fit_max=float(fit[eligible].max()) if eligible.any() else None)


def main():
    setup()
    meta = provenance()
    meta['timing'] = 'Representation and single higher-resolution diagnostic only; no runtime claim.'
    diagnostics = json.loads((HERE / 'diagnostics.json').read_text())
    rows = []
    for state in diagnostics['states']:
        arrays = np.load(HERE / state['file'])
        target, canvas = arrays['target'], arrays['canvas']
        selected = state['selected']
        c, x, y, w, h = selected['box']
        residual = target[y:y+h, x:x+w, c] - canvas[y:y+h, x:x+w, c]
        small = np.rint(ops.project_residual_to_grid(residual, selected['cell'])).astype(np.int16)
        centers = np.array([small.min(), small.max()], dtype=np.float64)
        for _ in range(4):
            group = np.abs(small - centers[1]) < np.abs(small - centers[0])
            for k in (0, 1):
                subset = small[group == bool(k)]
                if subset.size:
                    centers[k] = subset.mean()
        centers = np.rint(np.clip(centers, -255, 255)).astype(np.int16)
        group = np.abs(small.astype(np.int32) - centers[1]) < np.abs(small.astype(np.int32) - centers[0])
        values = centers[group.astype(np.int8)]
        native = lf._build_action(target, canvas, tuple(selected['box']), PBC3Config(**state['config']), state['channel_bits'], selected['cell'], 1)
        gain, _, _ = lf._score_patch(target, canvas, tuple(selected['box']), state['channel_bits'], native[0], values)
        neg_sse = np.sum(np.minimum(residual.astype(np.int64), 0) ** 2)
        pos_sse = np.sum(np.maximum(residual.astype(np.int64), 0) ** 2)
        rows.append(dict(id=state['id'], native_1bit=describe(native, state['channel_bits']), selected_gain=selected['gain'], free_levels=centers.tolist(), free_levels_interpolated_gain=gain, free_level_gain_fraction=gain/selected['gain'] if selected['gain']>0 else None, negative_residual_sse_fraction=float(neg_sse/max(1, neg_sse+pos_sse)), residual_fit=fit_map(target, canvas, c)))
    image = load('googlewiki.png')
    config = PBC3Config.high_quality()
    prepared = PBC3.prepare(image, config)
    original = np.asarray(image.convert('YCbCr'), dtype=np.int32)
    maps = dict(source_luminance=fit_map(original, np.zeros_like(original)), working250k_luminance=fit_map(prepared['target'], np.zeros_like(prepared['target'])))
    original_select = lf.LearnedFiller.select_patch
    captures = []

    def capture(self, target, canvas, config, rng, channel_bits, step, channel, trace=None, abs_error=None):
        result = original_select(self, target, canvas, config, rng, channel_bits, step, channel, trace, abs_error)
        if step in (3, 8, 13, 18):
            patch, values, delta = result
            captures.append(dict(step=step, channel=channel, residual_fit=fit_map(target, canvas, channel), content_proposals=len(content_boxes(target, canvas, channel, config)[0]), selected_cell=patch['cell_size'] if patch else None, selected_bits=patch['max_bitcount'] if patch else None))
        return result

    lf.LearnedFiller.select_patch = capture
    try:
        config = PBC3Config.high_quality(auto_downsample_max_pixels=500000)
        encoded = PBC3.compress(image, config=config)
    finally:
        lf.LearnedFiller.select_patch = original_select
    decoded = PBC3.decompress(encoded.data).image
    error = np.asarray(image, dtype=np.float32) - np.asarray(decoded, dtype=np.float32)
    (HERE / 'artifacts/googlewiki.png.high_quality_500k.pbc').write_bytes(encoded.data)
    working = PBC3.prepare(image, config)['target']
    maps['working500k_luminance'] = fit_map(working, np.zeros_like(working))
    write_json('representation.json', dict(provenance=meta, note='Free two-level means are an unencoded representation diagnostic, not a legal v0 candidate or completed-file RD result. Four deterministic Lloyd updates on the same projected grid; normal bicubic reconstruction and exact working SSE.', rows=rows, googlewiki=dict(maps=maps, higher_resolution=dict(config=config.__dict__, bytes=len(encoded.data), stream_sha256=sha(encoded.data), pixels_sha256=sha(np.asarray(decoded).tobytes()), rgb_mse=float(np.mean(error**2, dtype=np.float64)), working_size=[working.shape[1],working.shape[0]], states=captures))))
    provenance()
    print(json.dumps(dict(maps=maps, higher_resolution=captures)), flush=True)


if __name__ == '__main__':
    main()
