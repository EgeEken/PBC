import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_pass18.diagnose import ARTIFACTS, HERE, PBC3, PBC3Config, describe, lf, load, np, ops, provenance, setup, sha, write_json
from pbc3_types import BitReader, BitWriter


def patch_roundtrip(built, channel_bits):
    patch, values, delta, gain, bits = built
    writer = BitWriter()
    PBC3._write_patch(writer, patch, channel_bits)
    assert len(writer.data) * 8 + writer.nbits == bits
    channel, x, y, w, h, cell, recovered, effective = PBC3._read_patch(BitReader(writer.finish()), channel_bits)
    assert [channel, x, y, w, h, cell, effective] == [patch[k] for k in ('channel', 'x', 'y', 'w', 'h', 'cell_size', 'bitcount')]
    assert np.array_equal(recovered, values)
    assert np.array_equal(ops.signed_resample(recovered, h, w), delta)


def main():
    setup()
    meta = provenance()
    meta['timing'] = 'Correctness verification only; no timing claim.'
    diagnostics = json.loads((HERE / 'diagnostics.json').read_text())
    frozen = json.loads((HERE / 'frozen_results.json').read_text())
    ledger = json.loads((HERE / 'decisions_raw.json').read_text())
    assert sha((HERE / 'decisions_raw.json').read_bytes()) == diagnostics['raw_ledger_sha256']
    files = []
    for row in diagnostics['rows']:
        data = (ARTIFACTS / (row['image'] + '.' + row['preset'] + '.pbc')).read_bytes()
        assert len(data) == row['bytes'] and sha(data) == row['stream_sha256']
        decoded = PBC3.decompress(data).image
        assert sha(np.asarray(decoded).tobytes()) == row['pixels_sha256']
        error = np.asarray(load(row['image']), dtype=np.int32) - np.asarray(decoded, dtype=np.int32)
        assert float(np.mean(error**2, dtype=np.float64)) == row['rgb_mse']
        _, body = PBC3._open_body(data)
        reader = BitReader(body)
        header = PBC3._read_header(reader)
        channel_bits, count = header[7], header[10]
        accepted = [r for r in ledger if r['image'] == row['image'] and r['preset'] == row['preset'] and r['accepted']]
        assert count == len(accepted) == row['patches']
        bits_read = reader.i * 8 - reader.nbits
        header_bits = bits_read
        for decision in accepted:
            c, x, y, w, h, cell, values, effective = PBC3._read_patch(reader, channel_bits)
            assert [c, x, y, w, h] == decision['box']
            assert [cell, effective, int(values.size)] == [decision['cell'], decision['effective_bits'], decision['grid_cells']]
            current = reader.i * 8 - reader.nbits
            assert current - bits_read == decision['raw_bits']
            bits_read = current
        assert 0 <= len(body) * 8 - bits_read < 8
        files.append(dict(image=row['image'], preset=row['preset'], file_sha256=sha(data), pixels_sha256=row['pixels_sha256'], patches=count, header_bits=header_bits, patch_bits=sum(r['raw_bits'] for r in accepted), padding_bits=len(body)*8-bits_read))
    roundtrips = 0
    for state, result in zip(diagnostics['states'], frozen['rows']):
        assert state['id'] == result['id']
        arrays = np.load(HERE / state['file'])
        target, canvas = arrays['target'], arrays['canvas']
        config = PBC3Config(**state['config'])
        for candidate in [result['selected']] + result['same_region'] + result['spatial']:
            built = lf._build_action(target, canvas, tuple(candidate['box']), config, state['channel_bits'], candidate['cell'], candidate['requested_bits'])
            assert describe(built, state['channel_bits']) == {k:candidate[k] for k in describe(built, state['channel_bits'])}
            patch_roundtrip(built, state['channel_bits'])
            roundtrips += 1
    resolution = json.loads((HERE / 'representation.json').read_text())['googlewiki']['higher_resolution']
    resolution_data = (ARTIFACTS / 'googlewiki.png.high_quality_500k.pbc').read_bytes()
    resolution_pixels = np.asarray(PBC3.decompress(resolution_data).image)
    assert sha(resolution_data) == resolution['stream_sha256'] and len(resolution_data) == resolution['bytes']
    assert sha(resolution_pixels.tobytes()) == resolution['pixels_sha256']
    difference = np.asarray(load('googlewiki.png'), dtype=np.int32) - resolution_pixels.astype(np.int32)
    assert float(np.mean(difference**2, dtype=np.float64)) == resolution['rgb_mse']
    code = "import sys,hashlib;sys.path.insert(0,sys.argv[1]);from PBC3 import PBC3;from learned_filler import LearnedFiller;LearnedFiller.load=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('policy loaded')));import numpy as np;print(hashlib.sha256(np.asarray(PBC3.decompress(sys.argv[2]).image).tobytes()).hexdigest())"
    for name, preset in [('googlewiki.png','high_quality'),('cat.jpg','quality')]:
        pixels = subprocess.check_output([sys.executable, '-c', code, str(ROOT), str(ARTIFACTS / (name+'.'+preset+'.pbc'))], cwd=ROOT, text=True).strip()
        assert pixels == next(r['pixels_sha256'] for r in diagnostics['rows'] if r['image']==name and r['preset']==preset)
    meta['research_source_sha256'] = {p.name:sha(p.read_bytes()) for p in sorted(HERE.glob('*.py'))}
    write_json('validation.json', dict(provenance=meta, baseline_files=files, higher_resolution_file=resolution, patch_roundtrips=roundtrips, fresh_process_policy_forbidden_cases=2, captured_states=len(frozen['rows']), production_unchanged=True))
    provenance()
    print('baseline files',len(files),'patch roundtrips',roundtrips,'file-only subprocesses',2,flush=True)


if __name__ == '__main__':
    main()
