import hashlib
import json
import os
import sys

from benchmark import ROOT, OUT, THREAD_ENV

os.environ.update(THREAD_ENV)
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageOps
from PBC3 import PBC3
from pbc3_types import PBC3Config, BitReader


if __name__ == '__main__':
    path = ROOT / 'hpt_data' / 'googlewiki.png'
    source = ImageOps.exif_transpose(Image.open(path)).convert('RGBA')
    reference = np.asarray(source).astype(np.float64)
    rows = []
    for remote in json.loads((OUT / 'deployment_encodes.json').read_text()):
        preset = remote['preset']
        config = getattr(PBC3Config, preset)(learned_filler_model_path=str(ROOT / 'patch_policy.npz'))
        local = PBC3.compress(source, config=config)
        data = (OUT / ('space_googlewiki.' + preset + '.pbc3')).read_bytes()
        decoded = np.asarray(PBC3.decompress(data).image.convert('RGBA'))
        local_decoded = np.asarray(PBC3.decompress(local.data).image.convert('RGBA'))
        assert decoded.shape == reference.shape
        rgb_mse = float(np.mean((reference[:, :, :3] - decoded[:, :, :3].astype(np.float64)) ** 2))
        rgba_mse = float(np.mean((reference - decoded.astype(np.float64)) ** 2))
        _, body = PBC3._open_body(data)
        header = PBC3._read_header(BitReader(body))
        _, local_body = PBC3._open_body(local.data)
        remote_reader, local_reader = BitReader(body), BitReader(local_body)
        remote_header = PBC3._read_header(remote_reader)
        local_header = PBC3._read_header(local_reader)
        first_difference = None
        for index in range(min(remote_header[10], local_header[10])):
            remote_patch = PBC3._read_patch(remote_reader, remote_header[7], remote_header[8])
            local_patch = PBC3._read_patch(local_reader, local_header[7], local_header[8])
            if first_difference is None and any(not np.array_equal(a, b) for a, b in zip(remote_patch, local_patch)):
                first_difference = {'patch_index': index, 'remote_geometry': list(remote_patch[:6]), 'local_geometry': list(local_patch[:6]),
                                    'remote_grid_shape': list(remote_patch[6].shape), 'local_grid_shape': list(local_patch[6].shape),
                                    'grid_values_equal': bool(np.array_equal(remote_patch[6], local_patch[6]))}
        row = {'preset': preset, 'source_mode': source.mode, 'width': source.width, 'height': source.height,
               'remote_bytes_equal_local': data == local.data,
               'remote_decoded_pixels_equal_local': bool(np.array_equal(decoded, local_decoded)),
               'remote_preview_equals_decoded_rgb': hashlib.sha256(decoded[:, :, :3].tobytes()).hexdigest() == remote['pixels_sha256'],
               'remote_displayed_mse': remote['mse'], 'decoded_rgb_mse': rgb_mse, 'decoded_rgba_mse': rgba_mse,
               'local_decoded_rgb_mse': float(np.mean((reference[:, :, :3] - local_decoded[:, :, :3].astype(np.float64)) ** 2)),
               'local_final_bytes': len(local.data), 'headers_equal': remote_header == local_header,
               'first_differing_decoded_patch': first_difference,
               'differing_pixel_count': int(np.count_nonzero(np.any(decoded != local_decoded, axis=2))),
               'remote_displayed_mse_matches_rgba': remote['mse'] == round(rgba_mse, 2),
               'remote_displayed_kb_matches_bytes': remote['compressed_kb'] == round(len(data) / 1024, 2),
               'raw_bytes': len(body) + 6, 'final_bytes': len(data), 'bpp': 8 * len(data) / (source.width * source.height),
               'patch_count': header[10], 'channels': header[6], 'remote_stream_sha256': hashlib.sha256(data).hexdigest(),
               'local_stream_sha256': hashlib.sha256(local.data).hexdigest(),
               'remote_reported_encode_seconds': remote['time_seconds'], 'http_wall_seconds': remote['http_wall_seconds']}
        assert row['remote_preview_equals_decoded_rgb']
        assert row['remote_displayed_mse_matches_rgba']
        assert row['remote_displayed_kb_matches_bytes']
        rows.append(row)
        (OUT / ('local_rgba_googlewiki.' + preset + '.pbc3')).write_bytes(local.data)
        print(preset, 'bytes_equal', row['remote_bytes_equal_local'], 'pixels_equal', row['remote_decoded_pixels_equal_local'], flush=True)
    assert len(rows) == 4
    (OUT / 'deployment_comparison.json').write_text(json.dumps(rows, indent=2))
