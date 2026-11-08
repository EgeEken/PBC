import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

from PIL import Image, ImageOps
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
START = pathlib.Path.cwd()
FROZEN = HERE / 'outputs' / 'v31_frozen'
os.chdir(FROZEN)
sys.path.insert(0, str(FROZEN))
from PBC3 import PBC3
from pbc3_types import PBC3Config


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source_path = (START / sys.argv[1]).resolve()
    preset = sys.argv[2]
    output_prefix = (START / sys.argv[3]).resolve()
    cfg = getattr(PBC3Config, preset)()
    with Image.open(source_path) as im:
        source = np.asarray(ImageOps.exif_transpose(im).convert('RGB'), dtype=np.uint8)
        pil_source = Image.fromarray(source)
    # Warm the exact config and file path once; timings below include full encode and decode calls.
    warm = PBC3.compress(pil_source, config=cfg, rng_mode='release')
    PBC3.decompress(warm.data)
    rows = []
    for repeat in range(1, 4):
        t0 = time.perf_counter()
        result = PBC3.compress(pil_source, config=cfg, rng_mode='release')
        encode_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        decoded = PBC3.decompress(result.data)
        decode_seconds = time.perf_counter() - t0
        pixels = np.asarray(decoded.image.convert('RGB'), dtype=np.uint8)
        diff = source.astype(np.int32) - pixels.astype(np.int32)
        mse = float(np.mean(diff.astype(np.int64) ** 2))
        target = output_prefix.with_name(output_prefix.name + f'_r{repeat}.pbc')
        target.write_bytes(result.data)
        rows.append({'repeat':repeat,'bytes':len(result.data),'rgb_mse':mse,
                     'rgb_psnr':float('inf') if mse == 0 else 10*np.log10(255.0**2/mse),
                     'encode_seconds':encode_seconds,'decode_seconds':decode_seconds,
                     'file_sha256':digest(result.data),'pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                     'decoded_dimensions':[int(pixels.shape[1]),int(pixels.shape[0])]})
    print(json.dumps({'source':source_path.name,'preset':preset,'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
                      'frozen_source_commit':'8006f30debfc4a57289df06df21636aeb52baffd',
                      'model_sha256':digest((FROZEN/'patch_policy.npz').read_bytes()),
                      'rows':rows},separators=(',',':')))

if __name__=='__main__': main()
