import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'outputs'
VIS = HERE / 'visuals'
VIS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'research_pass42'))
from pbc4_v2_stream import unpack_v2
from pbc4_stream import unpack_bits


def mode_image(path):
    header, payload = unpack_v2(path.read_bytes())
    coarse_len = int.from_bytes(payload[:4], 'little')
    pos = 4 + coarse_len
    rows = (header['height'] + 15) // 16
    cols = (header['width'] + 15) // 16
    modes = np.asarray(unpack_bits(payload[pos:pos + (rows * cols * 2 + 7) // 8], rows * cols, 2), dtype=np.uint8).reshape(rows, cols)
    colors = np.array([[30, 150, 220], [60, 195, 90], [240, 95, 50]], dtype=np.uint8)
    rgb = colors[np.minimum(modes, 2)]
    return Image.fromarray(rgb).resize((header['width'], header['height']), Image.Resampling.NEAREST)


def sheet(name, crop):
    stem = pathlib.Path(name).stem
    paths = [
        ('Source', ROOT / 'hpt_data' / name),
        ('V3.1 Compression', ROOT / 'research_pass42' / 'outputs' / f'{stem}_v31_compression_fresh.png'),
        ('V4-A D low', ROOT / 'research_pass42' / 'outputs' / f'{stem}_v4_D_low_fresh.png'),
        ('V4-B D low', OUT / f'{stem}_v4_v2_low_p3003.png'),
        ('V4-B coarse base', OUT / f'{stem}_v31_prefix_003.png'),
    ]
    mode_path = OUT / f'{stem}_v4_v2_low_p3003.pbc'
    images = [(label, Image.open(path).convert('RGB')) for label, path in paths]
    images.append(('V4-B mode map (blue SKIP, green procedural, red transform)', mode_image(mode_path)))
    x0, y0, width, height = crop
    panel_w, panel_h = width, height + 34
    sheet = Image.new('RGB', (panel_w * 3, panel_h * 2), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(images):
        image = image.crop((x0, y0, x0 + width, y0 + height))
        x, y = (i % 3) * panel_w, (i // 3) * panel_h
        sheet.paste(image, (x, y + 34))
        draw.text((x + 6, y + 8), label, fill='black')
    target = VIS / f'{stem}_low_rate_comparison.png'
    sheet.save(target)
    print(target)


if __name__ == '__main__':
    sheet('coast.jpg', (1750, 1850, 768, 512))
    sheet('googlewiki.png', (120, 120, 512, 512))
