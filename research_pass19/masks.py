import struct
import lzma
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

RECORD = struct.Struct('<BHHHHhh')
GROUP = struct.Struct('<4sI')


def fit(target, canvas, box):
    c, x, y, w, h = box
    residual = target[y:y+h, x:x+w, c].astype(np.int32) - canvas[y:y+h, x:x+w, c].astype(np.int32)
    positive = residual[residual > 0]
    negative = residual[residual < 0]
    lo = int(round(float(negative.mean()))) if negative.size else 0
    hi = int(round(float(positive.mean()))) if positive.size else 0
    for _ in range(4):
        neg = residual < lo / 2 if lo < 0 else np.zeros(residual.shape, dtype=bool)
        pos = residual > hi / 2 if hi > 0 else np.zeros(residual.shape, dtype=bool)
        if neg.any(): lo = min(-1, int(np.rint(residual[neg].mean())))
        if pos.any(): hi = max(1, int(np.rint(residual[pos].mean())))
    lo, hi = max(-255, lo), min(255, hi)
    mask = np.zeros(residual.shape, dtype=np.uint8)
    mask[residual < lo / 2] = 1 if lo < 0 else 0
    mask[residual > hi / 2] = 2 if hi > 0 else 0
    levels = np.array([0, lo, hi], dtype=np.int16)
    return dict(box=tuple(map(int, box)), lo=lo, hi=hi, mask=mask), levels[mask]


def record_bytes(record):
    c, x, y, w, h = record['box']
    mask = record['mask'].ravel()
    padded = np.pad(mask, (0, (-len(mask)) % 4)).reshape(-1, 4)
    packed = (padded[:,0] << 6) | (padded[:,1] << 4) | (padded[:,2] << 2) | padded[:,3]
    return RECORD.pack(c, x, y, w, h, record['lo'], record['hi']) + packed.tobytes()


def pack(records):
    raw = GROUP.pack(b'M19R', len(records)) + b''.join(record_bytes(r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=lzma.PRESET_EXTREME)


def unpack(payload, width, height):
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    raw = decoder.decompress(payload)
    if not decoder.eof or decoder.unused_data or len(raw) < GROUP.size:
        raise ValueError('mask entropy framing')
    magic, count = GROUP.unpack_from(raw)
    if magic != b'M19R' or count > 4096: raise ValueError('mask group')
    offset = GROUP.size
    records = []
    for _ in range(count):
        if offset + RECORD.size > len(raw): raise ValueError('short record')
        c, x, y, w, h, lo, hi = RECORD.unpack_from(raw, offset)
        offset += RECORD.size
        size = (w * h + 3) // 4
        if not w or not h or c > 2 or x+w > width or y+h > height or not -255 <= lo <= 0 or not 0 <= hi <= 255 or offset+size > len(raw):
            raise ValueError('invalid record')
        packed = np.frombuffer(raw, dtype=np.uint8, count=size, offset=offset)
        mask = ((packed[:, None] >> np.array([6,4,2,0], dtype=np.uint8)) & 3).ravel()
        if np.any(mask[:w*h] == 3) or np.any(mask[w*h:]): raise ValueError('invalid mask')
        records.append(dict(box=(c,x,y,w,h), lo=lo, hi=hi, mask=mask[:w*h].reshape(h,w).copy()))
        offset += size
    if offset != len(raw): raise ValueError('extra mask data')
    return records


def apply(canvas, record):
    c, x, y, w, h = record['box']
    canvas[y:y+h, x:x+w, c] += np.array([0, record['lo'], record['hi']], dtype=np.int16)[record['mask']]


def proposals(target, canvas, limit=4):
    residual = target.astype(np.int32) - canvas.astype(np.int32)
    h, w, channels = residual.shape
    size = 64
    hh, ww = (h+size-1)//size*size, (w+size-1)//size*size
    energy = residual.astype(np.float32)**2
    gradient = np.zeros_like(energy)
    gradient[1:] += np.abs(residual[1:] - residual[:-1])
    gradient[:,1:] += np.abs(residual[:,1:] - residual[:,:-1])
    score = np.pad(energy * (1 + np.minimum(gradient, 64) / 64), ((0,hh-h),(0,ww-w),(0,0)))
    scores = score.reshape(hh//size,size,ww//size,size,channels).sum(axis=(1,3),dtype=np.float64)
    order = np.argsort(-scores.ravel(), kind='stable')
    boxes = []
    anchors = []
    for index in order:
        by, bx, c = np.unravel_index(index, scores.shape)
        if scores[by,bx,c] <= 0: break
        if any(c==old[0] and abs(int(bx)-old[1])+abs(int(by)-old[2])<=1 for old in anchors): continue
        anchors.append((int(c),int(bx),int(by)))
        for extent in (64,128):
            bw, bh = min(extent,w), min(extent,h)
            x = min(w-bw,max(0,int(bx)*size+size//2-bw//2))
            y = min(h-bh,max(0,int(by)*size+size//2-bh//2))
            box = (int(c),x,y,bw,bh)
            if box not in boxes: boxes.append(box)
        if len(anchors) == 2: break
    return boxes[:limit]
