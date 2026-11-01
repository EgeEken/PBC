from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

PARAM_LENGTHS = (0, 3, 0, 0, 48, 3, 3)


def cell_context(index: int, width: int, cell: int = 16) -> int:
    cols = math.ceil(width / cell)
    row, col = divmod(index, cols)
    if row == 0 and col == 0:
        return 0
    if row == 0:
        return 1
    if col == 0:
        return 2
    return 3


def samples_from_modes(modes: np.ndarray, params: bytes, width: int, height: int, stream_sha256: str = '') -> dict:
    modes = np.asarray(modes, np.uint8).reshape(-1)
    cols, rows = math.ceil(width / 16), math.ceil(height / 16)
    if modes.size != cols * rows:
        raise ValueError('mode count does not match dimensions')
    return {'modes': modes, 'params': bytes(params), 'width': int(width), 'height': int(height),
            'stream_sha256': stream_sha256}


class HistogramRateModel:
    FORMAT = 'pbc32-histogram-r1-v1'

    def __init__(self, payload: dict):
        if payload.get('format') != self.FORMAT:
            raise ValueError('unsupported rate model')
        self.alpha = float(payload['alpha'])
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        self.mode_counts = np.asarray(payload['mode_counts'], np.float64)
        if self.mode_counts.shape != (4, 8, 7) or np.any(self.mode_counts < 0):
            raise ValueError('invalid mode histogram')
        self.mode_totals = self.mode_counts.sum(axis=2)
        self.param_counts = {}
        for mode_s, slots in payload['parameter_counts'].items():
            mode = int(mode_s)
            self.param_counts[mode] = {}
            for slot_s, pairs in slots.items():
                slot = int(slot_s)
                dense = np.zeros(256, np.float64)
                for symbol, count in pairs:
                    if not 0 <= int(symbol) <= 255 or int(count) <= 0:
                        raise ValueError('invalid parameter histogram entry')
                    dense[int(symbol)] = int(count)
                self.param_counts[mode][slot] = dense
        self.payload = payload

    @classmethod
    def fit(cls, samples: list[dict], alpha: float = 0.5) -> 'HistogramRateModel':
        if not samples or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('nonempty calibration samples and positive smoothing are required')
        mode_counts = np.zeros((4, 8, 7), np.int64)
        param_counts: dict[int, dict[int, np.ndarray]] = {}
        for sample in samples:
            modes = np.asarray(sample['modes'], np.uint8).reshape(-1)
            width, height = int(sample['width']), int(sample['height'])
            if width < 1 or height < 1 or modes.size != math.ceil(width / 16) * math.ceil(height / 16):
                raise ValueError('invalid calibration dimensions')
            cursor = 0
            previous = 7
            raw = sample['params']
            for i, mode_v in enumerate(modes):
                mode = int(mode_v)
                if mode > 6:
                    raise ValueError('invalid mode symbol')
                context = cell_context(i, width)
                mode_counts[context, previous, mode] += 1
                length = PARAM_LENGTHS[mode]
                if cursor + length > len(raw):
                    raise ValueError('truncated calibration parameters')
                if length:
                    slots = param_counts.setdefault(mode, {})
                    values = np.frombuffer(raw[cursor:cursor + length], np.uint8)
                    for slot, value in enumerate(values):
                        hist = slots.setdefault(slot, np.zeros(256, np.int64))
                        hist[int(value)] += 1
                cursor += length
                previous = mode
            if cursor != len(raw):
                raise ValueError('unused calibration parameters')
        encoded_params = {}
        for mode, slots in param_counts.items():
            encoded_params[str(mode)] = {}
            for slot, hist in slots.items():
                encoded_params[str(mode)][str(slot)] = [[int(v), int(hist[v])] for v in np.flatnonzero(hist)]
        payload = {'format': cls.FORMAT, 'alpha': float(alpha), 'context_definition': 'cell boundary class: top-left, top, left, interior; mode cost conditioned on previous selected mode (start=7)',
                   'mode_counts': mode_counts.tolist(), 'parameter_counts': encoded_params,
                   'calibration_streams': sorted(s['stream_sha256'] for s in samples if s.get('stream_sha256')),
                   'calibration_sample_count': len(samples)}
        return cls(payload)

    def cost_parts(self, mode: int, previous_mode: int, context: int, params: bytes) -> tuple[float, float]:
        if not 0 <= mode <= 6 or not 0 <= context <= 3 or previous_mode not in range(8):
            raise ValueError('rate-model context outside bounds')
        row = self.mode_counts[context, previous_mode]
        mode_cost = -math.log2((row[mode] + self.alpha) / (row.sum() + self.alpha * 7))
        length = PARAM_LENGTHS[mode]
        if len(params) != length:
            raise ValueError('parameter length does not match mode')
        param_cost = 0.0
        if length:
            values = np.frombuffer(params, np.uint8)
            for slot, value in enumerate(values):
                hist = self.param_counts.get(mode, {}).get(slot)
                count = 0.0 if hist is None else hist[int(value)]
                total = 0.0 if hist is None else hist.sum()
                param_cost -= math.log2((count + self.alpha) / (total + self.alpha * 256))
        if not math.isfinite(mode_cost + param_cost):
            raise ValueError('nonfinite empirical symbol estimate')
        return mode_cost, param_cost

    def save(self, path: Path) -> str:
        data = (json.dumps(self.payload, sort_keys=True, separators=(',', ':')) + '\n').encode()
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def load(cls, path: Path) -> tuple['HistogramRateModel', str]:
        data = path.read_bytes()
        return cls(json.loads(data)), hashlib.sha256(data).hexdigest()