# ====================================================================================================
#   PBC3 range-coder prototype (measurement only, does NOT change the codec/format)
#
#   Purpose: answer "is a symbol-level entropy coder over the grid indices worth it?"
#
#   Background -- why this might help:
#     Every grid cell currently stores its palette index in a FIXED number of bits
#     (patch_palette_bitcount, default 2 -> 0..3). But index 0 ("no change") dominates
#     hugely, so a symbol that occurs ~70% of the time still costs a full 2 bits when it
#     ideally costs ~0.5 bits. Global LZMA recovers some of this, but it models BYTES and
#     can't go sub-bit on a tiny, skewed alphabet packed at the bit level.
#     A range/arithmetic coder encodes each symbol in ~ -log2(p) bits, so it gets close to
#     the Shannon entropy of the index distribution (the theoretical floor below).
#
#   "Structure-of-arrays (SoA) layout" just means: instead of writing each patch as
#     [header...][its grid indices], group ALL headers together and ALL grid indices
#     together. Then the entropy coder sees one long homogeneous index stream (better
#     modeling), and LZMA sees clean, separated structures. This prototype already
#     concatenates all indices to estimate that best case.
#
#   This script: encodes an image with the real PBC3, pulls out the grid indices, and
#   compares fixed-bit cost vs Shannon entropy vs an adaptive range coder vs LZMA-on-indices.
# ====================================================================================================

import sys
import lzma
import numpy as np

from PBC3 import PBC3, PBC3Config, BitReader, BitWriter

MASK = 0xFFFFFFFF
TOP = 1 << 24
BOT = 1 << 16


class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.rng = MASK
        self.out = bytearray()

    def encode(self, cum, freq, tot):
        r = self.rng // tot
        self.low = (self.low + r * cum) & MASK
        self.rng = r * freq
        while True:
            if (self.low ^ ((self.low + self.rng) & MASK)) < TOP:
                pass
            elif self.rng < BOT:
                self.rng = (-self.low) & (BOT - 1)
            else:
                break
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK
            self.rng = (self.rng << 8) & MASK

    def finish(self):
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK
        return bytes(self.out)


class RangeDecoder:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.low = 0
        self.rng = MASK
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._byte()) & MASK

    def _byte(self):
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def decode_freq(self, tot):
        self.r = self.rng // tot
        return min(tot - 1, ((self.code - self.low) & MASK) // self.r)

    def decode_update(self, cum, freq, tot):
        self.low = (self.low + self.r * cum) & MASK
        self.rng = self.r * freq
        while True:
            if (self.low ^ ((self.low + self.rng) & MASK)) < TOP:
                pass
            elif self.rng < BOT:
                self.rng = (-self.low) & (BOT - 1)
            else:
                break
            self.code = ((self.code << 8) | self._byte()) & MASK
            self.low = (self.low << 8) & MASK
            self.rng = (self.rng << 8) & MASK


class AdaptiveModel:
    def __init__(self, nsym, inc=24, limit=1 << 14):
        self.n = nsym
        self.inc = inc
        self.limit = limit
        self.freq = [1] * nsym
        self.tot = nsym

    def encode_freq(self, sym):
        return sum(self.freq[:sym]), self.freq[sym], self.tot

    def decode_sym(self, target):
        cum = 0
        for s in range(self.n):
            if cum + self.freq[s] > target:
                return s, cum, self.freq[s]
            cum += self.freq[s]
        s = self.n - 1
        return s, cum - self.freq[s], self.freq[s]

    def update(self, sym):
        self.freq[sym] += self.inc
        self.tot += self.inc
        if self.tot > self.limit:
            self.tot = 0
            for i in range(self.n):
                self.freq[i] = (self.freq[i] + 1) >> 1
                self.tot += self.freq[i]


def range_encode(symbols, nsym):
    enc = RangeEncoder()
    m = AdaptiveModel(nsym)
    for s in symbols:
        cum, freq, tot = m.encode_freq(s)
        enc.encode(cum, freq, tot)
        m.update(s)
    return enc.finish()


def range_decode(data, count, nsym):
    dec = RangeDecoder(data)
    m = AdaptiveModel(nsym)
    out = []
    for _ in range(count):
        tot = m.tot
        s, cum, freq = m.decode_sym(dec.decode_freq(tot))
        dec.decode_update(cum, freq, tot)
        m.update(s)
        out.append(s)
    return out


def selftest():
    rng = np.random.default_rng(0)
    for nsym in (2, 4, 8, 16):
        p = rng.random(nsym) ** 3
        p /= p.sum()
        syms = rng.choice(nsym, size=20000, p=p).tolist()
        enc = range_encode(syms, nsym)
        dec = range_decode(enc, len(syms), nsym)
        assert dec == syms, f"round-trip FAILED for nsym={nsym}"
    print("range coder self-test: OK (lossless round-trip)")


def iter_patch_indices(data):
    """Walk a PBC3 bitstream and yield (bitcount, flat_indices) for each patch.
    Mirrors PBC3._read_patch but captures the raw grid indices."""
    if isinstance(data, str):
        with open(data, "rb") as f:
            data = f.read()
    _, body = PBC3._open_body(data)
    br = BitReader(body)
    (_ds, _ow, _oh, _w, _h, _cs, channels, channel_bits,
     positive_bias, _alpha, patch_count, _base) = PBC3._read_header(br)
    for _ in range(patch_count):
        br.read(channel_bits)
        br.read(16); br.read(16)
        pw = br.read(16); ph = br.read(16)
        pm = br.read(1)
        if pm == PBC3.PALETTE_EXPLICIT:
            bitcount = br.read(4)
            for _ in range(1 << bitcount):
                br.read(9)
        else:
            mask_size = br.read(10)
            mask = [br.read(1) for _ in range(mask_size)]
            neg = br.read(8); pos = br.read(8); max_bc = br.read(4)
            bitcount = PBC3.resolve_palette_bitcount(mask, max_bc, neg, pos, positive_bias)
        cell = br.read(16)
        gw = PBC3._ceil_div(pw, cell)
        gh = PBC3._ceil_div(ph, cell)
        flat = PBC3._read_grid(br, gw * gh, bitcount)
        yield bitcount, np.asarray(flat, dtype=np.int64)


def lzma_indices_bits(groups):
    """Cost of just the indices, fixed-bit packed (per-patch bitcount) then LZMA'd."""
    bw = BitWriter()
    for bitcount, flat in groups:
        for v in flat:
            bw.write(int(v), bitcount)
    packed = bw.finish()
    comp = lzma.compress(packed, format=lzma.FORMAT_RAW, filters=PBC3._LZMA_FILTERS)
    return len(comp) * 8


def analyze(image, config=None):
    config = config or PBC3Config()
    result = PBC3.compress(image, config)
    data = result.data

    groups = list(iter_patch_indices(data))
    if not groups:
        print("no patches found")
        return

    fixed_bits = sum(len(flat) * bitcount for bitcount, flat in groups)
    all_idx = np.concatenate([flat for _, flat in groups])
    total_cells = int(all_idx.size)
    nsym = int(all_idx.max()) + 1 if total_cells else 1

    counts = np.bincount(all_idx, minlength=nsym).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy_per_sym = float(-(nz * np.log2(nz)).sum())
    entropy_bits = entropy_per_sym * total_cells

    rc = range_encode(all_idx.tolist(), nsym)
    rc_bits = len(rc) * 8
    assert range_decode(rc, total_cells, nsym) == all_idx.tolist(), "range round-trip failed on real data"

    lzma_bits = lzma_indices_bits(groups)
    file_bits = len(data) * 8

    def line(label, bits):
        bpc = bits / total_cells
        print(f"  {label:<34} {bits/8/1024:8.2f} KB  ({bpc:5.3f} bits/cell)")

    print(f"\nimage: {image if isinstance(image, str) else '<in-memory>'}")
    print(f"patches: {len(groups)} | index cells: {total_cells} | alphabet: {nsym} | dist: "
          + ", ".join(f"{i}:{p[i]*100:.1f}%" for i in range(nsym)))
    print(f"full compressed file (headers + indices, current pipeline): {file_bits/8/1024:.2f} KB")
    print("\nINDEX-STREAM cost comparison (this is the part a range coder would replace):")
    line("fixed bits (current, pre-LZMA)", fixed_bits)
    line("LZMA over indices only", lzma_bits)
    line("adaptive range coder", rc_bits)
    line("Shannon entropy (floor)", int(round(entropy_bits)))

    def pct(a, b):
        return (1 - a / b) * 100 if b else 0.0
    print("\nrange coder vs:")
    print(f"  fixed bits      : {pct(rc_bits, fixed_bits):+.1f}%")
    print(f"  LZMA-on-indices : {pct(rc_bits, lzma_bits):+.1f}%   <-- the number that matters")
    print(f"  (distance to entropy floor: {pct(int(round(entropy_bits)), rc_bits):+.1f}% headroom left)\n")


if __name__ == "__main__":
    selftest()
    if len(sys.argv) >= 2:
        analyze(sys.argv[1])
    else:
        print("usage: python PBC3_rangecoder_prototype.py <image>   (after self-test)")
