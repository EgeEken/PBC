# ====================================================================================================
#
#           PBC v3.0 - Probabilistic Brush Compression
#           Lossy Image Compression Algorithm by EgeEken (github.com/EgeEken)
#           3.0 Update - 2026-06 - Whole algorithm overhaul
#
# ====================================================================================================
import lzma
import math
import time

import numpy as np
from PIL import Image

import pbc3_ops as ops
from pbc3_heads import DownsampleInitHead, FillerHead, SearchHead
from pbc3_types import BitReader, BitWriter, PBC3Config, PBC3Result


class PBC3:
    MAGIC = b"PBC3"
    VERSION = 0
    PALETTE_GENERATED = 0
    PALETTE_EXPLICIT = 1
    ENTROPY_STORE = 0
    ENTROPY_LZMA = 2
    _LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": lzma.PRESET_EXTREME}]
    COLOR_SPACES = {"RGB": 0, "YCbCr": 1}
    COLOR_SPACE_NAMES = {0: "RGB", 1: "YCbCr"}
    RESAMPLE_FILTER = ops.RESAMPLE_FILTER
    RESAMPLE_REDUCING_GAP = ops.RESAMPLE_REDUCING_GAP

    # ---- image conversion -------------------------------------------------

    @staticmethod
    def _to_image(image) -> Image.Image:
        """## Returns a PIL image from a path, PIL image, or image-like array"""
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            return Image.open(image)
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        mode = "RGBA" if arr.ndim == 3 and arr.shape[-1] == 4 else "RGB"
        return Image.fromarray(arr, mode)

    @staticmethod
    def _has_alpha(img: Image.Image) -> bool:
        """## Returns whether the source image has a meaningful alpha channel"""
        return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)

    @classmethod
    def _canvas_to_image(cls, canvas, color_space: str, has_alpha: bool) -> Image.Image:
        """## Converts the internal int canvas back to a displayable PIL image"""
        arr = np.clip(canvas, 0, 255).astype(np.uint8)
        if has_alpha:
            color = Image.fromarray(arr[:, :, :3], color_space).convert("RGB").convert("RGBA")
            color.putalpha(Image.fromarray(arr[:, :, 3], "L"))
            return color
        return Image.fromarray(arr, color_space).convert("RGB")

    # ---- entropy + bitstream framing -------------------------------------

    @classmethod
    def _entropy_pack(cls, body: bytes, use_lzma: bool = True) -> tuple[int, bytes]:
        """## Returns the smaller of raw body or LZMA-compressed body"""
        if not use_lzma:
            return cls.ENTROPY_STORE, body
        x = lzma.compress(body, format=lzma.FORMAT_RAW, filters=cls._LZMA_FILTERS)
        if len(x) < len(body):
            return cls.ENTROPY_LZMA, x
        return cls.ENTROPY_STORE, body

    @classmethod
    def _entropy_unpack(cls, method: int, body: bytes) -> bytes:
        """## Reverses the stream body entropy wrapper"""
        if method == cls.ENTROPY_STORE:
            return body
        if method == cls.ENTROPY_LZMA:
            return lzma.decompress(body, format=lzma.FORMAT_RAW, filters=cls._LZMA_FILTERS)
        raise ValueError(f"unknown entropy method {method}")

    @classmethod
    def _open_body(cls, data: bytes) -> tuple[int, bytes]:
        """## Validates the PBC3 header and returns the unpacked bitstream body"""
        if data[:4] != cls.MAGIC:
            raise ValueError("not a PBC3 file")
        version = data[4]
        if version != cls.VERSION:
            raise ValueError(f"unsupported PBC3 version {version}")
        return version, cls._entropy_unpack(data[5], data[6:])

    @classmethod
    def _write_grid(cls, bw: BitWriter, flat, bitcount: int) -> None:
        """## Writes a flat grid of palette indices"""
        for value in flat:
            bw.write(int(value), bitcount)

    @classmethod
    def _read_grid(cls, br: BitReader, n: int, bitcount: int) -> np.ndarray:
        """## Reads a flat grid of palette indices"""
        flat = np.zeros(n, dtype=np.uint16)
        for k in range(n):
            flat[k] = br.read(bitcount)
        return flat

    @classmethod
    def _write_header(
        cls,
        bw: BitWriter,
        w: int,
        h: int,
        original_w: int,
        original_h: int,
        downsampled: bool,
        color_id: int,
        channels: int,
        channel_bits: int,
        positive_bias: bool,
        has_alpha: bool,
        patch_count: int,
        base_values,
        warmup=None,
    ) -> None:
        """## Writes the image-level stream header"""
        bw.write(int(downsampled), 1)
        if downsampled:
            bw.write(original_w, 16)
            bw.write(original_h, 16)
        bw.write(w, 16)
        bw.write(h, 16)
        bw.write(color_id, 2)
        bw.write(channels, 8)
        bw.write(channel_bits, 4)
        bw.write(int(positive_bias), 1)
        bw.write(int(has_alpha), 1)
        bw.write(patch_count, 32)
        for base in base_values:
            bw.write(base, 8)
        bw.write(int(warmup is not None), 1)
        if warmup is not None:
            wm_w, wm_h, wm_split = warmup
            bw.write(wm_w, 16)
            bw.write(wm_h, 16)
            bw.write(wm_split, 32)

    @classmethod
    def _read_header(cls, br: BitReader):
        """## Reads the image-level stream header"""
        downsampled = bool(br.read(1))
        original_w = br.read(16) if downsampled else None
        original_h = br.read(16) if downsampled else None
        w = br.read(16)
        h = br.read(16)
        color_id = br.read(2)
        channels = br.read(8)
        channel_bits = br.read(4)
        positive_bias = bool(br.read(1))
        has_alpha = bool(br.read(1))
        patch_count = br.read(32)
        base_values = [br.read(8) for _ in range(channels)]
        warmup_on = bool(br.read(1))
        warm_w = warm_h = warmup_split = None
        if warmup_on:
            warm_w = br.read(16)
            warm_h = br.read(16)
            warmup_split = br.read(32)
        return (
            downsampled, original_w, original_h, w, h, cls.COLOR_SPACE_NAMES[color_id], channels,
            channel_bits, positive_bias, has_alpha, patch_count, base_values, warmup_on, warm_w,
            warm_h, warmup_split,
        )

    @classmethod
    def _write_patch(cls, bw: BitWriter, patch, channel_bits: int) -> None:
        """## Writes one generated-palette patch"""
        bw.write(patch["channel"], channel_bits)
        bw.write(patch["x"], 16)
        bw.write(patch["y"], 16)
        bw.write(patch["w"], 16)
        bw.write(patch["h"], 16)
        bw.write(cls.PALETTE_GENERATED, 1)
        mask = patch["mask"]
        bw.write(len(mask), 10)
        for bit in mask:
            bw.write(bit, 1)
        bw.write(patch["neg"], 8)
        bw.write(patch["pos"], 8)
        bw.write(patch["max_bitcount"], 4)
        bw.write(patch["cell_size"], 16)
        cls._write_grid(bw, patch["indices"].ravel().astype(np.int64), patch["bitcount"])

    @classmethod
    def _read_patch(cls, br: BitReader, channel_bits: int, positive_bias: bool = True):
        """## Reads one generated-palette patch and returns its decoded values"""
        channel = br.read(channel_bits)
        x = br.read(16)
        y = br.read(16)
        w = br.read(16)
        h = br.read(16)
        pm = br.read(1)
        if pm != cls.PALETTE_GENERATED:
            raise ValueError("explicit palette patches were removed in PBC3 3.0 release cleanup")
        mask_size = br.read(10)
        mask = [br.read(1) for _ in range(mask_size)]
        negative_max = br.read(8)
        positive_max = br.read(8)
        max_bitcount = br.read(4)
        bitcount = ops.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, positive_bias)
        pal = ops.palette_generator(mask, max_bitcount, negative_max, positive_max, positive_bias)
        cell_size = br.read(16)
        gw = ops.ceil_div(w, cell_size)
        gh = ops.ceil_div(h, cell_size)
        indices = cls._read_grid(br, gh * gw, bitcount).reshape(gh, gw)
        return channel, x, y, w, h, cell_size, pal[indices], bitcount

    # ---- image preparation -----------------------------------------------

    @classmethod
    def _auto_downsample_rate(cls, image_size, downsample_rate: float, max_pixels: int) -> float:
        """## Returns the requested downsample rate, or an automatic rate from max pixels"""
        if downsample_rate != -1:
            return float(downsample_rate)
        w, h = image_size
        pixels = w * h
        max_pixels = max(1, int(max_pixels))
        if pixels <= max_pixels:
            return 1.0
        return math.sqrt(pixels / max_pixels)

    @classmethod
    def _downsample_image(cls, img: Image.Image, rate: float) -> Image.Image:
        """## Downsamples an image by rate, or copies it when rate is 1"""
        if rate <= 1:
            return img.copy()
        w = max(1, int(round(img.size[0] / rate)))
        h = max(1, int(round(img.size[1] / rate)))
        return img.resize((w, h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)

    @classmethod
    def _resize_canvas(cls, canvas, new_w: int, new_h: int) -> np.ndarray:
        """## Resizes an internal int canvas without clipping it to display range"""
        h, w, ch = canvas.shape
        if (w, h) == (new_w, new_h):
            return canvas
        out = np.empty((new_h, new_w, ch), dtype=np.int32)
        for c in range(ch):
            layer = Image.fromarray(canvas[:, :, c].astype(np.float32), mode="F")
            layer = layer.resize((new_w, new_h), cls.RESAMPLE_FILTER)
            out[:, :, c] = np.rint(np.asarray(layer, dtype=np.float64)).astype(np.int32)
        return out

    @classmethod
    def _warmup_plan(cls, config: PBC3Config, original_size, init_rate: float):
        """## Returns the warmup resize plan, or None when warmup is disabled"""
        ratio = config.warmup_ratio
        if ratio is None or ratio <= 0:
            return None
        warm_max = int(config.warm_downsample_max_pixels)
        warm_rate = 1.0 if warm_max <= 0 else cls._auto_downsample_rate(original_size, -1, warm_max)
        if warm_rate >= init_rate:
            print(f"[warmup] warm target rate {warm_rate:.3f} is not higher-res than initial rate {init_rate:.3f}; ignoring warmup.", flush=True)
            return None
        k = int(round(float(ratio) * int(config.patch_count)))
        if k <= 0 or k >= int(config.patch_count):
            return None
        return warm_rate, k

    @classmethod
    def prepare(cls, image, config: PBC3Config = None, **kwargs) -> dict:
        """## Prepares the source image and reusable encoder arrays"""
        if config is None:
            config = PBC3Config(**kwargs)
        elif kwargs:
            config = PBC3Config(**{**config.__dict__, **kwargs})

        src = cls._to_image(image)
        has_alpha = cls._has_alpha(src)
        if has_alpha:
            rgba = src.convert("RGBA")
            color_img = rgba.convert("RGB").convert(config.color_space)
            alpha_img = rgba.getchannel("A")
            orig_compare = rgba
        else:
            color_img = src.convert(config.color_space)
            alpha_img = None
            orig_compare = src.convert("RGB")

        original_w, original_h = color_img.size
        rate = cls._auto_downsample_rate(color_img.size, config.downsample_rate, config.auto_downsample_max_pixels)
        color_ds = cls._downsample_image(color_img, rate)
        downsampled = color_ds.size != color_img.size
        arr = np.asarray(color_ds, dtype=np.uint8)
        if has_alpha:
            alpha_ds = alpha_img.resize(color_ds.size, cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP) if downsampled else alpha_img
            arr = np.dstack([arr, np.asarray(alpha_ds, dtype=np.uint8)])

        warm_plan = cls._warmup_plan(config, color_img.size, rate)
        warm_w = warm_h = warmup_patches = warm_target = None
        if warm_plan is not None:
            warm_rate, warmup_patches = warm_plan
            warm_color_ds = cls._downsample_image(color_img, warm_rate)
            warm_w, warm_h = warm_color_ds.size
            warm_arr = np.asarray(warm_color_ds, dtype=np.uint8)
            if has_alpha:
                warm_alpha = alpha_img.resize(warm_color_ds.size, cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
                warm_arr = np.dstack([warm_arr, np.asarray(warm_alpha, dtype=np.uint8)])
            warm_target = warm_arr.astype(np.int32)

        h, w, channels = arr.shape
        return {
            "arr": arr,
            "target": arr.astype(np.int32),
            "h": h,
            "w": w,
            "channels": channels,
            "original_w": original_w,
            "original_h": original_h,
            "downsampled": downsampled,
            "has_alpha": has_alpha,
            "orig_compare": orig_compare,
            "rate": rate,
            "color_id": cls.COLOR_SPACES[config.color_space],
            "color_space": config.color_space,
            "warm_w": warm_w,
            "warm_h": warm_h,
            "warmup_patches": warmup_patches,
            "warm_target": warm_target,
        }

    # ---- encode -----------------------------------------------------------

    @staticmethod
    def _choose_channel(scores, step: int, channels: int, mode: str) -> int:
        """## Chooses the next channel by round-robin or current total error"""
        return (step - 1) % channels if str(mode).lower() == "mod" else int(max(range(channels), key=lambda c: scores[c]))

    @staticmethod
    def _channel_sum_error(target, canvas, c: int) -> float:
        """## Returns the visible absolute error for one channel"""
        return float(np.sum(np.abs(target[:, :, c] - np.clip(canvas[:, :, c], 0, 255))))

    @classmethod
    def compress(cls, image, config: PBC3Config = None, *, reuse=None, **kwargs) -> PBC3Result:
        """## Compresses an image and returns the final result"""
        result = None
        for ev in cls.compress_stream(image, config, reuse=reuse, frame_every=0, **kwargs):
            if ev["event"] == "done":
                result = ev["result"]
        return result

    @classmethod
    def compress_stream(cls, image, config: PBC3Config = None, *, reuse=None, frame_every: int = 25, **kwargs):
        """## Compresses an image and yields optional preview frames plus the final result"""
        if config is None:
            config = PBC3Config(**kwargs)
        elif kwargs:
            config = PBC3Config(**{**config.__dict__, **kwargs})

        t0 = time.perf_counter()
        debug_lines = []
        prep = reuse if reuse is not None else cls.prepare(image, config)
        arr, target = prep["arr"], prep["target"]
        h, w, channels = prep["h"], prep["w"], prep["channels"]
        original_w, original_h = prep["original_w"], prep["original_h"]
        downsampled, has_alpha = prep["downsampled"], prep["has_alpha"]
        orig_compare, color_id, rate = prep["orig_compare"], prep["color_id"], prep["rate"]
        warm_w, warm_h = prep.get("warm_w"), prep.get("warm_h")
        warmup_patches, warm_target = prep.get("warmup_patches"), prep.get("warm_target")
        warmup_on = warmup_patches is not None
        did_warmup = False
        warmup_split = None

        if w > 65535 or h > 65535 or original_w > 65535 or original_h > 65535:
            raise ValueError("this prototype stores dimensions as uint16")
        if config.mask_size < 1 or config.mask_size > 1023:
            raise ValueError("mask_size must be in 1..1023")
        if config.auto_downsample_max_pixels < 1:
            raise ValueError("auto_downsample_max_pixels must be >= 1")
        if not (1 <= config.downsample_palette_bitcount <= 9 and 1 <= config.patch_palette_bitcount <= 9):
            raise ValueError("palette bitcounts must be in 1..9")
        if str(config.channel_cycle).lower() not in {"sum", "mod"}:
            raise ValueError('channel_cycle must be "Sum" or "Mod"')

        channel_bits = max(1, math.ceil(math.log2(channels)))
        base_values = [int(round(float(np.mean(arr[:, :, c])))) for c in range(channels)]
        canvas = np.zeros((h, w, channels), dtype=np.int32)
        for c, base in enumerate(base_values):
            canvas[:, :, c] = base
        if frame_every:
            yield {"event": "frame", "step": 0, "total": int(config.patch_count), "image": cls._canvas_to_image(canvas, config.color_space, has_alpha)}

        patches = []
        init_head = DownsampleInitHead()
        for c in range(channels):
            patch, values, init_cell, init_bits = init_head.select(c, target, canvas, w, h, config, channel_bits)
            if config.debug_print:
                print(f"[auto-init] channel {c}: cell={init_cell}, bitcount={init_bits}")
            ops.apply_grid(canvas[:, :, c], 0, 0, w, h, init_cell, values)
            patches.append(patch)
            if config.debug_mode:
                debug_lines.append(ops.debug_line("INIT", stream_patch=len(patches), channel=c, x=0, y=0, w=w, h=h, cell_size=init_cell, bitcount=init_bits))
        if frame_every:
            yield {"event": "frame", "step": 0, "total": int(config.patch_count), "image": cls._canvas_to_image(canvas, config.color_space, has_alpha)}

        channel_scores = [cls._channel_sum_error(target, canvas, c) for c in range(channels)]
        quality_target = float(config.quality_target_mae)
        filler = FillerHead(config, channel_bits, (h, w, channels), patches, (original_w, original_h))
        search = SearchHead()
        rng = np.random.default_rng(config.random_seed)
        applied = 0
        for step in range(1, max(0, int(config.patch_count)) + 1):
            current_channel = cls._choose_channel(channel_scores, step, channels, config.channel_cycle)
            boxes = None if filler.learned is not None else search.propose(target, canvas, config, rng, step, current_channel)
            patch, values = filler.select(target, canvas, config, rng, channel_bits, step, current_channel, boxes, len(patches), debug_lines)
            if patch is None:
                break

            c = patch["channel"]
            ops.apply_grid(canvas[:, :, c], patch["x"], patch["y"], patch["w"], patch["h"], patch["cell_size"], values)
            patches.append(patch)
            channel_scores[c] = cls._channel_sum_error(target, canvas, c)
            applied += 1

            if warmup_on and not did_warmup and applied == warmup_patches:
                canvas = cls._resize_canvas(canvas, warm_w, warm_h)
                target = warm_target
                channel_scores = [cls._channel_sum_error(target, canvas, c) for c in range(channels)]
                warmup_split = len(patches)
                did_warmup = True
            if config.debug_mode:
                debug_lines.append(ops.debug_line("APPLIED", patch_step=step, stream_patch=len(patches), channel=c, channel_score=f"{channel_scores[c]:.4f}", x=patch["x"], y=patch["y"], w=patch["w"], h=patch["h"], cell_size=patch["cell_size"]))
            if config.debug_print:
                print("|", end="", flush=True)
            if frame_every and applied % frame_every == 0:
                yield {"event": "frame", "step": step, "total": int(config.patch_count), "image": cls._canvas_to_image(canvas, config.color_space, has_alpha)}
            if quality_target > 0 and float(np.mean(np.abs(target - np.clip(canvas, 0, 255)))) <= quality_target:
                break
        if config.debug_print:
            print()

        bw = BitWriter()
        cls._write_header(
            bw, w, h, original_w, original_h, downsampled, color_id, channels, channel_bits,
            config.positive_bias, has_alpha, len(patches), base_values,
            warmup=(warm_w, warm_h, warmup_split) if did_warmup else None,
        )
        for patch in patches:
            cls._write_patch(bw, patch, channel_bits)
        method, body = cls._entropy_pack(bw.finish(), config.use_lzma)
        data = cls.MAGIC + bytes([cls.VERSION, method]) + body

        out_img = cls._canvas_to_image(canvas, config.color_space, has_alpha)
        if out_img.size != (original_w, original_h):
            out_img = out_img.resize((original_w, original_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
        mse = ops.final_mse(orig_compare, out_img) if config.compute_final_mse else None
        total_seconds = time.perf_counter() - t0

        debug_path = None
        if config.debug_mode:
            ts = time.strftime("%Y%m%d_%H%M%S")
            debug_path = config.debug_path or f"debug_{ts}.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(ops.debug_line("CONFIG", **{k: v for k, v in config.__dict__.items() if k not in {"debug_path"}}) + "\n")
                f.write(ops.debug_line("IMAGE", original_w=original_w, original_h=original_h, working_w=w, working_h=h, original_pixels=original_w * original_h, working_pixels=w * h, downsample_rate=f"{rate:.6f}", downsampled=int(downsampled), has_alpha=int(has_alpha)) + "\n")
                for line in debug_lines:
                    f.write(line + "\n")

        yield {
            "event": "done",
            "result": PBC3Result(
                out_img, data, config, mse, total_seconds, len(data) * 8,
                original_w, original_h, canvas.shape[1], canvas.shape[0], debug_path, channels=channels,
            ),
        }

    # ---- decode (deterministic, ML-free) ---------------------------------

    @classmethod
    def _decode_to_canvas(cls, data, max_patches: int = None):
        """## Decodes a PBC3 stream to the internal canvas without making a PIL image"""
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        version, body = cls._open_body(data)
        br = BitReader(body)
        header = cls._read_header(br)
        downsampled, original_w, original_h, w, h, color_space, channels, channel_bits, positive_bias, has_alpha, patch_count, base_values, warmup_on, warm_w, warm_h, warmup_split = header
        canvas = np.zeros((h, w, channels), dtype=np.int32)
        for c, base in enumerate(base_values):
            canvas[:, :, c] = base
        patches_to_read = patch_count if max_patches is None else min(int(max_patches), patch_count)
        for idx in range(patches_to_read):
            if warmup_on and idx == warmup_split:
                canvas = cls._resize_canvas(canvas, warm_w, warm_h)
            channel, x, y, pw, ph, cell_size, values, _ = cls._read_patch(br, channel_bits, positive_bias)
            ops.apply_grid(canvas[:, :, channel], x, y, pw, ph, cell_size, values)
        return canvas, color_space, downsampled, original_w, original_h, canvas.shape[1], canvas.shape[0], has_alpha, channels, patch_count

    @classmethod
    def decompress(cls, data, max_patches: int = None) -> PBC3Result:
        """## Decompresses a PBC3 stream or file path"""
        t0 = time.perf_counter()
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        canvas, color_space, downsampled, original_w, original_h, w, h, has_alpha, channels, patch_count = cls._decode_to_canvas(data, max_patches=max_patches)
        img = cls._canvas_to_image(canvas, color_space, has_alpha)
        if downsampled and img.size != (original_w, original_h):
            img = img.resize((original_w, original_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
        cfg = PBC3Config(color_space=color_space)
        return PBC3Result(img, data, cfg, None, time.perf_counter() - t0, len(data) * 8, original_w or w, original_h or h, w, h, channels=channels)

    # ---- file helpers -----------------------------------------------------

    @classmethod
    def encode_file(cls, input_path: str, output_path: str, config: PBC3Config = None, **kwargs) -> PBC3Result:
        """## Compresses a file and writes the .pbc3 output"""
        result = cls.compress(Image.open(input_path), config=config, **kwargs)
        with open(output_path, "wb") as f:
            f.write(result.data)
        return result

    @classmethod
    def decode_file(cls, input_path: str, output_path: str = None) -> Image.Image:
        """## Decodes a .pbc3 file and optionally writes the image output"""
        image = cls.decompress(input_path).image
        if output_path is not None:
            image.save(output_path)
        return image


def preload_numba() -> None:
    """## Runs a tiny encode so numba compiles the hot kernels once"""
    img = Image.fromarray(np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8))
    PBC3.compress(img, PBC3Config(patch_count=10, auto_downsample_init=True, learned_filler_enabled=False))
    print("[preload] numba kernels compiled")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python PBC3.py input_image output.pbc3")
    else:
        preload_numba()
        res = PBC3.encode_file(sys.argv[1], sys.argv[2])
        print(f"MSE: {res.mse:.2f} | Size: {len(res.data) / 1024:.2f} KB | Rate: {res.compression_rate:.2f}x | Time: {res.encode_seconds:.3f}s")