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
from PIL import Image, ImageOps

import pbc3_stream as stream
import pbc3_ops as ops
from pbc3_heads import DownsampleInitHead, FillerHead, SearchHead
from pbc3_types import BitReader, BitWriter, PBC3Config, PBC3Result
from pbc3_trace import TimingTrace, timed


def _config(config, kwargs):
    if config is None:
        return PBC3Config(**kwargs)
    return PBC3Config(**{**config.__dict__, **kwargs}) if kwargs else config


def _canvas_from_bases(shape, bases):
    canvas = np.zeros(shape, dtype=np.int16)
    for channel, base in enumerate(bases):
        canvas[:, :, channel] = base
    return canvas


def _validate(prep, config):
    h, w = prep["h"], prep["w"]
    ow, oh = prep["original_w"], prep["original_h"]
    if max(w, h, ow, oh) > 65535:
        raise ValueError("this prototype stores dimensions as uint16")
    if not 1 <= config.mask_size <= 1023:
        raise ValueError("mask_size must be in 1..1023")
    if config.auto_downsample_max_pixels < 1:
        raise ValueError("auto_downsample_max_pixels must be >= 1")
    if not (
        1 <= config.downsample_palette_bitcount <= 9
        and 1 <= config.patch_palette_bitcount <= 9
    ):
        raise ValueError("palette bitcounts must be in 1..9")
    if str(config.channel_cycle).lower() not in {"sum", "mod"}:
        raise ValueError('channel_cycle must be "Sum" or "Mod"')


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

    @staticmethod
    def _to_image(image) -> Image.Image:
        """## Returns a PIL image from a path, PIL image, or image-like array"""
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image)
        if isinstance(image, str):
            return ImageOps.exif_transpose(Image.open(image))
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
    def _open_body(cls, data: bytes, trace=None) -> tuple[int, bytes]:
        """## Validates the PBC3 header and returns the unpacked bitstream body"""
        with timed(trace, "decode.validate_magic_version"):
            if data[:4] != cls.MAGIC:
                raise ValueError("not a PBC3 file")
            version = data[4]
            if version != cls.VERSION:
                raise ValueError(f"unsupported PBC3 version {version}")
        with timed(trace, "decode.entropy_unpack"):
            body = cls._entropy_unpack(data[5], data[6:])
        return version, body

    _write_grid = staticmethod(stream.write_grid)
    _read_grid = staticmethod(stream.read_grid)
    _write_header = staticmethod(stream.write_header)
    _write_patch = staticmethod(stream.write_patch)

    @classmethod
    def _read_header(cls, br):
        return stream.read_header(br, cls.COLOR_SPACE_NAMES)

    @staticmethod
    def _read_patch(br, channel_bits: int, positive_bias: bool = True):
        return stream.read_patch(br, channel_bits, positive_bias)

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
        out = np.empty((new_h, new_w, ch), dtype=np.int16)
        for c in range(ch):
            layer_arr = np.ascontiguousarray(canvas[:, :, c], dtype=np.float32)
            layer = Image.frombuffer("F", (w, h), layer_arr, "raw", "F", 0, 1)
            layer = layer.resize((new_w, new_h), cls.RESAMPLE_FILTER)
            out[:, :, c] = np.rint(np.asarray(layer, dtype=np.float32)).astype(np.int16)
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
    def prepare(cls, image, config: PBC3Config = None, *, trace=None, **kwargs) -> dict:
        """## Prepares the source image and reusable encoder arrays"""
        config = _config(config, kwargs)

        with timed(trace, "prepare.input_normalize"):
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

        with timed(trace, "prepare.downsample_color"):
            original_w, original_h = color_img.size
            rate = cls._auto_downsample_rate(
                color_img.size, config.downsample_rate, config.auto_downsample_max_pixels
            )
            color_ds = cls._downsample_image(color_img, rate)
            downsampled = color_ds.size != color_img.size
            arr = np.asarray(color_ds, dtype=np.uint8)
            if has_alpha:
                alpha_ds = (
                    alpha_img.resize(color_ds.size, cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
                    if downsampled else alpha_img
                )
                arr = np.dstack([arr, np.asarray(alpha_ds, dtype=np.uint8)])

        with timed(trace, "prepare.build_warmup_target"):
            warm_plan = cls._warmup_plan(config, color_img.size, rate)
            warm_w = warm_h = warmup_patches = warm_target = None
            if warm_plan is not None:
                warm_rate, warmup_patches = warm_plan
                warm_color_ds = cls._downsample_image(color_img, warm_rate)
                warm_w, warm_h = warm_color_ds.size
                warm_arr = np.asarray(warm_color_ds, dtype=np.uint8)
                if has_alpha:
                    warm_alpha = alpha_img.resize(
                        warm_color_ds.size, cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP
                    )
                    warm_arr = np.dstack([warm_arr, np.asarray(warm_alpha, dtype=np.uint8)])
                warm_target = warm_arr.astype(np.int32)

        with timed(trace, "prepare.materialize_arrays"):
            h, w, channels = arr.shape
            target = arr.astype(np.int32)
        return {
            "arr": arr,
            "target": target,
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

    @staticmethod
    def _choose_channel(scores, step: int, channels: int, mode: str) -> int:
        """## Chooses the next channel by round-robin or current total error"""
        return (
            (step - 1) % channels
            if str(mode).lower() == "mod"
            else int(max(range(channels), key=lambda c: scores[c]))
        )

    @staticmethod
    def _channel_sum_error(target, canvas, c: int) -> float:
        """## Returns the visible absolute error for one channel"""
        return float(np.sum(np.abs(target[:, :, c] - np.clip(canvas[:, :, c], 0, 255))))

    @classmethod
    def compress(cls, image, config: PBC3Config = None, *, reuse=None, trace=None, **kwargs) -> PBC3Result:
        """## Compresses an image and returns the final result"""
        result = None
        for ev in cls.compress_stream(image, config, reuse=reuse, trace=trace, frame_every=0, **kwargs):
            if ev["event"] == "done":
                result = ev["result"]
        return result

    @classmethod
    def compress_stream(cls, image, config: PBC3Config = None, *, reuse=None, trace=None, frame_every: int = 25, **kwargs):
        """## Compresses an image and yields optional preview frames plus the final result"""
        config = _config(config, kwargs)
        owns_trace = trace is True
        trace = TimingTrace("encode", {"patch_count_requested": int(config.patch_count)}) if owns_trace else trace

        t0 = time.perf_counter()
        debug_lines = []
        if reuse is not None:
            prep = reuse
        else:
            with timed(trace, "encode.prepare"):
                prep = cls.prepare(image, config, trace=trace)
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

        with timed(trace, "encode.validate"):
            _validate(prep, config)

        with timed(trace, "encode.initialize_canvas"):
            channel_bits = max(1, math.ceil(math.log2(channels)))
            base_values = [int(round(float(np.mean(arr[:, :, c])))) for c in range(channels)]
            canvas = _canvas_from_bases((h, w, channels), base_values)
        if frame_every:
            with timed(trace, "encode.preview_frame", step=0):
                yield {
                    "event": "frame", "step": 0, "total": int(config.patch_count),
                    "image": cls._canvas_to_image(canvas, config.color_space, has_alpha),
                }

        patches = []
        with timed(trace, "encode.initial_patch_selection"):
            init_head = DownsampleInitHead()
            for c in range(channels):
                with timed(trace, "encode.initial_patch.channel", channel=c):
                    with timed(trace, "head.downsample_init", channel=c):
                        patch, values, init_delta, init_cell, init_bits = init_head.select(
                            c, target, canvas, w, h, config, channel_bits, trace=trace
                        )
                    if config.debug_print:
                        print(f"[auto-init] channel {c}: cell={init_cell}, bitcount={init_bits}")
                    with timed(trace, "encode.initial_patch.apply", channel=c):
                        if config.reuse_selected_delta:
                            ops.apply_delta(canvas[:, :, c], 0, 0, w, h, init_delta)
                            if trace is not None:
                                trace.count("selected_deltas_reused")
                        else:
                            ops.apply_grid(canvas[:, :, c], 0, 0, w, h, init_cell, values, trace=trace)
                    patches.append(patch)
                    if config.debug_mode:
                        debug_lines.append(ops.debug_line(
                            "INIT", stream_patch=len(patches), channel=c, x=0, y=0, w=w, h=h,
                            cell_size=init_cell, bitcount=init_bits,
                        ))
        if frame_every:
            with timed(trace, "encode.preview_frame", step=0, kind="initialized"):
                yield {
                    "event": "frame", "step": 0, "total": int(config.patch_count),
                    "image": cls._canvas_to_image(canvas, config.color_space, has_alpha),
                }

        with timed(trace, "encode.initial_error_scores"):
            search_state = ops.SearchState()
            channel_scores = [search_state.score(target, canvas, c) for c in range(channels)]
        quality_target = float(config.quality_target_mae)
        with timed(trace, "encode.filler_initialize"):
            filler = FillerHead(config, channel_bits, (h, w, channels), patches, (original_w, original_h), trace=trace)
        with timed(trace, "encode.search_initialize"):
            search = SearchHead()
            rng = ops.PBC3Rng(config.random_seed)
        applied = 0
        for step in range(1, max(0, int(config.patch_count)) + 1):
            with timed(trace, "encode.patch.total", step=step):
                with timed(trace, "encode.patch.choose_channel", step=step):
                    current_channel = cls._choose_channel(channel_scores, step, channels, config.channel_cycle)
                with timed(trace, "encode.patch.search", step=step, channel=current_channel):
                    with timed(trace, "head.search", step=step, channel=current_channel):
                        boxes = (
                            None if filler.learned is not None
                            else search.propose(target, canvas, config, rng, step, current_channel, trace=trace,
                                abs_error=search_state.error(target, canvas, current_channel))
                        )
                with timed(trace, "encode.patch.fill", step=step, channel=current_channel):
                    with timed(trace, "head.filler", step=step, channel=current_channel):
                        patch, values, delta = filler.select(
                            target, canvas, config, rng, channel_bits, step, current_channel,
                            boxes, len(patches), debug_lines, trace=trace,
                            abs_error=search_state.error(target, canvas, current_channel),
                        )
                if patch is None:
                    break

                c = patch["channel"]
                with timed(trace, "encode.patch.apply", step=step, channel=c):
                    if config.reuse_selected_delta:
                        ops.apply_delta(canvas[:, :, c], patch["x"], patch["y"], patch["w"], patch["h"], delta)
                        if trace is not None:
                            trace.count("selected_deltas_reused")
                    else:
                        ops.apply_grid(
                            canvas[:, :, c], patch["x"], patch["y"], patch["w"], patch["h"],
                            patch["cell_size"], values, trace=trace,
                        )
                patches.append(patch)
                with timed(trace, "encode.patch.score", step=step, channel=c):
                    search_state.update(target, canvas, c, patch["x"], patch["y"], patch["w"], patch["h"])
                    channel_scores[c] = search_state.score(target, canvas, c)
                applied += 1
                if trace is not None:
                    trace.count("patches_accepted")

                if warmup_on and not did_warmup and applied == warmup_patches:
                    with timed(trace, "encode.warmup_resize", step=step, width=warm_w, height=warm_h):
                        canvas = cls._resize_canvas(canvas, warm_w, warm_h)
                        target = warm_target
                        channel_scores = [search_state.score(target, canvas, c) for c in range(channels)]
                        warmup_split = len(patches)
                        did_warmup = True
                if config.debug_mode:
                    debug_lines.append(ops.debug_line(
                        "APPLIED", patch_step=step, stream_patch=len(patches), channel=c,
                        channel_score=f"{channel_scores[c]:.4f}", x=patch["x"], y=patch["y"],
                        w=patch["w"], h=patch["h"], cell_size=patch["cell_size"],
                    ))
                if config.debug_print:
                    print("|", end="", flush=True)
                if frame_every and applied % frame_every == 0:
                    with timed(trace, "encode.preview_frame", step=step):
                        yield {
                            "event": "frame", "step": step, "total": int(config.patch_count),
                            "image": cls._canvas_to_image(canvas, config.color_space, has_alpha),
                        }
                with timed(trace, "encode.patch.quality_check", step=step):
                    if quality_target > 0 and float(
                        np.mean(np.abs(target - np.clip(canvas, 0, 255)))
                    ) <= quality_target:
                        break
        if config.debug_print:
            print()

        with timed(trace, "encode.serialize.header"):
            bw = BitWriter()
            cls._write_header(
                bw, w, h, original_w, original_h, downsampled, color_id, channels, channel_bits,
                config.positive_bias, has_alpha, len(patches), base_values,
                warmup=(warm_w, warm_h, warmup_split) if did_warmup else None,
            )
        with timed(trace, "encode.serialize.patches", patch_count=len(patches)):
            for patch in patches:
                cls._write_patch(bw, patch, channel_bits)
        with timed(trace, "encode.entropy_pack"):
            method, body = cls._entropy_pack(bw.finish(), config.use_lzma)
            data = cls.MAGIC + bytes([cls.VERSION, method]) + body

        with timed(trace, "encode.reconstruct.canvas_to_image"):
            out_img = cls._canvas_to_image(canvas, config.color_space, has_alpha)
        if out_img.size != (original_w, original_h):
            with timed(trace, "encode.reconstruct.final_resize", width=original_w, height=original_h):
                out_img = out_img.resize((original_w, original_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
        with timed(trace, "encode.metric.final_mse"):
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
                timings=trace.report(
                    {"patches_applied": applied, "patches_serialized": len(patches)},
                    finish=owns_trace,
                ) if trace else None,
            ),
        }

    @classmethod
    def _decode_to_canvas(cls, data, max_patches: int = None, trace=None):
        """## Decodes a PBC3 stream to the internal canvas without making a PIL image"""
        if isinstance(data, str):
            with timed(trace, "decode.receive.file_read"):
                with open(data, "rb") as f:
                    data = f.read()
        with timed(trace, "decode.open_body"):
            version, body = cls._open_body(data, trace=trace)
        with timed(trace, "decode.parse_header"):
            br = BitReader(body)
            header = cls._read_header(br)
        (
            downsampled, original_w, original_h, w, h, color_space, channels,
            channel_bits, positive_bias, has_alpha, patch_count, base_values,
            warmup_on, warm_w, warm_h, warmup_split,
        ) = header
        with timed(trace, "decode.initialize_canvas", width=w, height=h, channels=channels):
            canvas = _canvas_from_bases((h, w, channels), base_values)
        patches_to_read = patch_count if max_patches is None else min(int(max_patches), patch_count)
        read_patch = cls._read_patch
        resize_canvas = cls._resize_canvas
        signed_resample = ops.signed_resample
        with timed(trace, "decode.patch_loop", patch_count=patches_to_read):
            for idx in range(patches_to_read):
                if warmup_on and idx == warmup_split:
                    with timed(trace, "decode.warmup_resize", patch=idx, width=warm_w, height=warm_h):
                        canvas = resize_canvas(canvas, warm_w, warm_h)
                with timed(trace, "decode.patch.read", patch=idx):
                    channel, x, y, pw, ph, cell_size, values, _ = read_patch(
                        br, channel_bits, positive_bias
                    )
                with timed(trace, "decode.patch.resample_apply", patch=idx, channel=channel):
                    canvas[y:y + ph, x:x + pw, channel] += signed_resample(
                        values, ph, pw, trace=trace, purpose="decoder_reconstruction"
                    )
        return (
            canvas, color_space, downsampled, original_w, original_h,
            canvas.shape[1], canvas.shape[0], has_alpha, channels, patch_count,
        )

    @classmethod
    def decompress(cls, data, max_patches: int = None, *, trace=None) -> PBC3Result:
        """## Decompresses a PBC3 stream or file path"""
        owns_trace = trace is True
        trace = TimingTrace("decode") if owns_trace else trace
        t0 = time.perf_counter()
        if isinstance(data, str):
            with timed(trace, "decode.receive.file_read"):
                with open(data, "rb") as f:
                    data = f.read()
        with timed(trace, "decode.to_canvas"):
            with timed(trace, "head.decoder"):
                (
                    canvas, color_space, downsampled, original_w, original_h, w, h,
                    has_alpha, channels, patch_count,
                ) = cls._decode_to_canvas(data, max_patches=max_patches, trace=trace)
        with timed(trace, "decode.reconstruct.canvas_to_image"):
            img = cls._canvas_to_image(canvas, color_space, has_alpha)
        if downsampled and img.size != (original_w, original_h):
            with timed(trace, "decode.reconstruct.final_resize", width=original_w, height=original_h):
                img = img.resize((original_w, original_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
        cfg = PBC3Config(color_space=color_space)
        return PBC3Result(
            img, data, cfg, None, time.perf_counter() - t0, len(data) * 8,
            original_w or w, original_h or h, w, h, channels=channels,
            timings=trace.report(
                {"patches_decoded": patch_count if max_patches is None else min(int(max_patches), patch_count)},
                finish=owns_trace,
            ) if trace else None,
        )

    @classmethod
    def encode_file(
        cls, input_path: str, output_path: str, config: PBC3Config = None, **kwargs
    ) -> PBC3Result:
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


def preload_numba(model_path: str = "patch_policy.npz") -> None:
    """## Warms the production learned RGB/RGBA encode paths."""
    h, w = 512, 768
    base = np.arange(h * w * 4, dtype=np.uint32).reshape(h, w, 4)
    config = PBC3Config.quality(
        patch_count=50,
        learned_filler_enabled=True,
        learned_filler_model_path=model_path,
        auto_downsample_max_pixels=250_000,
        use_lzma=True,
        compute_final_mse=True,
    )
    for channels in (3, 4):
        arr = ((base[:, :, :channels] * 37 + channels * 19) % 256).astype(np.uint8)
        PBC3.compress(Image.fromarray(arr), config=config)
    print("[preload] production PBC3 paths warmed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python PBC3.py input_image output.pbc3")
    else:
        preload_numba()
        res = PBC3.encode_file(sys.argv[1], sys.argv[2])
        print(f"MSE: {res.mse:.2f} | Size: {len(res.data) / 1024:.2f} KB | Rate: {res.compression_rate:.2f}x | Time: {res.encode_seconds:.3f}s")
