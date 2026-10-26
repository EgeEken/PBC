import math
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

import pbc3_ops as ops
from PBC3 import PBC3, _canvas_from_bases, _config, _validate
from pbc3_heads import DownsampleInitHead, FillerHead, SearchHead
from pbc3_types import BitReader, BitWriter, PBC3Config, PBC3Result


@dataclass(frozen=True)
class StagePlan:
    name: str
    scales: tuple[float, ...]
    budget_ratios: tuple[float, ...]

    def __post_init__(self):
        if len(self.scales) != len(self.budget_ratios) or not self.scales:
            raise ValueError("stage scales and budget ratios must have the same non-zero length")
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("stage scales must be positive")
        if any(a > b for a, b in zip(self.scales, self.scales[1:])):
            raise ValueError("stage scales must be monotonic")
        if not math.isclose(self.scales[-1], 1.0):
            raise ValueError("the final stage scale must be 1.0")
        if any(ratio < 0 for ratio in self.budget_ratios) or sum(self.budget_ratios) <= 0:
            raise ValueError("stage budget ratios must be non-negative with a positive sum")


SINGLE_STAGE = StagePlan("single_v1", (1.0,), (1.0,))
T2_COARSE_INIT_ONLY = StagePlan("T2_coarse_init_only", (0.5, 1.0), (0.0, 1.0))
T2_05_95 = StagePlan("T2a_05_95", (0.5, 1.0), (0.05, 0.95))
T2_10_90 = StagePlan("T2a_10_90", (0.5, 1.0), (0.1, 0.9))
T2_20_80 = StagePlan("T2a_A1", (0.5, 1.0), (0.2, 0.8))
T2_40_60 = StagePlan("T2a_A2", (0.5, 1.0), (0.4, 0.6))
T2_075_10_90 = StagePlan("T2_075_10_90", (0.75, 1.0), (0.1, 0.9))
T2_075_20_80 = StagePlan("T2_075_20_80", (0.75, 1.0), (0.2, 0.8))
T3_10_20_70 = StagePlan("T3a_A3", (0.25, 0.5, 1.0), (0.1, 0.2, 0.7))


def _allocate(total, weights):
    weights = np.asarray(weights, dtype=np.float64)
    exact = weights / weights.sum() * max(0, int(total))
    allocated = np.floor(exact).astype(np.int64)
    remaining = int(total) - int(allocated.sum())
    order = np.argsort(-(exact - allocated), kind="stable")
    for index in order[:remaining]:
        allocated[index] += 1
    return [int(value) for value in allocated]


def _stage_specs(final_size, plan, patch_count):
    final_w, final_h = final_size
    unique = []
    for scale, ratio in zip(plan.scales, plan.budget_ratios):
        dims = (
            max(1, min(final_w, int(round(final_w * scale)))),
            max(1, min(final_h, int(round(final_h * scale)))),
        )
        if unique and unique[-1][0] == dims:
            unique[-1] = (dims, unique[-1][1] + ratio)
        else:
            unique.append((dims, ratio))
    budgets = _allocate(patch_count, [ratio for _, ratio in unique])
    return [(dims[0], dims[1], budget) for (dims, _), budget in zip(unique, budgets)]


def _write_header(
    bw, initial_w, initial_h, original_w, original_h, downsampled, color_id, channels,
    channel_bits, positive_bias, has_alpha, patch_count, base_values, stages,
):
    bw.write(int(downsampled), 1)
    if downsampled:
        bw.write(original_w, 16)
        bw.write(original_h, 16)
    bw.write(initial_w, 16)
    bw.write(initial_h, 16)
    bw.write(color_id, 2)
    bw.write(channels, 8)
    bw.write(channel_bits, 4)
    bw.write(int(positive_bias), 1)
    bw.write(int(has_alpha), 1)
    bw.write(patch_count, 32)
    for base in base_values:
        bw.write(base, 8)
    bw.write(len(stages), 8)
    for width, height, split in stages[1:]:
        bw.write(width, 16)
        bw.write(height, 16)
        bw.write(split, 32)


def _read_header(br, color_space_names):
    downsampled = bool(br.read(1))
    original_w = br.read(16) if downsampled else None
    original_h = br.read(16) if downsampled else None
    initial_w = br.read(16)
    initial_h = br.read(16)
    color_id = br.read(2)
    channels = br.read(8)
    channel_bits = br.read(4)
    positive_bias = bool(br.read(1))
    has_alpha = bool(br.read(1))
    patch_count = br.read(32)
    base_values = [br.read(8) for _ in range(channels)]
    stage_count = br.read(8)
    if stage_count < 1:
        raise ValueError("multistage stream must contain at least one stage")
    stages = [(initial_w, initial_h, 0)]
    for _ in range(stage_count - 1):
        stages.append((br.read(16), br.read(16), br.read(32)))
    return (
        downsampled, original_w, original_h, color_space_names[color_id], channels,
        channel_bits, positive_bias, has_alpha, patch_count, base_values, stages,
    )


class MultistagePBC31(PBC3):
    VERSION = 1

    @classmethod
    def _open_body(cls, data, trace=None):
        if data[:4] != cls.MAGIC:
            raise ValueError("not a PBC3 file")
        version = data[4]
        if version not in (PBC3.VERSION, cls.VERSION):
            raise ValueError(f"unsupported PBC3 version {version}")
        return version, cls._entropy_unpack(data[5], data[6:])

    @classmethod
    def _prepare(cls, image, config, plan):
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
        rate = cls._auto_downsample_rate(
            color_img.size, config.downsample_rate, config.auto_downsample_max_pixels
        )
        final_img = cls._downsample_image(color_img, rate)
        final_w, final_h = final_img.size
        specs = _stage_specs((final_w, final_h), plan, config.patch_count)
        targets = []
        for width, height, budget in specs:
            stage_color = (
                color_img.copy()
                if (width, height) == color_img.size
                else color_img.resize(
                    (width, height), cls.RESAMPLE_FILTER,
                    reducing_gap=cls.RESAMPLE_REDUCING_GAP,
                )
            )
            arr = np.asarray(stage_color, dtype=np.uint8)
            if has_alpha:
                stage_alpha = (
                    alpha_img
                    if (width, height) == alpha_img.size
                    else alpha_img.resize(
                        (width, height), cls.RESAMPLE_FILTER,
                        reducing_gap=cls.RESAMPLE_REDUCING_GAP,
                    )
                )
                arr = np.dstack([arr, np.asarray(stage_alpha, dtype=np.uint8)])
            targets.append({
                "width": width,
                "height": height,
                "budget": budget,
                "arr": arr,
                "target": arr.astype(np.int32),
            })
        first = targets[0]
        return {
            "arr": first["arr"],
            "target": first["target"],
            "h": first["height"],
            "w": first["width"],
            "channels": first["arr"].shape[2],
            "original_w": original_w,
            "original_h": original_h,
            "downsampled": (final_w, final_h) != (original_w, original_h),
            "has_alpha": has_alpha,
            "orig_compare": orig_compare,
            "color_id": cls.COLOR_SPACES[config.color_space],
            "color_space": config.color_space,
            "targets": targets,
            "final_w": final_w,
            "final_h": final_h,
        }

    @classmethod
    def compress(cls, image, config: PBC3Config = None, *, plan=SINGLE_STAGE, diagnostics=False, **kwargs):
        config = _config(config, kwargs)
        start = time.perf_counter()
        prep = cls._prepare(image, config, plan)
        _validate(prep, config)

        channels = prep["channels"]
        original_w, original_h = prep["original_w"], prep["original_h"]
        has_alpha = prep["has_alpha"]
        stages = prep["targets"]
        first = stages[0]
        channel_bits = max(1, math.ceil(math.log2(channels)))
        base_values = [
            int(round(float(np.mean(first["arr"][:, :, channel]))))
            for channel in range(channels)
        ]
        canvas = _canvas_from_bases(
            (first["height"], first["width"], channels), base_values
        )
        patches = []
        init_head = DownsampleInitHead()
        for channel in range(channels):
            patch, values, delta, _, _ = init_head.select(
                channel, first["target"], canvas, first["width"], first["height"],
                config, channel_bits,
            )
            if config.reuse_selected_delta:
                ops.apply_delta(
                    canvas[:, :, channel], 0, 0, first["width"], first["height"], delta
                )
            else:
                ops.apply_grid(
                    canvas[:, :, channel], 0, 0, first["width"], first["height"],
                    patch["cell_size"], values,
                )
            patches.append(patch)

        filler = FillerHead(
            config, channel_bits, canvas.shape, patches, (original_w, original_h)
        )
        search = SearchHead()
        rng = ops.PBC3Rng(config.random_seed)
        stage_stream = [(first["width"], first["height"], 0)]
        stage_rows = []
        slot_offset = 0
        quality_target = float(config.quality_target_mae)

        for stage_index, stage in enumerate(stages):
            resize_seconds = 0.0
            if stage_index:
                resize_start = time.perf_counter()
                canvas = cls._resize_canvas(canvas, stage["width"], stage["height"])
                resize_seconds = time.perf_counter() - resize_start
                stage_stream.append((stage["width"], stage["height"], len(patches)))
            target = stage["target"]
            channel_scores = [
                cls._channel_sum_error(target, canvas, channel)
                for channel in range(channels)
            ]
            residual_start = (
                float(np.mean(np.abs(target - np.clip(canvas, 0, 255))))
                if diagnostics else None
            )
            accepted = 0
            raw_patch_bits = 0
            stop_reason = "budget_exhausted"
            stage_start = time.perf_counter()
            for local_step in range(1, stage["budget"] + 1):
                step = slot_offset + local_step
                current_channel = cls._choose_channel(
                    channel_scores, step, channels, config.channel_cycle
                )
                boxes = (
                    None
                    if filler.learned is not None
                    else search.propose(
                        target, canvas, config, rng, step, current_channel
                    )
                )
                patch, values, delta = filler.select(
                    target, canvas, config, rng, channel_bits, step, current_channel,
                    boxes, len(patches), [],
                )
                if patch is None:
                    stop_reason = "filler_none"
                    break
                channel = patch["channel"]
                if config.reuse_selected_delta:
                    ops.apply_delta(
                        canvas[:, :, channel], patch["x"], patch["y"],
                        patch["w"], patch["h"], delta,
                    )
                else:
                    ops.apply_grid(
                        canvas[:, :, channel], patch["x"], patch["y"],
                        patch["w"], patch["h"], patch["cell_size"], values,
                    )
                patches.append(patch)
                accepted += 1
                raw_patch_bits += ops.patch_bits_for(patch, channel_bits)
                channel_scores[channel] = cls._channel_sum_error(
                    target, canvas, channel
                )
                if quality_target > 0 and float(
                    np.mean(np.abs(target - np.clip(canvas, 0, 255)))
                ) <= quality_target:
                    stop_reason = "quality_target"
                    break
            stage_seconds = time.perf_counter() - stage_start
            residual_end = (
                float(np.mean(np.abs(target - np.clip(canvas, 0, 255))))
                if diagnostics else None
            )
            stage_rows.append({
                "stage": stage_index,
                "width": stage["width"],
                "height": stage["height"],
                "budget": stage["budget"],
                "accepted": accepted,
                "stream_start": stage_stream[-1][2],
                "stream_end": len(patches),
                "stage_seconds": stage_seconds,
                "resize_seconds": resize_seconds,
                "raw_patch_bits": raw_patch_bits,
                "residual_start_mae": residual_start,
                "residual_end_mae": residual_end,
                "stop_reason": stop_reason,
            })
            slot_offset += stage["budget"]

        writer = BitWriter()
        _write_header(
            writer, first["width"], first["height"], original_w, original_h,
            prep["downsampled"], prep["color_id"], channels, channel_bits,
            config.positive_bias, has_alpha, len(patches), base_values, stage_stream,
        )
        for patch in patches:
            cls._write_patch(writer, patch, channel_bits)
        raw_body = writer.finish()
        method, body = cls._entropy_pack(raw_body, config.use_lzma)
        data = cls.MAGIC + bytes([cls.VERSION, method]) + body

        out_img = cls._canvas_to_image(canvas, config.color_space, has_alpha)
        if out_img.size != (original_w, original_h):
            out_img = out_img.resize(
                (original_w, original_h), cls.RESAMPLE_FILTER,
                reducing_gap=cls.RESAMPLE_REDUCING_GAP,
            )
        mse = (
            ops.final_mse(prep["orig_compare"], out_img)
            if config.compute_final_mse else None
        )
        return PBC3Result(
            out_img, data, config, mse, time.perf_counter() - start, len(data) * 8,
            original_w, original_h, prep["final_w"], prep["final_h"],
            channels=channels,
            timings={
                "plan": plan.name,
                "stages": stage_rows,
                "raw_bytes": len(raw_body),
                "final_bytes": len(data),
                "stage_header_bits": 8 + 64 * (len(stage_stream) - 1),
            },
        )

    @classmethod
    def _decode_to_canvas(cls, data, max_patches=None, trace=None):
        if isinstance(data, str):
            with open(data, "rb") as file:
                data = file.read()
        if data[4] == PBC3.VERSION:
            return PBC3._decode_to_canvas(data, max_patches=max_patches, trace=trace)

        _, body = cls._open_body(data, trace=trace)
        reader = BitReader(body)
        (
            downsampled, original_w, original_h, color_space, channels,
            channel_bits, positive_bias, has_alpha, patch_count, base_values, stages,
        ) = _read_header(reader, cls.COLOR_SPACE_NAMES)
        initial_w, initial_h, _ = stages[0]
        canvas = _canvas_from_bases((initial_h, initial_w, channels), base_values)
        patches_to_read = (
            patch_count if max_patches is None
            else min(max(0, int(max_patches)), patch_count)
        )
        transition_index = 1
        for patch_index in range(patches_to_read):
            while (
                transition_index < len(stages)
                and stages[transition_index][2] == patch_index
            ):
                width, height, _ = stages[transition_index]
                canvas = cls._resize_canvas(canvas, width, height)
                transition_index += 1
            channel, x, y, width, height, _, values, _ = cls._read_patch(
                reader, channel_bits, positive_bias
            )
            canvas[y:y + height, x:x + width, channel] += ops.signed_resample(
                values, height, width
            )
        if patches_to_read == patch_count:
            while transition_index < len(stages):
                width, height, _ = stages[transition_index]
                canvas = cls._resize_canvas(canvas, width, height)
                transition_index += 1
        return (
            canvas, color_space, downsampled, original_w, original_h,
            canvas.shape[1], canvas.shape[0], has_alpha, channels, patch_count,
        )
