import math
import time
from dataclasses import replace

import numpy as np
from PIL import Image

import learned_filler
import pbc3_ops as ops
from PBC3 import PBC3, _canvas_from_bases
from learned_filler import (
    ACTIONS,
    BITCOUNTS,
    CELL_SIZES,
    MASK_SIZES,
    STOP_AXIS,
    STOP_STOP,
    build_action,
    build_action_factored,
    extract_state,
    lambda_for_q,
)
from pbc3_heads import DownsampleInitHead, FillerHead


MIX = {"coarse": 0.5, "dynamic": 0.4, "explore": 0.1}
BASELINE_PROPOSE_BOXES = learned_filler.propose_boxes


def _clone_rng(rng, salt=0):
    clone = ops.PBC3Rng(0)
    clone.state = (int(rng.state) ^ int(salt)) & ((1 << 64) - 1)
    return clone


def _rank_boxes(abs_error, boxes, search_q):
    if not boxes:
        return []
    integral_abs = ops.integral(abs_error)
    sums = []
    areas = []
    for _, x, y, width, height in boxes:
        total = (
            integral_abs[y + height, x + width]
            - integral_abs[y, x + width]
            - integral_abs[y + height, x]
            + integral_abs[y, x]
        )
        sums.append(float(total))
        areas.append(float(width * height))
    scores = search_q * ops.norm(sums) - (1.0 - search_q) * ops.norm(areas)
    order = np.argsort(scores)[::-1]
    return [(boxes[int(index)], float(scores[int(index)])) for index in order]


def _sample_boxes(rng, anchors, width, height, config, depth):
    if not anchors:
        return []
    boxes = []
    for index in range(max(1, int(depth))):
        box = ops.sample_box(
            rng, anchors[index % len(anchors)], width, height, config
        )[:5]
        boxes.append(box)
    return boxes


def _coarse_parent_boxes(abs_error, config, rng, channel, depth):
    height, width = abs_error.shape
    coarse_width = max(1, int(round(width * 0.5)))
    coarse_height = max(1, int(round(height * 0.5)))
    coarse = np.asarray(
        Image.fromarray(abs_error.astype(np.float32)).resize(
            (coarse_width, coarse_height), Image.Resampling.BOX
        ),
        dtype=np.float32,
    )
    coarse_config = replace(
        config,
        min_patch_size=max(1, int(round(config.min_patch_size * 0.5))),
        max_patch_size=max(1, int(round(config.max_patch_size * 0.5))),
        anchor_block_size=max(1, int(round(config.anchor_block_size * 0.5))),
    )
    anchors = ops.top_anchors(
        coarse, config.top_k, coarse_config.anchor_block_size, channel
    )
    sampled = _sample_boxes(
        rng, anchors, coarse_width, coarse_height, coarse_config, depth
    )
    scale_x = width / coarse_width
    scale_y = height / coarse_height
    mapped = []
    for current_channel, x, y, box_width, box_height in sampled:
        x0 = min(width - 1, max(0, int(round(x * scale_x))))
        y0 = min(height - 1, max(0, int(round(y * scale_y))))
        x1 = min(width, max(x0 + 1, int(round((x + box_width) * scale_x))))
        y1 = min(height, max(y0 + 1, int(round((y + box_height) * scale_y))))
        mapped.append((current_channel, x0, y0, x1 - x0, y1 - y0))
    return mapped


def mixed_propose_boxes(target, canvas, config, rng, channel, step, trace=None):
    start = time.perf_counter()
    height, width, _ = target.shape
    visible = np.clip(canvas[:, :, channel], 0, 255).astype(np.int32)
    abs_error = np.abs(target[:, :, channel] - visible)
    search_q = ops.interp(
        config.search_q_start, config.search_q_end, step, config.patch_count
    )
    total_depth = max(1, int(config.search_depth))
    total_keep = max(1, int(config.proposal_depth))
    depths = {
        name: max(1, int(round(total_depth * fraction)))
        for name, fraction in MIX.items()
    }
    keeps = {
        name: max(1, int(round(total_keep * fraction)))
        for name, fraction in MIX.items()
    }
    keep_delta = total_keep - sum(keeps.values())
    keeps["dynamic"] += keep_delta

    dynamic_rng = _clone_rng(rng)
    anchors = ops.top_anchors(
        abs_error, config.top_k, config.anchor_block_size, channel
    )
    dynamic = _sample_boxes(
        dynamic_rng, anchors, width, height, config, depths["dynamic"]
    )
    rng.state = dynamic_rng.state

    coarse = _coarse_parent_boxes(
        abs_error,
        config,
        _clone_rng(rng, 0x9E3779B97F4A7C15),
        channel,
        depths["coarse"],
    )
    explore_rng = _clone_rng(rng, 0xD1B54A32D192ED03)
    explore_anchors = [
        (
            channel,
            explore_rng.integers(0, height),
            explore_rng.integers(0, width),
        )
        for _ in range(max(1, min(config.top_k, depths["explore"])))
    ]
    explore = _sample_boxes(
        explore_rng,
        explore_anchors,
        width,
        height,
        config,
        depths["explore"],
    )

    selected = []
    for source, boxes in (
        ("coarse", coarse),
        ("dynamic", dynamic),
        ("explore", explore),
    ):
        ranked = _rank_boxes(abs_error, boxes, search_q)[:keeps[source]]
        selected.extend(
            {"box": box, "source": source, "group_score": score}
            for box, score in ranked
        )

    deduplicated = {}
    for candidate in selected:
        deduplicated.setdefault(candidate["box"], candidate)
    selected = list(deduplicated.values())
    ranked = _rank_boxes(
        abs_error, [candidate["box"] for candidate in selected], search_q
    )
    by_box = {candidate["box"]: candidate for candidate in selected}
    output = []
    for box, score in ranked[:total_keep]:
        candidate = by_box[box]
        output.append({**candidate, "score": score})
    return output, time.perf_counter() - start


def _exact_for_box(model, target, canvas, config, channel_bits, step, box):
    height, width, channels = target.shape
    if hasattr(model, "Wa"):
        q = (
            model.q_override
            if model.q_override >= 0
            else ops.interp(config.q_start, config.q_end, step, config.patch_count)
        )
        features = model._featurize(
            target,
            canvas,
            [box],
            step,
            q,
            width,
            height,
            config.patch_count,
            channels,
        ).astype(np.float32)
        features = (features - model.feat_mean) / model.feat_std
        logits, _ = model._forward(features)
        action_index = int(logits[0].argmax())
        patch, values, delta, reduction, bits = build_action(
            target,
            canvas,
            box,
            config,
            channel_bits,
            action_index,
        )
        if reduction <= 0:
            return None
        return {
            "patch": patch,
            "values": values,
            "delta": delta,
            "reduction": float(reduction),
            "bits": float(bits),
            "rd": float(reduction - lambda_for_q(q, model.lmin, model.lmax) * bits),
            "cell_size": ACTIONS[action_index][0],
            "bitcount": ACTIONS[action_index][1],
            "mask_size": config.mask_size,
        }

    pixels = model._pixels or max(1, height * width)
    state = extract_state(
        target,
        canvas,
        box,
        step,
        width,
        height,
        config.patch_count,
        channels,
        model._lambda(config),
        model._bits,
        pixels,
    ).astype(np.float32)
    state = (state - model.feat_mean) / model.feat_std
    logits = model._forward(state)
    indices = [int(np.argmax(values)) for values in logits]
    if len(indices) > STOP_AXIS and indices[STOP_AXIS] == STOP_STOP:
        return None
    patch, values, delta, reduction, bits = build_action_factored(
        target,
        canvas,
        box,
        config,
        channel_bits,
        indices[0],
        indices[1],
        indices[2],
    )
    if reduction <= 0:
        return None
    return {
        "patch": patch,
        "values": values,
        "delta": delta,
        "reduction": float(reduction),
        "bits": float(bits),
        "rd": float(reduction - model._lambda(config) * bits),
        "cell_size": CELL_SIZES[indices[0]],
        "bitcount": BITCOUNTS[indices[1]],
        "mask_size": MASK_SIZES[indices[2]],
    }


def replay_image(image, config, image_name, image_type, capture_steps):
    prep = PBC3.prepare(image, config)
    target = prep["target"]
    height, width, channels = target.shape
    channel_bits = max(1, math.ceil(math.log2(channels)))
    bases = [
        int(round(float(np.mean(prep["arr"][:, :, channel]))))
        for channel in range(channels)
    ]
    canvas = _canvas_from_bases((height, width, channels), bases)
    patches = []
    init_head = DownsampleInitHead()
    for channel in range(channels):
        patch, values, delta, _, _ = init_head.select(
            channel, target, canvas, width, height, config, channel_bits
        )
        if config.reuse_selected_delta:
            ops.apply_delta(canvas[:, :, channel], 0, 0, width, height, delta)
        else:
            ops.apply_grid(
                canvas[:, :, channel],
                0,
                0,
                width,
                height,
                patch["cell_size"],
                values,
            )
        patches.append(patch)

    filler = FillerHead(
        config,
        channel_bits,
        canvas.shape,
        patches,
        (prep["original_w"], prep["original_h"]),
    )
    rng = ops.PBC3Rng(config.random_seed)
    channel_scores = [
        PBC3._channel_sum_error(target, canvas, channel)
        for channel in range(channels)
    ]
    rows = []
    for step in range(1, config.patch_count + 1):
        current_channel = PBC3._choose_channel(
            channel_scores, step, channels, config.channel_cycle
        )
        if step in capture_steps:
            baseline_rng = _clone_rng(rng)
            baseline_start = time.perf_counter()
            baseline_boxes = learned_filler.propose_boxes(
                target,
                canvas,
                config,
                baseline_rng,
                current_channel,
                step,
            )
            baseline_seconds = time.perf_counter() - baseline_start
            mixed, mixed_seconds = mixed_propose_boxes(
                target,
                canvas,
                config,
                _clone_rng(rng),
                current_channel,
                step,
            )
            if baseline_boxes:
                baseline_box = baseline_boxes[0]
                baseline_exact = _exact_for_box(
                    filler.learned,
                    target,
                    canvas,
                    config,
                    channel_bits,
                    step,
                    baseline_box,
                )
                if baseline_exact is not None:
                    row = {
                        "image": image_name,
                        "image_type": image_type,
                        "preset_step": step,
                        "channel": current_channel,
                        "width": width,
                        "height": height,
                        "baseline_proposals": len(baseline_boxes),
                        "mixed_proposals": len(mixed),
                        "baseline_proposal_seconds": baseline_seconds,
                        "mixed_proposal_seconds": mixed_seconds,
                        "baseline_box": baseline_box,
                        "baseline_rd": baseline_exact["rd"],
                        "baseline_reduction": baseline_exact["reduction"],
                        "baseline_bits": baseline_exact["bits"],
                    }
                    exact_cache = {}
                    for index, candidate in enumerate(mixed[:10]):
                        exact_cache[index] = _exact_for_box(
                            filler.learned,
                            target,
                            canvas,
                            config,
                            channel_bits,
                            step,
                            candidate["box"],
                        )
                    for top_k in (1, 4, 10):
                        prefix = mixed[:top_k]
                        exact = [
                            (index, exact_cache[index])
                            for index in range(len(prefix))
                            if exact_cache[index] is not None
                        ]
                        best = max(exact, key=lambda item: item[1]["rd"]) if exact else None
                        exact_box_recall = any(
                            candidate["box"] == baseline_box for candidate in prefix
                        )
                        equal_or_better = bool(
                            best is not None
                            and best[1]["rd"] >= baseline_exact["rd"] - 1e-9
                        )
                        best_rd = best[1]["rd"] if best is not None else None
                        row[f"top{top_k}_box_recall"] = exact_box_recall
                        row[f"top{top_k}_equal_or_better"] = equal_or_better
                        row[f"top{top_k}_recall"] = (
                            exact_box_recall or equal_or_better
                        )
                        row[f"top{top_k}_best_rd"] = best_rd
                        row[f"top{top_k}_relative_rd"] = (
                            None
                            if best_rd is None
                            else (best_rd - baseline_exact["rd"])
                            / max(abs(baseline_exact["rd"]), 1.0)
                        )
                        row[f"top{top_k}_best_source"] = (
                            None if best is None else mixed[best[0]]["source"]
                        )
                        row[f"top{top_k}_source_counts"] = {
                            source: sum(
                                candidate["source"] == source
                                for candidate in prefix
                            )
                            for source in MIX
                        }
                        row[f"top{top_k}_exact_evaluations"] = len(exact)
                        row[f"top{top_k}_exact_avoided_vs_10"] = 10 - len(exact)
                    rows.append(row)

        patch, values, delta = filler.select(
            target,
            canvas,
            config,
            rng,
            channel_bits,
            step,
            current_channel,
            None,
            len(patches),
            [],
        )
        if patch is None:
            break
        channel = patch["channel"]
        if config.reuse_selected_delta:
            ops.apply_delta(
                canvas[:, :, channel],
                patch["x"],
                patch["y"],
                patch["w"],
                patch["h"],
                delta,
            )
        else:
            ops.apply_grid(
                canvas[:, :, channel],
                patch["x"],
                patch["y"],
                patch["w"],
                patch["h"],
                patch["cell_size"],
                values,
            )
        patches.append(patch)
        channel_scores[channel] = PBC3._channel_sum_error(
            target, canvas, channel
        )
    return rows


def proposal_function(target, canvas, config, rng, channel, step, trace=None):
    candidates, _ = mixed_propose_boxes(
        target, canvas, config, rng, channel, step, trace=trace
    )
    return [candidate["box"] for candidate in candidates]


def exact_fallback_proposal_function(
    target, canvas, config, rng, channel, step, trace=None
):
    mixed_propose_boxes(
        target,
        canvas,
        config,
        _clone_rng(rng),
        channel,
        step,
        trace=trace,
    )
    return BASELINE_PROPOSE_BOXES(
        target, canvas, config, rng, channel, step, trace=trace
    )
