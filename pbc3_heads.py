import time
import numpy as np


class ChannelState:
    def __init__(self, index, target, canvas):
        self.index = int(index)
        self.target = None if target is None else target[:, :, self.index]
        self.canvas = canvas[:, :, self.index]
        self.patches_applied = 0
        self.score = 0.0
        if self.target is not None:
            self.update_score()

    @property
    def visible_canvas(self):
        return np.clip(self.canvas, 0, 255)

    def update_score(self, mode="Sum"):
        if self.target is None:
            self.score = 0.0
        else:
            self.score = float(np.sum(np.abs(self.target - self.visible_canvas)))
        return self.score

    def apply(self, codec, patch, values):
        codec.apply_grid(self.canvas, patch["x"], patch["y"], patch["w"], patch["h"], patch["cell_size"], values)
        self.patches_applied += 1
        if self.target is not None:
            self.update_score()


class DownsampleInitHead:
    def __init__(self, codec):
        self.codec = codec

    def auto_init_candidates(self, residual, w, h, config):
        mean_abs = float(np.mean(np.abs(residual))) + 1.0
        gx = float(np.mean(np.abs(np.diff(residual, axis=1)))) if residual.shape[1] > 1 else 0.0
        gy = float(np.mean(np.abs(np.diff(residual, axis=0)))) if residual.shape[0] > 1 else 0.0
        freq = (gx + gy) / mean_abs; std = float(np.std(residual))
        cell0 = 4 if freq >= 1.0 else 8 if freq >= 0.5 else 12 if freq >= 0.25 else 16 if freq >= 0.12 else 24
        bits0 = 3 if std < 6 else 4 if std < 12 else 5 if std < 24 else 6
        lo_c, hi_c = max(1, int(config.min_cell_size)), min(int(config.max_cell_size), max(w, h))
        max_b = int(config.downsample_palette_bitcount)
        out = []
        for cell, bits in [(cell0, bits0), (cell0, bits0 - 1), (cell0, bits0 + 1), (cell0 // 2, bits0), (cell0 * 2, bits0), (cell0 // 2, bits0 - 1), (cell0 * 2, bits0 + 1), (cell0 // 4, bits0), (cell0 * 4, bits0), (cell0 // 2, bits0 + 1), (cell0 * 2, bits0 - 1), (cell0, bits0 + 2), (cell0, bits0 - 2), (cell0 // 4, bits0 + 1), (cell0 * 4, bits0 - 1)]:
            pair = (max(lo_c, min(hi_c, int(cell))), max(1, min(max_b, int(bits))))
            if pair not in out:
                out.append(pair)
        return out

    def candidates(self, residual, w, h, config):
        return self.auto_init_candidates(residual, w, h, config)[:max(1, int(config.init_search_depth))]

    def select(self, channel, target, canvas, w, h, config, channel_bits):
        codec = self.codec
        base_layer = canvas[:, :, channel]
        residual = target[:, :, channel] - base_layer
        before = target[:, :, channel] - np.clip(base_layer, 0, 255)
        before_sse = float(np.sum(before.astype(np.int64) ** 2))
        cands = self.candidates(residual, w, h, config) if config.auto_downsample_init else [(config.downsample_init_cell_size, config.downsample_palette_bitcount)]
        reductions, bit_costs, built = [], [], []
        for cell, bits in cands:
            patch, values = codec._make_patch(channel, 0, 0, w, h, cell, residual, config, bits, codec.PALETTE_GENERATED, 0)
            delta = codec.signed_resample(values, h, w).astype(np.int32)
            after = target[:, :, channel] - np.clip(base_layer + delta, 0, 255)
            reductions.append(before_sse - float(np.sum(after.astype(np.int64) ** 2)))
            bit_costs.append(codec._patch_bits_for(patch, channel_bits))
            built.append((patch, values, cell, bits))
        q = float(config.q_init)
        scores = q * codec._norm(reductions) - (1.0 - q) * codec._norm(bit_costs)
        return built[int(np.argmax(scores))]


class SearchHead:
    def __init__(self, codec):
        self.codec = codec

    def q_for_step(self, config, step):
        return self.codec._interp(config.search_q_start, config.search_q_end, step, config.patch_count)

    def propose(self, target, canvas, config, rng, step, channel, timings=None):
        return self.search(target, canvas, config, rng, channel, config.search_depth, self.q_for_step(config, step), timings if timings is not None else {})

    def search(self, target, canvas, config, rng, channel, depth, search_q, timings=None):
        codec = self.codec
        timings = timings if timings is not None else {}
        t = time.perf_counter()
        visible_canvas_channel = np.clip(canvas[:, :, channel], 0, 255).astype(np.int32)
        visible_error = (target[:, :, channel] - visible_canvas_channel).astype(np.int64)
        abs_error = np.abs(visible_error)
        integral_abs = codec._integral(abs_error)
        codec._add_time(timings, "visible_error", time.perf_counter() - t)

        t = time.perf_counter()
        anchors = codec._top_anchors(abs_error.astype(np.float32), config.top_k, config.anchor_block_size, channel)
        codec._add_time(timings, "anchors", time.perf_counter() - t)
        if not anchors:
            return []

        h, w, _ = target.shape
        box_sums, box_areas, box_specs = [], [], []
        t = time.perf_counter()
        for i in range(max(1, int(depth))):
            c, x, y, bw, bh, ax, ay = codec._sample_box(rng, anchors[i % len(anchors)], w, h, config)
            box_sum = integral_abs[y + bh, x + bw] - integral_abs[y, x + bw] - integral_abs[y + bh, x] + integral_abs[y, x]
            if box_sum <= 0:
                continue
            box_sums.append(float(box_sum))
            box_areas.append(float(bw * bh))
            box_specs.append((c, x, y, bw, bh))
        codec._add_time(timings, "search_prescore", time.perf_counter() - t)
        if not box_specs:
            return []
        pre_scores = search_q * codec._norm(box_sums) - (1.0 - search_q) * codec._norm(box_areas)
        keep = codec._select_top_indices(pre_scores, config.proposal_depth)
        return [box_specs[i] for i in keep]


class FillerHead:
    def __init__(self, codec, config=None, channel_bits=None, image_shape=None, init_patches=None, original_size=None):
        self.codec = codec
        self.learned = None
        if config is None or not getattr(config, "learned_filler_enabled", False):
            return

        from learned_filler import LearnedFiller
        self.learned = LearnedFiller.load(
            config.learned_filler_model_path,
            top_k=config.learned_filler_top_k,
            q_override=config.learned_filler_q,
            candidates=getattr(config, "learned_filler_candidates", 1),
        )
        if hasattr(self.learned, "begin_encode"):
            h, w, channels = image_shape
            original_w, original_h = original_size
            self.learned.begin_encode(config, channels, channel_bits, w, h, original_w, original_h, init_patches or [])

    def select(self, target, canvas, config, rng, channel_bits, step, current_channel, boxes, canvas_patches, debug_lines, timings):
        if self.learned is not None:
            return self.learned.select_patch(target, canvas, config, rng, channel_bits, step, current_channel)
        return self.select_heuristic(target, canvas, config, channel_bits, step, boxes, canvas_patches, debug_lines, timings)
    
    def candidate_cell_sizes(self, base, config):
        cells = []
        for off in [0, 1, -1, 2, -2, 3, -3]:
            if len(cells) >= max(1, int(config.cell_sizes_per_candidate)):
                break
            cell = int(round(base * (2 ** off)))
            cell = max(int(config.min_cell_size), min(int(config.max_cell_size), cell))
            if cell not in cells:
                cells.append(cell)
        return cells

    def select_heuristic(self, target, canvas, config, channel_bits, step, boxes, canvas_patches, debug_lines, timings):
        codec = self.codec
        if not boxes:
            return None, None
        t = time.perf_counter()
        current_channel = boxes[0][0]
        visible_canvas_channel = np.clip(canvas[:, :, current_channel], 0, 255).astype(np.int32)
        visible_error = (target[:, :, current_channel] - visible_canvas_channel).astype(np.int64)
        integral_signed = codec._integral(visible_error)
        codec._add_time(timings, "integral_signed", time.perf_counter() - t)

        q = codec._interp(config.q_start, config.q_end, step, config.patch_count)
        header_bits = codec._patch_header_bits(channel_bits, config.mask_size)
        bitcount = int(config.patch_palette_bitcount)

        t = time.perf_counter()
        mid_bounds, mid_bits, mid_specs = [], [], []
        for c, x, y, bw, bh in boxes:
            hidden_residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
            base_cell = codec._base_cell_size(hidden_residual, config)
            for cell_size in self.candidate_cell_sizes(base_cell, config):
                cell_size = max(1, min(cell_size, bw, bh))
                bound = codec._box_cell_bound(integral_signed, x, y, bw, bh, cell_size)
                if bound <= 0:
                    continue
                grid_cells = codec._ceil_div(bw, cell_size) * codec._ceil_div(bh, cell_size)
                mid_bounds.append(bound)
                mid_bits.append(header_bits + grid_cells * bitcount)
                mid_specs.append((c, x, y, bw, bh, cell_size))
        codec._add_time(timings, "mid_score", time.perf_counter() - t)
        if not mid_specs:
            return None, None
        mid_scores = q * codec._norm(mid_bounds) - (1.0 - q) * codec._norm(mid_bits)
        mid_specs = [mid_specs[i] for i in codec._select_top_indices(mid_scores, config.exact_depth)]

        reductions, bit_costs, built = [], [], []
        t = time.perf_counter()
        for proposal_i, (c, x, y, bw, bh, cell_size) in enumerate(mid_specs):
            hidden_residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
            before = target[y:y + bh, x:x + bw, c] - np.clip(canvas[y:y + bh, x:x + bw, c], 0, 255)
            before_sse = float(np.sum(before.astype(np.int64) ** 2))
            patch, values = codec._make_patch(c, x, y, bw, bh, cell_size, hidden_residual, config, bitcount, codec.PALETTE_GENERATED, 0)
            delta = codec.signed_resample(values, bh, bw).astype(np.int32)
            after = target[y:y + bh, x:x + bw, c] - np.clip(canvas[y:y + bh, x:x + bw, c] + delta, 0, 255)
            reduction = before_sse - float(np.sum(after.astype(np.int64) ** 2))
            if reduction <= 0:
                continue
            reductions.append(reduction)
            bit_costs.append(codec._patch_bits_for(patch, channel_bits))
            built.append((patch, values))
            if config.debug_mode:
                debug_lines.append(codec._debug_line("CANDIDATE", patch_step=step, canvas_patches=canvas_patches, proposal=proposal_i, channel=c, x=x, y=y, w=bw, h=bh, cell_size=cell_size, bitcount=bitcount, reduction=f"{reduction:.4f}"))
        codec._add_time(timings, "fill_score", time.perf_counter() - t)
        if not built:
            return None, None
        scores = q * codec._norm(reductions) - (1.0 - q) * codec._norm(bit_costs)
        best_i = int(np.argmax(scores))
        best_patch, best_values = built[best_i]
        if config.debug_mode:
            debug_lines.append(codec._debug_line("SELECTED", patch_step=step, canvas_patches=canvas_patches, channel=best_patch["channel"], x=best_patch["x"], y=best_patch["y"], w=best_patch["w"], h=best_patch["h"], cell_size=best_patch["cell_size"], bitcount=best_patch["bitcount"], score=f"{float(scores[best_i]):.6f}"))
        return best_patch, best_values