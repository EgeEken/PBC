import numpy as np

import pbc3_ops as ops
from learned_filler import LearnedFiller
from pbc3_trace import timed


class DownsampleInitHead:
    """## Picks the initial full-image downsample initialization patch for each channel"""

    def auto_init_candidates(self, residual, w: int, h: int, config) -> list[tuple[int, int]]:
        """## Returns likely cell-size/bitcount pairs for the init patch"""
        mean_abs = float(np.mean(np.abs(residual))) + 1.0
        gx = float(np.mean(np.abs(np.diff(residual, axis=1)))) if residual.shape[1] > 1 else 0.0
        gy = float(np.mean(np.abs(np.diff(residual, axis=0)))) if residual.shape[0] > 1 else 0.0
        freq = (gx + gy) / mean_abs
        std = float(np.std(residual))
        cell0 = 4 if freq >= 1.0 else 8 if freq >= 0.5 else 12 if freq >= 0.25 else 16 if freq >= 0.12 else 24
        bits0 = 3 if std < 6 else 4 if std < 12 else 5 if std < 24 else 6
        lo_c = max(1, int(config.min_cell_size))
        hi_c = min(int(config.max_cell_size), max(w, h))
        max_b = int(config.downsample_palette_bitcount)
        out = []
        for cell, bits in [
            (cell0, bits0), (cell0, bits0 - 1), (cell0, bits0 + 1),
            (cell0 // 2, bits0), (cell0 * 2, bits0),
            (cell0 // 2, bits0 - 1), (cell0 * 2, bits0 + 1),
            (cell0 // 4, bits0), (cell0 * 4, bits0),
            (cell0 // 2, bits0 + 1), (cell0 * 2, bits0 - 1),
            (cell0, bits0 + 2), (cell0, bits0 - 2),
            (cell0 // 4, bits0 + 1), (cell0 * 4, bits0 - 1),
        ]:
            pair = (max(lo_c, min(hi_c, int(cell))), max(1, min(max_b, int(bits))))
            if pair not in out:
                out.append(pair)
        return out

    def candidates(self, residual, w: int, h: int, config) -> list[tuple[int, int]]:
        """## Returns the init candidates capped by init_search_depth"""
        return self.auto_init_candidates(residual, w, h, config)[:max(1, int(config.init_search_depth))]

    def select(self, channel: int, target, canvas, w: int, h: int, config, channel_bits: int, trace=None):
        """## Builds init candidates and returns the best patch/value pair"""
        with timed(trace, "action.downsample_init.residual"):
            base_layer = canvas[:, :, channel]
            residual = target[:, :, channel] - base_layer
            before = target[:, :, channel] - np.clip(base_layer, 0, 255)
            before_sse = float(np.sum(before.astype(np.int64) ** 2))
        with timed(trace, "action.downsample_init.candidates"):
            if config.auto_downsample_init:
                cands = self.candidates(residual, w, h, config)
            else:
                cands = [(config.downsample_init_cell_size, config.downsample_palette_bitcount)]

        reductions, bit_costs, built = [], [], []
        for cell, bits in cands:
            with timed(trace, "action.downsample_init.build_candidate", cell_size=cell, bitcount=bits):
                patch, values = ops.make_patch(channel, 0, 0, w, h, cell, residual, config, bits, trace=trace)
            with timed(trace, "action.downsample_init.resample_candidate", cell_size=cell):
                delta = ops.signed_resample(values, h, w, trace=trace, purpose="candidate_reconstruction")
            with timed(trace, "action.downsample_init.score_candidate", cell_size=cell, bitcount=bits):
                after = target[:, :, channel] - np.clip(base_layer + delta, 0, 255)
                reductions.append(before_sse - float(np.sum(after.astype(np.int64) ** 2)))
                bit_costs.append(ops.patch_bits_for(patch, channel_bits))
                built.append((patch, values, delta, cell, bits))
                if trace is not None:
                    trace.count("candidates_exact_scored")

        with timed(trace, "action.downsample_init.select_candidate"):
            q = float(config.q_init)
            scores = q * ops.norm(reductions) - (1.0 - q) * ops.norm(bit_costs)
            return built[int(np.argmax(scores))]


class SearchHead:
    """## Proposes candidate patch bounding boxes for a channel"""

    def q_for_step(self, config, step: int) -> float:
        """## Returns the search quality/bit-cost mix for this patch step"""
        return ops.interp(config.search_q_start, config.search_q_end, step, config.patch_count)

    def propose(self, target, canvas, config, rng, step: int, channel: int, trace=None):
        """## Returns candidate boxes using the configured search depth"""
        with timed(trace, "action.search.quality_mix"):
            search_q = self.q_for_step(config, step)
        return self.search(target, canvas, config, rng, channel, config.search_depth, search_q, trace=trace)

    def search(self, target, canvas, config, rng, channel: int, depth: int, search_q: float, trace=None) -> list[tuple[int, int, int, int, int]]:
        """## Samples boxes around strong error anchors and keeps the best prescores"""
        return ops.search_boxes(target, canvas, config, rng, channel, depth, search_q, trace=trace)


class FillerHead:
    """## Fills candidate boxes, making patches, either through the learned policy or heuristic fill"""

    def __init__(self, config=None, channel_bits: int = None, image_shape=None, init_patches=None, original_size=None, trace=None) -> None:
        self.learned = None
        if config is None or not getattr(config, "learned_filler_enabled", False):
            return

        with timed(trace, "action.filler.load_policy"):
            self.learned = LearnedFiller.load(
                config.learned_filler_model_path,
                top_k=config.learned_filler_top_k,
                q_override=config.learned_filler_q,
                candidates=getattr(config, "learned_filler_candidates", 1),
            )
        if hasattr(self.learned, "begin_encode"):
            h, w, channels = image_shape
            original_w, original_h = original_size
            with timed(trace, "action.filler.begin_policy_encode"):
                self.learned.begin_encode(config, channels, channel_bits, w, h, original_w, original_h, init_patches or [])

    def select(self, target, canvas, config, rng, channel_bits: int, step: int, current_channel: int, boxes, canvas_patches: int, debug_lines, trace=None):
        """## Returns the next patch/value pair, or (None, None) when encoding should stop"""
        if self.learned is not None:
            with timed(trace, "action.filler.learned_select", step=step):
                return self.learned.select_patch(target, canvas, config, rng, channel_bits, step, current_channel, trace=trace)
        with timed(trace, "action.filler.heuristic_select", step=step):
            return self.select_heuristic(target, canvas, config, channel_bits, step, boxes, canvas_patches, debug_lines, trace=trace)

    def candidate_cell_sizes(self, base: int, config) -> list[int]:
        """## Returns nearby power-of-two cell sizes around a base suggestion"""
        cells = []
        for off in [0, 1, -1, 2, -2, 3, -3]:
            if len(cells) >= max(1, int(config.cell_sizes_per_candidate)):
                break
            cell = int(round(base * (2 ** off)))
            cell = max(int(config.min_cell_size), min(int(config.max_cell_size), cell))
            if cell not in cells:
                cells.append(cell)
        return cells

    def select_heuristic(self, target, canvas, config, channel_bits: int, step: int, boxes, canvas_patches: int, debug_lines, trace=None):
        """## Exact-scores patch fills for candidate boxes and returns the best one"""
        if not boxes:
            return None, None, None
        with timed(trace, "action.filler.heuristic.visible_error"):
            current_channel = boxes[0][0]
            visible_canvas_channel = np.clip(canvas[:, :, current_channel], 0, 255).astype(np.int32)
            visible_error = (target[:, :, current_channel] - visible_canvas_channel).astype(np.int64)
        with timed(trace, "action.filler.heuristic.integral_error"):
            integral_signed = ops.integral(visible_error)

        q = ops.interp(config.q_start, config.q_end, step, config.patch_count)
        header_bits = ops.patch_header_bits(channel_bits, config.mask_size)
        bitcount = int(config.patch_palette_bitcount)

        mid_bounds, mid_bits, mid_specs, mid_rhos = [], [], [], []
        selection_mode = getattr(config, "cell_selection_mode", "gradient")
        representability_threshold = float(getattr(config, "representability_threshold", 0.0))
        with timed(trace, "action.filler.heuristic.mid_candidates"):
            for c, x, y, bw, bh in boxes:
                hidden_residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
                base_cell = ops.base_cell_size(hidden_residual, config)
                for cell_size in self.candidate_cell_sizes(base_cell, config):
                    cell_size = max(1, min(cell_size, bw, bh))
                    bound = ops.box_cell_bound(integral_signed, x, y, bw, bh, cell_size)
                    if bound <= 0:
                        continue
                    grid_cells = ops.ceil_div(bw, cell_size) * ops.ceil_div(bh, cell_size)
                    rho = None
                    if selection_mode == "hybrid":
                        rho = ops.cell_representability(hidden_residual, cell_size)
                        if trace is not None:
                            trace.count("representability_evaluated")
                    mid_bounds.append(bound)
                    mid_bits.append(header_bits + grid_cells * bitcount)
                    mid_specs.append((c, x, y, bw, bh, cell_size))
                    mid_rhos.append(rho)
        if not mid_specs:
            return None, None, None

        if selection_mode == "hybrid" and representability_threshold > 0:
            keep = [i for i, rho in enumerate(mid_rhos) if rho >= representability_threshold]
            if keep:
                mid_bounds = [mid_bounds[i] for i in keep]
                mid_bits = [mid_bits[i] for i in keep]
                mid_specs = [mid_specs[i] for i in keep]

        with timed(trace, "action.filler.heuristic.mid_score"):
            mid_scores = q * ops.norm(mid_bounds) - (1.0 - q) * ops.norm(mid_bits)
        with timed(trace, "action.filler.heuristic.select_exact_candidates"):
            mid_specs = [mid_specs[i] for i in ops.select_top_indices(mid_scores, config.exact_depth)]

        reductions, bit_costs, built = [], [], []
        for proposal_i, (c, x, y, bw, bh, cell_size) in enumerate(mid_specs):
            with timed(trace, "action.filler.heuristic.build_candidate", proposal=proposal_i, cell_size=cell_size):
                hidden_residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
                before = target[y:y + bh, x:x + bw, c] - np.clip(canvas[y:y + bh, x:x + bw, c], 0, 255)
                before_sse = float(np.sum(before.astype(np.int64) ** 2))
                patch, values = ops.make_patch(
                    c, x, y, bw, bh, cell_size, hidden_residual, config, bitcount, trace=trace,
                )
                delta = ops.signed_resample(values, bh, bw, trace=trace, purpose="candidate_reconstruction")
                after = target[y:y + bh, x:x + bw, c] - np.clip(canvas[y:y + bh, x:x + bw, c] + delta, 0, 255)
                reduction = before_sse - float(np.sum(after.astype(np.int64) ** 2))
                if trace is not None:
                    trace.count("candidates_exact_scored")
                if reduction <= 0:
                    continue
                reductions.append(reduction)
                bit_costs.append(ops.patch_bits_for(patch, channel_bits))
                built.append((patch, values, delta))
            if config.debug_mode:
                debug_lines.append(ops.debug_line("CANDIDATE", patch_step=step, canvas_patches=canvas_patches, proposal=proposal_i, channel=c, x=x, y=y, w=bw, h=bh, cell_size=cell_size, bitcount=bitcount, reduction=f"{reduction:.4f}"))
        if not built:
            return None, None, None

        with timed(trace, "action.filler.heuristic.final_score"):
            scores = q * ops.norm(reductions) - (1.0 - q) * ops.norm(bit_costs)
            best_i = int(np.argmax(scores))
            best_patch, best_values, best_delta = built[best_i]
        if config.debug_mode:
            debug_lines.append(ops.debug_line("SELECTED", patch_step=step, canvas_patches=canvas_patches, channel=best_patch["channel"], x=best_patch["x"], y=best_patch["y"], w=best_patch["w"], h=best_patch["h"], cell_size=best_patch["cell_size"], bitcount=best_patch["bitcount"], score=f"{float(scores[best_i]):.6f}"))
        return best_patch, best_values, best_delta
