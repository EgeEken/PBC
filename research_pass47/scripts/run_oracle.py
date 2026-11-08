import copy
import hashlib
import json
import math
import os
import pathlib
import statistics
import shutil
import subprocess
import sys
import time

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = pathlib.Path(__file__).resolve().parents[1]
OUT = PASS / "outputs"
STATES = OUT / "states"
BASELINE = OUT / "baseline"
CODEC = PASS / "codec"
ASSETS = json.loads((ROOT / "research" / "assets.json").read_text())
SOURCES = ("coast.jpg", "googlewiki.png", "mario64.png")
PRESETS = ("compression", "balanced")
STEPS = (1, 25, 50)
SSIMULACRA2 = pathlib.Path(os.environ["SSIMULACRA2"]) if os.environ.get("SSIMULACRA2") else shutil.which("ssimulacra2")
if SSIMULACRA2 is None:
    raise SystemExit("Set SSIMULACRA2 or add ssimulacra2 to PATH")

sys.path.insert(0, str(CODEC))
import learned_filler as lf
import pbc3_ops as ops
import pbc3_stream as stream_ops
from PBC3 import PBC3
from pbc3_types import BitReader, PBC3Config


def sha(data):
    return hashlib.sha256(data).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def rgb_mse(source, recon):
    d = source.astype(np.int32) - recon.astype(np.int32)
    return float(np.mean(d * d, dtype=np.float64))


def patch_signature(patch):
    return {
        "channel": int(patch["channel"]), "x": int(patch["x"]), "y": int(patch["y"]),
        "w": int(patch["w"]), "h": int(patch["h"]), "cell_size": int(patch["cell_size"]),
        "bitcount": int(patch["bitcount"]), "max_bitcount": int(patch["max_bitcount"]),
        "mask": [int(v) for v in patch["mask"]], "negative": int(patch["neg"]),
        "positive": int(patch["pos"]), "index_shape": list(patch["indices"].shape),
        "index_sha256": sha(np.asarray(patch["indices"], dtype=np.uint16).tobytes()),
    }


def parse_patch_sequence(data):
    _, body = PBC3._open_body(data)
    br = BitReader(body)
    header = PBC3._read_header(br)
    channel_bits, positive_bias, count = header[7], header[8], header[10]
    parsed = []
    for _ in range(count):
        channel = br.read(channel_bits)
        x, y, w, h = br.read(16), br.read(16), br.read(16), br.read(16)
        if br.read(1):
            raise RuntimeError("unexpected explicit palette patch")
        mask = [br.read(1) for _ in range(br.read(10))]
        negative, positive, max_bits = br.read(8), br.read(8), br.read(4)
        bitcount = ops.resolve_palette_bitcount(mask, max_bits, negative, positive, positive_bias)
        cell_size = br.read(16)
        grid_n = ops.ceil_div(w, cell_size) * ops.ceil_div(h, cell_size)
        indices = stream_ops.read_grid(br, grid_n, bitcount)
        parsed.append({
            "channel": channel, "x": x, "y": y, "w": w, "h": h,
            "cell_size": cell_size, "bitcount": bitcount, "max_bitcount": max_bits,
            "mask": mask, "negative": negative, "positive": positive,
            "index_shape": [ops.ceil_div(h, cell_size), ops.ceil_div(w, cell_size)],
            "index_sha256": sha(np.asarray(indices, dtype=np.uint16).tobytes()),
        })
    return parsed


def config_for(preset):
    cfg = getattr(PBC3Config, preset)()
    cfg.learned_filler_model_path = str(CODEC / "patch_policy.npz")
    return cfg


def config_json(cfg):
    return plain(cfg.__dict__)


def install_capture_hooks(ctx, sequence):
    original_select = lf.LearnedFiller.select_patch
    original_propose = lf.propose_boxes
    original_build = lf.build_action
    original_writer = PBC3._write_patch

    def propose(*args, **kwargs):
        boxes = original_propose(*args, **kwargs)
        if ctx.get("active") is not None:
            ctx["active"]["proposals"] = [list(map(int, b)) for b in boxes]
            ctx["active"]["rng_after_proposals"] = plain(copy.deepcopy(args[3].bit_generator.state))
        return boxes

    def build(target, canvas, box, config, channel_bits, idx, trace=None):
        started = time.perf_counter()
        result = original_build(target, canvas, box, config, channel_bits, idx, trace=trace)
        if ctx.get("active") is not None:
            ctx["active"]["canonical_builds"].append({
                "action_index": int(idx), "seconds": time.perf_counter() - started,
                "patch_signature": patch_signature(result[0]), "reduction": float(result[3]),
                "bits": float(result[4]),
            })
        return result

    def select(self, target, canvas, config, rng, channel_bits, step, current_channel, trace=None, abs_error=None):
        capture = ctx.get("capture_enabled", True) and int(step) in STEPS
        state = None
        if capture:
            state = {
                "image": ctx["image"], "preset": ctx["preset"], "step": int(step),
                "target": target.copy(), "canvas": canvas.copy(),
                "abs_error": None if abs_error is None else np.array(abs_error, copy=True),
                "channel": int(current_channel), "channel_bits": int(channel_bits),
                "config": config_json(config), "rng_before": plain(copy.deepcopy(rng.bit_generator.state)),
                "policy_bits_before": float(getattr(self, "_bits", 0.0)),
                "policy_pixels": getattr(self, "_pixels", None), "canonical_builds": [],
            }
            ctx["active"] = state
        result = original_select(self, target, canvas, config, rng, channel_bits, step, current_channel, trace=trace, abs_error=abs_error)
        if capture:
            state["rng_after_select"] = plain(copy.deepcopy(rng.bit_generator.state))
            state["canonical_patch"] = None if result[0] is None else patch_signature(result[0])
            state["proposals"] = state.get("proposals", [])
            state["rng_after_proposals"] = state.get("rng_after_proposals")
            file_stub = f"{pathlib.Path(ctx['image']).stem}_{ctx['preset']}_step{int(step):02d}"
            state_path = STATES / f"{file_stub}.npz"
            arrays = {"target": state.pop("target"), "canvas": state.pop("canvas")}
            if state["abs_error"] is not None:
                arrays["abs_error"] = state.pop("abs_error")
            else:
                state.pop("abs_error")
            np.savez_compressed(state_path, **arrays)
            state["state_npz"] = str(state_path.relative_to(PASS))
            state["target_canvas_sha256"] = sha(arrays["target"].tobytes() + arrays["canvas"].tobytes())
            (STATES / f"{file_stub}.json").write_text(json.dumps(plain(state), indent=2) + "\n")
            ctx["states"].append(str((STATES / f"{file_stub}.json").relative_to(PASS)))
            ctx["active"] = None
        return result

    def write_patch(bw, patch, channel_bits):
        sequence.append(patch_signature(patch))
        return original_writer(bw, patch, channel_bits)

    lf.propose_boxes = propose
    lf.build_action = build
    lf.LearnedFiller.select_patch = select
    PBC3._write_patch = staticmethod(write_patch)
    return original_select, original_propose, original_build, original_writer


def restore_hooks(originals):
    lf.LearnedFiller.select_patch, lf.propose_boxes, lf.build_action = originals[:3]
    PBC3._write_patch = staticmethod(originals[3])


def read_source(name):
    path = ROOT / "hpt_data" / name
    raw_hash = sha(path.read_bytes())
    expected = ASSETS["inputs"][f"hpt_data/{name}"]["sha256"]
    if raw_hash != expected:
        raise RuntimeError(f"input hash mismatch: {name}")
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    return path, image, np.asarray(image, dtype=np.uint8), raw_hash


def fresh_decode(stream_path, png_path):
    proc = subprocess.run(
        [sys.executable, str(PASS / "scripts" / "fresh_decode.py"), str(stream_path), str(png_path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_case(name, preset):
    _, image, src, source_hash = read_source(name)
    cfg = config_for(preset)
    stream_path = BASELINE / f"{pathlib.Path(name).stem}_{preset}.pbc"
    warm = PBC3.compress(image, cfg, rng_mode="release")
    base_times, base_results = [], []
    for _ in range(3):
        start = time.perf_counter()
        result = PBC3.compress(image, cfg, rng_mode="release")
        base_times.append(time.perf_counter() - start)
        base_results.append(result)
    base_stream = base_results[-1].data
    if any(r.data != base_stream for r in base_results):
        raise RuntimeError(f"unmodified canonical encoding not deterministic: {name}/{preset}")
    stream_path.write_bytes(base_stream)
    base_pixels = np.asarray(base_results[-1].image.convert("RGB"), dtype=np.uint8)
    base_pixel_hash = sha(base_pixels.tobytes())
    base_sequence = parse_patch_sequence(base_stream)

    ctx = {"image": name, "preset": preset, "states": [], "active": None}
    instrument_sequence = []
    hooks = install_capture_hooks(ctx, instrument_sequence)
    try:
        capture_result = PBC3.compress(image, cfg, rng_mode="release")
        capture_sequence = instrument_sequence.copy()
        instrument_sequence.clear()
        ctx["capture_enabled"] = False
        instrument_repeat = PBC3.compress(image, cfg, rng_mode="release")
    finally:
        restore_hooks(hooks)
    if capture_result.data != base_stream or instrument_repeat.data != base_stream:
        raise RuntimeError(f"instrumentation changed stream: {name}/{preset}")
    capture_pixels = np.asarray(capture_result.image.convert("RGB"), dtype=np.uint8)
    if sha(capture_pixels.tobytes()) != base_pixel_hash:
        raise RuntimeError(f"instrumentation changed pixels: {name}/{preset}")
    if capture_sequence != base_sequence or instrument_sequence != capture_sequence:
        actual_sequence = capture_sequence if capture_sequence != base_sequence else instrument_sequence
        for index, (actual, decoded) in enumerate(zip(actual_sequence, base_sequence)):
            if actual != decoded:
                differences = {key: (actual.get(key), decoded.get(key)) for key in actual if actual.get(key) != decoded.get(key)}
                raise RuntimeError(f"instrumented internal patch sequence differs: {name}/{preset} patch={index} diff={differences}")
        raise RuntimeError(f"instrumented internal patch sequence length or repeat differs: {name}/{preset} capture={len(capture_sequence)} repeat={len(instrument_sequence)} baseline={len(base_sequence)}")

    decoded_png = BASELINE / f"{pathlib.Path(name).stem}_{preset}_decoded.png"
    fresh = fresh_decode(stream_path, decoded_png)
    with Image.open(decoded_png) as im:
        decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if sha(decoded.tobytes()) != base_pixel_hash:
        raise RuntimeError(f"fresh process pixels mismatch: {name}/{preset}")

    history = {}
    pass46 = json.loads((ROOT / "research_pass46" / "outputs" / "baseline" / "baseline_metrics.json").read_text())
    archived = next((r for r in pass46["rows"] if r["image"] == name and r["codec"] == "canonical_pbc31" and r["preset"] == preset), None)
    history_match = archived is not None and archived["stream_sha256"] == sha(base_stream) and archived["decoded_rgb_sha256"] == base_pixel_hash
    return {
        "image": name, "preset": preset, "source_sha256": source_hash,
        "source_rgb_sha256": sha(src.tobytes()), "dimensions": list(image.size),
        "completed_bytes": len(base_stream), "bpp": len(base_stream) * 8 / (image.width * image.height),
        "rgb_mse": rgb_mse(src, decoded), "rgb_psnr": 10 * math.log10(255**2 / max(rgb_mse(src, decoded), 1e-30)),
        "stream_sha256": sha(base_stream), "decoded_rgb_sha256": base_pixel_hash,
        "encode_samples_seconds": base_times, "encode_median_seconds": statistics.median(base_times),
        "fresh_decode": fresh, "canonical_roundtrip_identity": True,
        "instrumented_control_byte_identity": True, "instrumented_patch_sequence_identity": True,
        "instrumented_pixel_identity": True, "historical_pass46_exact_match": history_match,
        "historical_pass46_stream_sha256": archived["stream_sha256"] if archived else None,
        "sampled_states": ctx["states"], "effective_config": config_json(cfg),
    }, src


def candidate_signature(patch):
    return patch_signature(patch)


def reconstruct_rgb(canvas, config, width, height):
    image = PBC3._canvas_to_image(canvas, config.color_space, False)
    if image.size != (width, height):
        image = image.resize((width, height), PBC3.RESAMPLE_FILTER, reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def oracle_state(state_path):
    state_path = PASS / state_path
    meta = json.loads(state_path.read_text())
    with np.load(PASS / meta["state_npz"]) as data:
        target, canvas = data["target"], data["canvas"]
        abs_error = data["abs_error"] if "abs_error" in data else None
    cfg = PBC3Config(**meta["config"])
    rng = np.random.default_rng()
    rng.bit_generator.state = meta["rng_before"]
    t0 = time.perf_counter()
    boxes = lf.propose_boxes(target, canvas, cfg, rng, meta["channel"], meta["step"], abs_error=abs_error)
    proposal_seconds = time.perf_counter() - t0
    boxes_json = [list(map(int, b)) for b in boxes]
    if boxes_json != meta["proposals"]:
        raise RuntimeError(f"proposal replay mismatch: {state_path.name}")
    if plain(rng.bit_generator.state) != meta["rng_after_proposals"] or meta["rng_after_proposals"] != meta["rng_after_select"]:
        raise RuntimeError(f"RNG replay mismatch: {state_path.name}")

    policy = lf.LearnedFiller.load(cfg.learned_filler_model_path, top_k=cfg.learned_filler_top_k,
                                   q_override=cfg.learned_filler_q, candidates=cfg.learned_filler_candidates)
    q = policy.q_override if policy.q_override >= 0 else ops.interp(cfg.q_start, cfg.q_end, meta["step"], cfg.patch_count)
    lam = lf.lambda_for_q(q, policy.lmin, policy.lmax)
    original = ImageOps.exif_transpose(Image.open(ROOT / "hpt_data" / meta["image"])).convert("RGB")
    source = np.asarray(original, dtype=np.uint8)
    base_rgb = reconstruct_rgb(canvas, cfg, original.width, original.height)
    base_mse = rgb_mse(source, base_rgb)
    rows = []
    for box_rank, box in enumerate(boxes[:4], start=1):
        t0 = time.perf_counter()
        feats = policy._featurize(target, canvas, [box], meta["step"], q, target.shape[1], target.shape[0], cfg.patch_count, target.shape[2]).astype(np.float32)
        feats = (feats - policy.feat_mean) / policy.feat_std
        logits, values = policy._forward(feats)
        model_seconds = time.perf_counter() - t0
        action_logits = logits[0]
        best_idx = int(np.argmax(action_logits))
        second_logits = action_logits.copy()
        second_logits[best_idx] = -np.inf
        action_ids = [best_idx, int(np.argmax(second_logits))]
        for action_rank, action_idx in enumerate(action_ids, start=1):
            t0 = time.perf_counter()
            patch, patch_values, delta, reduction, bits = lf.build_action(target, canvas, box, cfg, meta["channel_bits"], action_idx)
            build_seconds = time.perf_counter() - t0
            local_score = float(reduction - lam * bits)
            cand_canvas = canvas.copy()
            c, x, y, w, h = box
            ops.apply_grid(cand_canvas[:, :, c], x, y, w, h, patch["cell_size"], patch_values)
            one_step_mse = rgb_mse(source, reconstruct_rgb(cand_canvas, cfg, original.width, original.height))
            rows.append({
                "box_rank": box_rank, "box": list(map(int, box)), "action_rank": action_rank,
                "action_index": action_idx, "cell_size": lf.ACTIONS[action_idx][0],
                "bitcount_requested": lf.ACTIONS[action_idx][1], "model_value": float(values[0, 0]),
                "model_top_action": best_idx, "ordinary_candidate": box_rank == 1 and action_idx == best_idx,
                "patch_bits": int(bits), "working_sse_reduction": float(reduction),
                "working_rd_score": local_score, "gain_per_bit": float(reduction / bits) if bits else None,
                "feature_and_model_seconds": model_seconds, "patch_build_seconds": build_seconds,
                "candidate_evaluation_seconds": model_seconds + build_seconds,
                "one_step_original_rgb_mse": one_step_mse, "patch_signature": candidate_signature(patch),
            })
    ordinary_sig = meta.get("canonical_patch")
    ordinary = next((r for r in rows if r["ordinary_candidate"]), None)
    if ordinary is not None and ordinary["patch_signature"] != ordinary_sig:
        raise RuntimeError(f"canonical action replay mismatch: {state_path.name}")
    if ordinary is None and ordinary_sig is not None:
        raise RuntimeError(f"missing ordinary action candidate: {state_path.name}")
    best_alt = None
    if ordinary is not None:
        alternatives = [r for r in rows if not r["ordinary_candidate"]]
        if alternatives:
            best_alt = max(alternatives, key=lambda r: r["working_rd_score"])
        ordinary_score = ordinary["working_rd_score"]
        if best_alt is not None:
            best_alt["delta_local_rd_score"] = best_alt["working_rd_score"] - ordinary_score
            best_alt["relative_local_rd_score_gain"] = (best_alt["working_rd_score"] - ordinary_score) / max(abs(ordinary_score), 1e-30)
            best_alt["relative_patch_bits_change"] = best_alt["patch_bits"] / max(ordinary["patch_bits"], 1)
            best_alt["qualifies_gate_b"] = bool(
                ordinary_score > 0 and best_alt["working_rd_score"] >= ordinary_score * 1.05
                and best_alt["patch_bits"] <= ordinary["patch_bits"] * 1.05
            )
    return {
        "state": str(state_path.relative_to(PASS)), "image": meta["image"], "preset": meta["preset"],
        "step": meta["step"], "channel": meta["channel"], "rng_replay_exact": True,
        "rng_before_sha256": sha(json.dumps(meta["rng_before"], sort_keys=True).encode()),
        "rng_after_sha256": sha(json.dumps(meta["rng_after_select"], sort_keys=True).encode()),
        "proposal_pool_sha256": sha(json.dumps(meta["proposals"], separators=(",", ":")).encode()),
        "target_canvas_sha256": meta["target_canvas_sha256"],
        "random_samples": cfg.search_depth, "retained_proposals": len(boxes),
        "boxes_evaluated": min(4, len(boxes)), "patch_candidates_evaluated": len(rows),
        "proposal_replay_seconds": proposal_seconds, "lambda": lam,
        "pre_patch_original_rgb_mse": base_mse,
        "canonical_builds": meta["canonical_builds"], "ordinary_candidate": ordinary,
        "best_local_alternative": best_alt, "candidates": rows,
    }


def main():
    if "--oracle-only" in sys.argv:
        result_path = OUT / "oracle_results.json"
        result = json.loads(result_path.read_text())
        result["oracle_states"] = [oracle_state(p) for case in result["cases"] for p in case["sampled_states"]]
        qualifying = [r for r in result["oracle_states"] if r["best_local_alternative"] and r["best_local_alternative"]["qualifies_gate_b"]]
        images = sorted({r["image"] for r in qualifying})
        result["gates"]["B_local_candidate_opportunity"] = "pass" if len(qualifying) >= 3 and len(images) >= 2 else "fail"
        result["gates"]["B_qualifying_states"] = len(qualifying)
        result["gates"]["B_distinct_images"] = images
        result_path.write_text(json.dumps(plain(result), indent=2) + "\n")
        print(json.dumps(result["gates"], indent=2))
        return
    OUT.mkdir(exist_ok=True)
    STATES.mkdir(exist_ok=True)
    BASELINE.mkdir(exist_ok=True)
    case_rows, source_arrays = [], {}
    for name in SOURCES:
        for preset in PRESETS:
            print(f"control {name}/{preset}", flush=True)
            row, src = run_case(name, preset)
            case_rows.append(row)
            source_arrays[name] = src
    oracle_rows = [oracle_state(p) for row in case_rows for p in row["sampled_states"]]
    qualifying = [
        r for r in oracle_rows
        if r["best_local_alternative"] and r["best_local_alternative"]["qualifies_gate_b"]
    ]
    distinct_images = sorted({r["image"] for r in qualifying})
    gate_b = len(qualifying) >= 3 and len(distinct_images) >= 2
    result = {
        "pass": 47,
        "source_revision": "3723dc14de88c6c3dcbd98dd80a097e019a92a9f",
        "branch": "research/experiments",
        "runtime": {"python": sys.version, "numpy": np.__version__, "pillow": Image.__version__, "threads": 1},
        "model_sha256": sha((ROOT / "patch_policy.npz").read_bytes()),
        "cases": case_rows,
        "oracle_states": oracle_rows,
        "gates": {
            "A_canonical_instrumented_control_identity": all(
                r["canonical_roundtrip_identity"] and r["instrumented_control_byte_identity"]
                and r["instrumented_patch_sequence_identity"] and r["instrumented_pixel_identity"]
                for r in case_rows
            ),
            "B_local_candidate_opportunity": "pass" if gate_b else "fail",
            "B_qualifying_states": len(qualifying),
            "B_distinct_images": distinct_images,
            "C_completed_file_counterfactual": "pending" if gate_b else "not attempted: Gate B failed",
            "E_selection_rule": "not attempted",
        },
    }
    (OUT / "oracle_results.json").write_text(json.dumps(plain(result), indent=2) + "\n")
    print(json.dumps({"gates": result["gates"], "case_bytes": [(r["image"], r["preset"], r["completed_bytes"]) for r in case_rows], "states": len(oracle_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
