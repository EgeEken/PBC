import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import pbc3_ops as ops
from PBC3 import PBC3, preload_numba
from learned_filler import build_action, propose_boxes
from pbc3_features import extract_cheap, extract_features
from pbc3_heads import DownsampleInitHead
from pbc3_types import BitWriter, PBC3Config

PRESETS = {
    "compression": PBC3Config.compression,
    "balanced": PBC3Config.balanced,
    "quality": PBC3Config.quality,
    "high_quality": PBC3Config.high_quality,
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def image_paths(folder):
    return sorted(path for path in Path(folder).glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def init_canvas(prep, config):
    target = prep["target"]
    h, w, channels = target.shape
    base_values = [int(round(float(np.mean(prep["arr"][:, :, c])))) for c in range(channels)]
    canvas = np.empty((h, w, channels), dtype=np.int32)
    for c, value in enumerate(base_values):
        canvas[:, :, c] = value
    channel_bits = max(1, math.ceil(math.log2(channels)))
    patches = []
    head = DownsampleInitHead()
    for c in range(channels):
        patch, _, delta, _, _ = head.select(c, target, canvas, w, h, config, channel_bits)
        ops.apply_delta(canvas[:, :, c], 0, 0, w, h, delta)
        patches.append(patch)
    return canvas, patches


class Policy(torch.nn.Module):
    def __init__(self, in_dim, hidden, layers, actions):
        super().__init__()
        self.trunk = torch.nn.ModuleList(
            [torch.nn.Linear(in_dim, hidden)]
            + [torch.nn.Linear(hidden, hidden) for _ in range(layers - 1)]
        )
        self.action = torch.nn.Linear(hidden, actions)
        self.value = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        for layer in self.trunk:
            x = F.silu(layer(x))
        return self.action(x), self.value(x).squeeze(-1)


def load_policy(path):
    data = dict(np.load(path, allow_pickle=True))
    layers = int(data.get("layer_count", 2))
    model = Policy(int(data["in_dim"]), int(data["hidden"]), layers, int(data["num_actions"]))
    state = model.state_dict()
    if "layer_count" in data:
        for i in range(layers):
            state[f"trunk.{i}.weight"] = torch.from_numpy(data[f"W{i}"])
            state[f"trunk.{i}.bias"] = torch.from_numpy(data[f"b{i}"])
    else:
        state["trunk.0.weight"] = torch.from_numpy(data["W0"])
        state["trunk.0.bias"] = torch.from_numpy(data["b0"])
        state["trunk.1.weight"] = torch.from_numpy(data["W2"])
        state["trunk.1.bias"] = torch.from_numpy(data["b2"])
    state["action.weight"] = torch.from_numpy(data["Wa"])
    state["action.bias"] = torch.from_numpy(data["ba"])
    state["value.weight"] = torch.from_numpy(data["Wv"])
    state["value.bias"] = torch.from_numpy(data["bv"])
    model.load_state_dict(state)
    return model, data


def export(model, source, path):
    state = model.state_dict()
    values = {
        "hidden": np.int64(model.trunk[0].out_features),
        "layer_count": np.int64(len(model.trunk)),
        "in_dim": np.int64(model.trunk[0].in_features),
        "num_actions": np.int64(model.action.out_features),
        "feat_mean": source["feat_mean"],
        "feat_std": source["feat_std"],
        "lmin": source["lmin"],
        "lmax": source["lmax"],
        "cheap": source["cheap"],
        "feature_names": source["feature_names"],
        "Wa": state["action.weight"].detach().numpy(),
        "ba": state["action.bias"].detach().numpy(),
        "Wv": state["value.weight"].detach().numpy(),
        "bv": state["value.bias"].detach().numpy(),
    }
    for i in range(len(model.trunk)):
        values[f"W{i}"] = state[f"trunk.{i}.weight"].detach().numpy()
        values[f"b{i}"] = state[f"trunk.{i}.bias"].detach().numpy()
    np.savez_compressed(path, **values)


def checkpoint_manifest_path(output):
    return Path(f"{output}.checkpoints.json")


def checkpoint_path(output, metric, epoch):
    base = Path(output)
    return base.with_name(f"{base.stem}.best_{metric}.epoch{epoch}.npz")


def save_checkpoint_groups(model, source, output, groups):
    manifest = checkpoint_manifest_path(output)
    manifest.write_text(json.dumps({"groups": groups}, indent=2), encoding="utf-8")


def update_checkpoint_groups(model, source, output, groups, epoch, validation):
    for split in ("validation", "training"):
        split_values = validation.get(split)
        if not split_values:
            continue
        split_groups = groups.setdefault(split, {})
        for metric in ("mse", "bpp", "work", "rd", "overall"):
            candidate = {
                "epoch": int(epoch), "mse": float(split_values["mse"]),
                "bpp": float(split_values["bpp"]), "work": float(split_values["work"]),
                "overall_score": float(split_values["overall_score"]),
                "rd_score": float(split_values["rd_score"]),
                "speed_seconds": float(split_values["encode_seconds"]),
                "baseline_mse": float(split_values["baseline_mse"]),
                "baseline_bpp": float(split_values["baseline_bpp"]),
                "baseline_work": float(split_values["baseline_work"]),
                "baseline_speed_seconds": float(split_values["baseline_speed_seconds"]),
                "path": str(checkpoint_path(output, f"{split}_{metric}", epoch)),
            }
            entries = split_groups.setdefault(metric, [])
            if any(all(entry.get(key) == candidate[key] for key in ("mse", "bpp", "work")) for entry in entries):
                continue
            entries = [entry for entry in entries if entry["epoch"] != epoch]
            entries.append(candidate)
            if metric == "mse":
                entries.sort(key=lambda entry: entry[metric])
            elif metric == "bpp":
                entries.sort(key=lambda entry: entry[metric])
            elif metric == "work":
                entries.sort(key=lambda entry: entry[metric])
            elif metric == "rd":
                entries.sort(key=lambda entry: -entry["rd_score"])
            else:
                entries.sort(key=lambda entry: -entry["overall_score"])
            keep = entries[:1 if metric == "overall" else 2]
            keep_paths = {entry["path"] for entry in keep}
            for entry in entries:
                if entry["path"] not in keep_paths:
                    Path(entry["path"]).unlink(missing_ok=True)
            for entry in keep:
                path = Path(entry["path"])
                if not path.exists():
                    export(model, source, path)
            split_groups[metric] = keep
    if "validation" not in groups:
        legacy_groups = dict(groups)
        groups.clear()
        groups.update({"validation": legacy_groups, "training": {}})
    save_checkpoint_groups(model, source, output, groups)


def _speed_features(steps, accepted_patches, candidate_grid_coefficients, candidate_reconstruction_pixels):
    return np.array([
        1.0,
        float(steps),
        float(accepted_patches),
        float(candidate_grid_coefficients),
        float(candidate_reconstruction_pixels),
    ], dtype=np.float64)


def _fit_speed_model(rows):
    x = np.stack([row[0] for row in rows])
    y = np.asarray([row[1] for row in rows], dtype=np.float64)
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    coef[1:] = np.maximum(coef[1:], 0.0)
    predicted = np.maximum(x @ coef, 1e-6)
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - predicted) ** 2)) / max(ss_total, 1e-9)
    return {"features": ["intercept", "steps", "accepted_patches", "candidate_grid_coefficients", "candidate_reconstruction_pixels"], "coefficients": coef.tolist(), "r2": r2, "mae": float(np.mean(np.abs(y - predicted)))}


def _estimate_speed(features, speed_model):
    return max(1e-6, float(np.dot(features, np.asarray(speed_model["coefficients"], dtype=np.float64))))


def stream_result(prep, config, patches, canvas, base_values):
    channel_bits = max(1, math.ceil(math.log2(prep["channels"])))
    writer = BitWriter()
    PBC3._write_header(
        writer, prep["w"], prep["h"], prep["original_w"], prep["original_h"],
        prep["downsampled"], prep["color_id"], prep["channels"], channel_bits,
        config.positive_bias, prep["has_alpha"], len(patches), base_values,
    )
    for patch in patches:
        PBC3._write_patch(writer, patch, channel_bits)
    method, body = PBC3._entropy_pack(writer.finish(), config.use_lzma)
    data = PBC3.MAGIC + bytes([PBC3.VERSION, method]) + body
    image = PBC3._canvas_to_image(canvas, config.color_space, prep["has_alpha"])
    if image.size != (prep["original_w"], prep["original_h"]):
        image = image.resize(
            (prep["original_w"], prep["original_h"]), PBC3.RESAMPLE_FILTER,
            reducing_gap=PBC3.RESAMPLE_REDUCING_GAP,
        )
    mse = ops.final_mse(prep["orig_compare"], image)
    bpp = len(data) * 8 / (prep["original_w"] * prep["original_h"])
    return mse, bpp


def rollout(model, reference, source, path, preset, sample, temperature):
    config = PRESETS[preset]()
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    prep = PBC3.prepare(image, config)
    target = prep["target"]
    canvas, patches = init_canvas(prep, config)
    patches = list(patches)
    h, w, channels = target.shape
    channel_bits = max(1, math.ceil(math.log2(channels)))
    base_values = [int(round(float(np.mean(prep["arr"][:, :, c])))) for c in range(channels)]
    channel_scores = [PBC3._channel_sum_error(target, canvas, c) for c in range(channels)]
    rng = ops.PBC3Rng(config.random_seed)
    mean = source["feat_mean"].astype(np.float32)
    std = source["feat_std"].astype(np.float32)
    names = [str(x) for x in source["feature_names"]]
    cheap = bool(source["cheap"])
    logps, values, entropies, kls = [], [], [], []
    work = 0.0
    accepted_patches = 0
    candidate_grid_coefficients = 0
    candidate_reconstruction_pixels = 0
    started = time.perf_counter()
    for step in range(1, config.patch_count + 1):
        channel = PBC3._choose_channel(channel_scores, step, channels, config.channel_cycle)
        boxes = propose_boxes(target, canvas, config, rng, channel, step)[:1]
        if not boxes:
            break
        box = boxes[0]
        q = config.learned_filler_q
        if cheap:
            features = extract_cheap(names, target, canvas, box, step, q, w, h, config.patch_count, channels)
        else:
            features = extract_features(target, canvas, box, step, q, w, h, config.patch_count, channels)
        x = torch.from_numpy(((features - mean) / std)[None].astype(np.float32))
        logits, value = model(x)
        with torch.no_grad():
            ref_logits, _ = reference(x)
        if sample:
            distribution = torch.distributions.Categorical(logits=logits[0] / temperature)
            action = distribution.sample()
            logps.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            kls.append(F.kl_div(F.log_softmax(logits, 1), F.softmax(ref_logits, 1), reduction="batchmean"))
        else:
            action = logits[0].argmax()
        values.append(value[0])
        patch, patch_values, delta, reduction, _ = build_action(
            target, canvas, box, config, channel_bits, int(action),
        )
        candidate_grid_coefficients += int(patch_values.size)
        candidate_reconstruction_pixels += int(delta.size)
        if reduction <= 0:
            break
        work += float(patch_values.size)
        accepted_patches += 1
        c = patch["channel"]
        ops.apply_delta(canvas[:, :, c], patch["x"], patch["y"], patch["w"], patch["h"], delta)
        patches.append(patch)
        channel_scores[c] = PBC3._channel_sum_error(target, canvas, c)
    mse, bpp = stream_result(prep, config, patches, canvas, base_values)
    seconds = time.perf_counter() - started
    return {
        "mse": float(mse),
        "bpp": float(bpp),
        "work": float(work),
        "seconds": float(seconds),
        "speed_features": _speed_features(step if "step" in locals() else 0, accepted_patches, candidate_grid_coefficients, candidate_reconstruction_pixels),
        "logps": logps,
        "values": values,
        "entropies": entropies,
        "kls": kls,
    }


def reward(base, result, args, speed_model):
    return reward_terms(base, result, args, speed_model)["score"]


def reward_terms(base, result, args, speed_model):
    base_mse, base_bpp, base_work = base["mse"], base["bpp"], base["work"]
    mse, bpp, work = result["mse"], result["bpp"], result["work"]
    dmse = (base_mse - mse) / max(base_mse, 1.0)
    dbpp = (base_bpp - bpp) / max(base_bpp, 1e-6)
    base_speed = _estimate_speed(base["speed_features"], speed_model)
    result_speed = _estimate_speed(result["speed_features"], speed_model)
    dspeed = (base_speed - result_speed) / max(base_speed, 1e-6)
    quality_weight = args.quality_weight if dmse >= 0 else args.worse_quality_weight
    return {
        "score": float(np.clip(quality_weight * dmse + args.rate_weight * dbpp + args.speed_weight * dspeed, -5, 5)),
        "dmse": float(dmse), "dbpp": float(dbpp), "dspeed": float(dspeed),
    }


def summarize(rows, bases, args, speed_model):
    base_mean = {
        key: float(np.mean([bases[row["key"]][key] for row in rows]))
        for key in ("mse", "bpp", "work")
    }
    result_mean = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("mse", "bpp", "work", "seconds")
    }
    base_speed = float(np.mean([_estimate_speed(bases[row["key"]]["speed_features"], speed_model) for row in rows]))
    result_speed = float(np.mean([_estimate_speed(row["speed_features"], speed_model) for row in rows]))
    terms = reward_terms(
        {**base_mean, "speed_features": np.array([base_speed, 0, 0, 0, 0], dtype=np.float64)},
        {**result_mean, "speed_features": np.array([result_speed, 0, 0, 0, 0], dtype=np.float64)},
        args,
        {"coefficients": [1, 0, 0, 0, 0]},
    )
    rd_score = 0.5 * (terms["dmse"] + terms["dbpp"])
    overall_score = float(np.mean([terms["dmse"], terms["dbpp"], terms["dspeed"]]))
    return {
        "reward": float(np.mean([row["reward"] for row in rows])) if rows else 0.0,
        "aggregate_score": terms["score"],
        "overall_score": overall_score,
        "rd_score": float(rd_score),
        "mse": result_mean["mse"], "bpp": result_mean["bpp"], "work": result_mean["work"],
        "encode_seconds": result_mean["seconds"], "speed_cost": result_speed,
        "baseline_mse": base_mean["mse"], "baseline_bpp": base_mean["bpp"], "baseline_work": base_mean["work"],
        "baseline_speed_seconds": float(np.mean([bases[row["key"]]["seconds"] for row in rows])),
        "baseline_speed_cost": base_speed,
        "dmse": terms["dmse"], "dbpp": terms["dbpp"], "dspeed": terms["dspeed"],
        "presets": {
            preset: {
                "mse": float(np.mean([row["mse"] for row in rows if row["preset"] == preset])),
                "bpp": float(np.mean([row["bpp"] for row in rows if row["preset"] == preset])),
                "encode_seconds": float(np.mean([row["seconds"] for row in rows if row["preset"] == preset])),
            }
            for preset in args.selected_presets
        },
        "rows": [
            {"image": row["path"].name, "preset": row["preset"], "mse": row["mse"], "bpp": row["bpp"], "work": row["work"], "encode_seconds": row["seconds"], "speed_cost": _estimate_speed(row["speed_features"], speed_model), "reward": row["reward"]}
            for row in rows
        ],
    }


def evaluate(model, reference, source, episodes, bases, args, speed_model):
    rows = []
    model.eval()
    with torch.no_grad():
        for path, preset in episodes:
            result = rollout(model, reference, source, path, preset, False, 1.0)
            r = reward(bases[(path, preset)], result, args, speed_model)
            rows.append({"key": (path, preset), "path": path, "preset": preset, **result, "reward": r})
    return summarize(rows, bases, args, speed_model)


def train(args):
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.selected_presets = [preset.strip() for preset in args.presets.split(",")]
    unknown = [preset for preset in args.selected_presets if preset not in PRESETS]
    if unknown:
        raise ValueError(f"unknown presets: {', '.join(unknown)}")
    model, source = load_policy(args.init)
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    torch.nn.init.zeros_(model.value.weight)
    torch.nn.init.zeros_(model.value.bias)
    train_paths = image_paths(args.images)[:args.limit or None]
    val_paths = image_paths(args.val_images)[:args.val_limit or None]
    train_episodes = [(path, preset) for path in train_paths for preset in args.selected_presets]
    val_episodes = [(path, preset) for path in val_paths for preset in args.selected_presets]
    all_episodes = train_episodes + val_episodes
    bases = {}
    speed_rows = []
    print(f"building baselines for {len(all_episodes)} episodes", flush=True)
    for i, (path, preset) in enumerate(all_episodes, 1):
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        production = PBC3.compress(image, config=PRESETS[preset](learned_filler_model_path="patch_policy.npz"))
        reference_result = rollout(reference, reference, source, path, preset, False, 1.0)
        bases[(path, preset)] = {
            "mse": reference_result["mse"],
            "bpp": reference_result["bpp"],
            "work": reference_result["work"],
            "seconds": reference_result["seconds"],
            "speed_features": reference_result["speed_features"],
            "production_mse": float(production.mse),
            "production_bpp": len(production.data) * 8 / (image.width * image.height),
            "production_seconds": float(production.encode_seconds),
        }
        speed_rows.append((reference_result["speed_features"], reference_result["seconds"]))
        if i % 10 == 0:
            print(f"baseline {i}/{len(all_episodes)}", flush=True)
    speed_model = _fit_speed_model(speed_rows)
    print("speed model", json.dumps(speed_model), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    start_epoch = 0
    checkpoint_groups = {"validation": {}, "training": {}}
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.random.set_rng_state(checkpoint["torch_rng"])
        start_epoch = int(checkpoint["epoch"])
        history = checkpoint.get("history", [])
        manifest = checkpoint_manifest_path(args.out)
        if manifest.exists():
            checkpoint_groups = json.loads(manifest.read_text(encoding="utf-8")).get("groups", checkpoint_groups)
    initial = evaluate(model, reference, source, val_episodes, bases, args, speed_model)
    initial_payload = {"validation": initial}
    print("initial", json.dumps(initial_payload), flush=True)
    best = initial["aggregate_score"]
    best_epoch = 0
    if not any(checkpoint_groups.values()):
        update_checkpoint_groups(model, source, args.out, checkpoint_groups, 0, initial_payload)
    train_baseline = {
        key: float(np.mean([bases[(path, preset)][key] for path, preset in train_episodes]))
        for key in ("mse", "bpp", "work", "seconds")
    }
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        random.shuffle(train_episodes)
        rewards = []
        losses = []
        train_rows = []
        for start in range(0, len(train_episodes), args.batch):
            optimizer.zero_grad()
            episode_losses = []
            for path, preset in train_episodes[start:start + args.batch]:
                result = rollout(
                    model, reference, source, path, preset, True, args.temperature,
                )
                r = reward(bases[(path, preset)], result, args, speed_model)
                rewards.append(r)
                train_rows.append({"key": (path, preset), "path": path, "preset": preset, **result, "reward": r})
                if not result["logps"]:
                    continue
                logps = torch.stack(result["logps"])
                values = torch.stack(result["values"])
                target = torch.full_like(values, r)
                advantage = (target - values).detach()
                loss = -(logps * advantage).mean()
                loss += args.value_weight * F.mse_loss(values, target)
                loss -= args.entropy_weight * torch.stack(result["entropies"]).mean()
                loss += args.kl_weight * torch.stack(result["kls"]).mean()
                episode_losses.append(loss)
            if episode_losses:
                loss = torch.stack(episode_losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach()))
        validation = evaluate(model, reference, source, val_episodes, bases, args, speed_model)
        training = summarize(train_rows, bases, args, speed_model)
        row = {
            "epoch": epoch,
            "train_reward": float(np.mean(rewards)),
            "train_mse": training["mse"],
            "train_bpp": training["bpp"],
            "train_work": training["work"],
            "train_encode_seconds": training["encode_seconds"],
            "train_speed_cost": training["speed_cost"],
            "train_baseline_mse": train_baseline["mse"],
            "train_baseline_bpp": train_baseline["bpp"],
            "train_baseline_work": train_baseline["work"],
            "train_baseline_speed_seconds": train_baseline["seconds"],
            "train_rd_score": training["rd_score"],
            "train_overall_score": training["overall_score"],
            "loss": float(np.mean(losses)),
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["aggregate_score"] > best:
            best = validation["aggregate_score"]
            best_epoch = epoch
        update_checkpoint_groups(model, source, args.out, checkpoint_groups, epoch, {"validation": validation, "training": training})
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
            "history": history,
        }, f"{args.out}.resume.pt")
    Path(f"{args.out}.json").write_text(json.dumps({"initial": initial_payload, "speed_model": speed_model, "best_score": best, "best_epoch": best_epoch, "history": history}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="hpt_data")
    parser.add_argument("--val-images", default="hpt_data_val")
    parser.add_argument("--presets", default=",".join(PRESETS))
    parser.add_argument("--init", default="pbc3_students/patch_policy_f26_a20_h512_l2_e1200.npz")
    parser.add_argument("--out", default="pbc3_students/patch_policy_f26_a20_rl.npz")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--quality-weight", type=float, default=2.0)
    parser.add_argument("--worse-quality-weight", type=float, default=6.0)
    parser.add_argument("--rate-weight", type=float, default=1.0)
    parser.add_argument("--speed-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2003)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
