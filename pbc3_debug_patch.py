import re
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def parse_debug_line(line):
    parts = line.strip().split()
    if not parts:
        return None
    out = {"kind": parts[0]}
    for part in parts[1:]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _channel_rgb(channel_2d, channel, normalize=False):
    arr = np.asarray(channel_2d, dtype=np.float32)
    if normalize:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        arr = (arr - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(arr)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[:, :, int(channel) % 3] = arr
    return out


def _format_debug_title(d):
    shown = dict(d)
    if "neg_max" in shown and shown["neg_max"] >= 0:
        shown["neg_min"] = -int(shown["neg_max"])
        del shown["neg_max"]
    return " | ".join(f"{k}: {v}" for k, v in shown.items() if k != "kind")


def show_debug_line_channel(PBC3, line, original_image=None, data=None, normalize=False):
    d = parse_debug_line(line)
    if not d:
        raise ValueError("could not parse debug line")
    if not all(k in d for k in ("x", "y", "w", "h", "channel")):
        raise ValueError("debug line needs x/y/w/h/channel")

    channel = int(d["channel"])
    images = []
    titles = []
    canvas_before = None
    color_space = "RGB"
    downsampled = False
    original_w = original_h = None
    working_w = working_h = None

    if data is not None and "canvas_patches" in d:
        canvas, color_space, downsampled, original_w, original_h, working_w, working_h, _ = PBC3._decode_to_canvas(
            data, max_patches=int(d["canvas_patches"])
        )
        canvas_before = np.clip(canvas, 0, 255).astype(np.uint8)

    if original_image is not None:
        img = PBC3._to_image(original_image).convert(color_space)
        if working_w is not None and (img.width != working_w or img.height != working_h):
            img = img.resize((working_w, working_h), PBC3.RESAMPLE_FILTER, reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
        original_arr = np.asarray(img, dtype=np.uint8)
        images.append(_channel_rgb(original_arr[:, :, channel], channel, normalize))
        titles.append(f"Original channel {channel}")

    if canvas_before is not None:
        images.append(_channel_rgb(canvas_before[:, :, channel], channel, normalize))
        titles.append(f"Canvas channel {channel} after {d['canvas_patches']} patches")

    if canvas_before is not None and original_image is not None and d["kind"] == "CANDIDATE":
        img = PBC3._to_image(original_image).convert(color_space)
        if img.width != canvas_before.shape[1] or img.height != canvas_before.shape[0]:
            img = img.resize((canvas_before.shape[1], canvas_before.shape[0]), PBC3.RESAMPLE_FILTER, reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
        target = np.asarray(img, dtype=np.uint8).astype(np.int32)
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
        hidden = target[y:y + h, x:x + w, channel] - canvas_before[y:y + h, x:x + w, channel].astype(np.int32)
        patch, values = PBC3._make_patch(
            channel, x, y, w, h,
            int(d["cell_size"]), hidden,
            int(d.get("mask_size", 9)), int(d.get("bitcount", 3)), True,
        )
        after = canvas_before.astype(np.int32)
        PBC3.apply_grid(after[:, :, channel], x, y, w, h, int(d["cell_size"]), values)
        after = np.clip(after, 0, 255).astype(np.uint8)
        images.append(_channel_rgb(after[:, :, channel], channel, normalize))
        titles.append(f"Candidate applied to channel {channel}")

    if not images:
        raise ValueError("provide original_image and/or data")

    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5), dpi=130)
    if len(images) == 1:
        axes = [axes]

    x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=3))
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1.5))

    fig.suptitle(_format_debug_title(d), fontsize=9)
    plt.tight_layout()
    plt.show()


def install_debug_patch(PBC3):
    PBC3.parse_debug_line = staticmethod(parse_debug_line)
    PBC3.show_debug_line = classmethod(lambda cls, line, original_image=None, data=None, normalize=False: show_debug_line_channel(cls, line, original_image, data, normalize))
    return PBC3