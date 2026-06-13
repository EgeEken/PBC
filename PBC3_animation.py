import os
import numpy as np
from PIL import Image, ImageDraw
from PBC3 import PBC3, BitReader


def _colorize_channel(channel, c):
    arr = np.clip(channel, 0, 255).astype(np.uint8)
    out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    out[:, :, c % 3] = arr
    return Image.fromarray(out, "RGB")


def _fit_panel(img, panel_w, panel_h):
    scale = min(panel_w / img.width, panel_h / img.height)
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(size, Image.Resampling.NEAREST)


def _draw_patch(draw, box, scale, offset, color="red", width=3):
    if box is None:
        return
    x, y, w, h = box
    ox, oy = offset
    rect = [ox + x * scale, oy + y * scale, ox + (x + w) * scale, oy + (y + h) * scale]
    for i in range(width):
        draw.rectangle([rect[0] - i, rect[1] - i, rect[2] + i, rect[3] + i], outline=color)


def _make_frame(canvas, color_space, patch_info, separated_channels, title, scale=2):
    arr = np.clip(canvas, 0, 255).astype(np.uint8)
    rgb = Image.fromarray(arr, color_space).convert("RGB")

    if not separated_channels:
        img = rgb.resize((rgb.width * scale, rgb.height * scale), Image.Resampling.NEAREST)
        frame = Image.new("RGB", (img.width, img.height + 34), "white")
        frame.paste(img, (0, 34))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 8), title, fill="black")
        if patch_info is not None:
            _draw_patch(draw, patch_info[1:5], scale, (0, 34))
        return frame

    h, w, _ = arr.shape
    panels = [
        (_colorize_channel(arr[:, :, 0], 0), "R"),
        (_colorize_channel(arr[:, :, 1], 1), "G"),
        (_colorize_channel(arr[:, :, 2], 2), "B"),
        (rgb, "RGB"),
    ]
    panel_w, panel_h = w * scale, h * scale
    title_h = 54
    label_h = 24
    gap = 12
    frame_w = panel_w * 4 + gap * 3
    frame_h = title_h + label_h + panel_h
    frame = Image.new("RGB", (frame_w, frame_h), "white")
    draw = ImageDraw.Draw(frame)
    draw.text((8, 8), title, fill="black")

    active_channel = patch_info[0] if patch_info is not None else None
    patch_box = patch_info[1:5] if patch_info is not None else None
    for i, (panel, label) in enumerate(panels):
        panel = panel.resize((panel_w, panel_h), Image.Resampling.NEAREST)
        x0 = i * (panel_w + gap)
        y0 = title_h + label_h
        draw.text((x0 + 4, title_h), label, fill="black")
        frame.paste(panel, (x0, y0))
        if patch_box is not None and (i == 3 or i == active_channel):
            _draw_patch(draw, patch_box, scale, (x0, y0))
    return frame


def _write_frames(frames, output_path, fps):
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        duration = int(1000 / fps)
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        return output_path
    try:
        import imageio.v2 as imageio
        imageio.mimsave(output_path, [np.asarray(f) for f in frames], fps=fps)
        return output_path
    except Exception:
        fallback = os.path.splitext(output_path)[0] + ".gif"
        duration = int(1000 / fps)
        frames[0].save(fallback, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        return fallback


def animate_pbc3(data, output_path="pbc3_animation.mp4", fps=3, separated_channels=True, scale=2, max_patches=None):
    if isinstance(data, str):
        with open(data, "rb") as f:
            data = f.read()
    if data[:4] != PBC3.MAGIC:
        raise ValueError("not a PBC3 file")

    version = data[4]
    br = BitReader(data[5:])
    downsampled, original_w, original_h, w, h, color_space, channels, channel_bits, positive_bias, patch_count, base_values = PBC3._read_header(br, version)
    if channels != 3 and separated_channels:
        separated_channels = False

    canvas = np.zeros((h, w, channels), dtype=np.int32)
    for c, base in enumerate(base_values):
        canvas[:, :, c] = base

    limit = patch_count if max_patches is None else min(int(max_patches), patch_count)
    frames = [_make_frame(canvas, color_space, None, separated_channels, "Patch 0 | average-color canvas", scale)]

    for i in range(1, limit + 1):
        channel, x, y, pw, ph, cell_size, values, mode = PBC3._read_patch(br, channel_bits, positive_bias)
        if mode != PBC3.MODE_RAW:
            raise ValueError(f"unsupported patch mode {mode}")
        PBC3.apply_grid(canvas[:, :, channel], x, y, pw, ph, cell_size, values)
        title = f"Patch {i}/{patch_count} | channel={channel} | box=({x},{y},{pw},{ph}) | cell={cell_size}"
        frames.append(_make_frame(canvas, color_space, (channel, x, y, pw, ph, cell_size), separated_channels, title, scale))

    return _write_frames(frames, output_path, fps)
