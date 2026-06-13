import os
import numpy as np
from PIL import Image, ImageDraw
from PBC3 import PBC3, BitReader


def _colorize_channel(channel, c):
    arr = np.clip(channel, 0, 255).astype(np.uint8)
    out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    out[:, :, c % 3] = arr
    return Image.fromarray(out, "RGB")


def _error_image(error, channel=None):
    err = np.asarray(error, dtype=np.float32)
    if err.ndim == 3:
        err = np.mean(np.abs(err), axis=2)
    else:
        err = np.abs(err)
    m = float(np.max(err))
    err = (err * (255.0 / m)) if m > 0 else err
    arr = np.clip(err, 0, 255).astype(np.uint8)
    if channel is None:
        return Image.fromarray(np.stack([arr, arr, arr], axis=2), "RGB")
    return _colorize_channel(arr, channel)


def _draw_patch(draw, box, scale, offset, color="red", width=3):
    if box is None:
        return
    x, y, w, h = box
    ox, oy = offset
    rect = [ox + x * scale, oy + y * scale, ox + (x + w) * scale, oy + (y + h) * scale]
    for i in range(width):
        draw.rectangle([rect[0] - i, rect[1] - i, rect[2] + i, rect[3] + i], outline=color)


def _make_frame(canvas, color_space, patch_info, separated_channels, title, scale=2, target=None, show_errors=False):
    arr = np.clip(canvas, 0, 255).astype(np.uint8)
    rgb = Image.fromarray(arr, color_space).convert("RGB")
    error = None if target is None else target.astype(np.int32) - arr.astype(np.int32)

    if not separated_channels:
        panels = [(rgb, "RGB")]
        if show_errors:
            if error is None:
                raise ValueError("show_errors=True requires original_image")
            panels.append((_error_image(error), "RGB error mean(|delta|)"))
        cols, rows = (1, len(panels))
    else:
        panels = [
            (_colorize_channel(arr[:, :, 0], 0), "R"),
            (_colorize_channel(arr[:, :, 1], 1), "G"),
            (_colorize_channel(arr[:, :, 2], 2), "B"),
            (rgb, "RGB"),
        ]
        if show_errors:
            if error is None:
                raise ValueError("show_errors=True requires original_image")
            panels.extend([
                (_error_image(error[:, :, 0], 0), "R error"),
                (_error_image(error[:, :, 1], 1), "G error"),
                (_error_image(error[:, :, 2], 2), "B error"),
                (_error_image(error), "RGB error mean"),
            ])
        cols, rows = 4, 2 if show_errors else 1

    h, w = arr.shape[:2]
    panel_w, panel_h = w * scale, h * scale
    title_h = 54
    label_h = 24
    gap = 12
    row_gap = 14
    frame_w = panel_w * cols + gap * (cols - 1)
    frame_h = title_h + rows * (label_h + panel_h) + row_gap * (rows - 1)
    frame = Image.new("RGB", (frame_w, frame_h), "white")
    draw = ImageDraw.Draw(frame)
    draw.text((8, 8), title, fill="black")

    active_channel = patch_info[0] if patch_info is not None else None
    patch_box = patch_info[1:5] if patch_info is not None else None
    for i, (panel, label) in enumerate(panels):
        col = i % cols
        row = i // cols
        x0 = col * (panel_w + gap)
        y_label = title_h + row * (label_h + panel_h + row_gap)
        y0 = y_label + label_h
        panel = panel.resize((panel_w, panel_h), Image.Resampling.NEAREST)
        draw.text((x0 + 4, y_label), label, fill="black")
        frame.paste(panel, (x0, y0))
        is_rgb_panel = (separated_channels and col == 3) or (not separated_channels and label.startswith("RGB"))
        is_active_channel_panel = separated_channels and col == active_channel
        if patch_box is not None and (is_rgb_panel or is_active_channel_panel):
            _draw_patch(draw, patch_box, scale, (x0, y0))
    return frame


def _even_rgb_array(frame):
    arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=255)
    return arr


def _write_mp4(frames, output_path, fps):
    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError("imageio is not importable. Try: pip install imageio imageio-ffmpeg") from e
    try:
        import imageio_ffmpeg  # noqa: F401
    except Exception as e:
        raise RuntimeError("MP4 writing needs ffmpeg. Try: pip install imageio-ffmpeg") from e

    arrays = [_even_rgb_array(frame) for frame in frames]
    h, w = arrays[0].shape[:2]
    arrays = [arr if arr.shape[:2] == (h, w) else np.asarray(Image.fromarray(arr).resize((w, h))) for arr in arrays]
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        for arr in arrays:
            writer.append_data(arr)
    finally:
        writer.close()
    return output_path


def _write_gif(frames, output_path, fps):
    duration = int(1000 / fps)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    return output_path


def _write_frames(frames, output_path, fps, fallback_to_gif=True):
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        return _write_gif(frames, output_path, fps)
    if ext not in {".mp4", ".m4v", ".mov"}:
        output_path = os.path.splitext(output_path)[0] + ".mp4"
    try:
        return _write_mp4(frames, output_path, fps)
    except Exception as e:
        if not fallback_to_gif:
            raise
        fallback = os.path.splitext(output_path)[0] + ".gif"
        print(f"MP4 export failed: {e}")
        print(f"Falling back to GIF: {fallback}")
        return _write_gif(frames, fallback, fps)


def _working_target(original_image, color_space, working_w, working_h):
    img = PBC3._to_image(original_image).convert(color_space)
    if img.width != working_w or img.height != working_h:
        img = img.resize((working_w, working_h), PBC3.RESAMPLE_FILTER, reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
    return np.asarray(img, dtype=np.uint8)


def animate_pbc3(
    data,
    output_path="pbc3_animation.mp4",
    fps=3,
    separated_channels=True,
    scale=2,
    max_patches=None,
    fallback_to_gif=True,
    show_errors=False,
    original_image=None,
):
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
    if show_errors and original_image is None:
        raise ValueError("show_errors=True requires original_image=...")

    target = _working_target(original_image, color_space, w, h) if show_errors else None
    canvas = np.zeros((h, w, channels), dtype=np.int32)
    for c, base in enumerate(base_values):
        canvas[:, :, c] = base

    limit = patch_count if max_patches is None else min(int(max_patches), patch_count)
    frames = [_make_frame(canvas, color_space, None, separated_channels, "Patch 0 | average-color canvas", scale, target, show_errors)]

    for i in range(1, limit + 1):
        channel, x, y, pw, ph, cell_size, values, mode = PBC3._read_patch(br, channel_bits, positive_bias)
        if mode != PBC3.MODE_RAW:
            raise ValueError(f"unsupported patch mode {mode}")
        PBC3.apply_grid(canvas[:, :, channel], x, y, pw, ph, cell_size, values)
        title = f"Patch {i}/{patch_count} | channel={channel} | box=({x},{y},{pw},{ph}) | cell={cell_size}"
        frames.append(_make_frame(canvas, color_space, (channel, x, y, pw, ph, cell_size), separated_channels, title, scale, target, show_errors))

    return _write_frames(frames, output_path, fps, fallback_to_gif=fallback_to_gif)