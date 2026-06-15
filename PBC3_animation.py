import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PBC3 import PBC3, BitReader


def _font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _colorize_rgb_channel(channel, c):
    arr = np.clip(channel, 0, 255).astype(np.uint8)
    out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    out[:, :, c % 3] = arr
    return Image.fromarray(out, "RGB")


def _colorize_ycbcr_channel(channel, c):
    arr = np.clip(channel, 0, 255).astype(np.uint8)
    if c == 0:
        return Image.fromarray(np.stack([arr, arr, arr], axis=2), "RGB")

    d = arr.astype(np.int16) - 128
    mag = np.clip(np.abs(d) * 2, 0, 255).astype(np.uint8)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    if c == 1:
        pos = d >= 0
        out[pos, 2] = mag[pos]
        out[~pos, 0] = mag[~pos]
        out[~pos, 1] = mag[~pos]
    else:
        pos = d >= 0
        out[pos, 0] = mag[pos]
        out[~pos, 1] = mag[~pos]
        out[~pos, 2] = mag[~pos]
    return Image.fromarray(out, "RGB")


def _colorize_channel(channel, c, color_space):
    if str(color_space).lower() == "ycbcr":
        return _colorize_ycbcr_channel(channel, c)
    return _colorize_rgb_channel(channel, c)


def _error_image(error, color_space, channel=None):
    err = np.asarray(error, dtype=np.float32)
    if err.ndim == 3:
        err = np.mean(np.abs(err), axis=2)
        m = float(np.max(err))
        arr = np.clip(err * (255.0 / m), 0, 255).astype(np.uint8) if m > 0 else np.zeros(err.shape, dtype=np.uint8)
        return Image.fromarray(np.stack([arr, arr, arr], axis=2), "RGB")

    signed = err
    mag = np.abs(signed)
    m = float(np.max(mag))
    arr = np.clip(mag * (255.0 / m), 0, 255).astype(np.uint8) if m > 0 else np.zeros(mag.shape, dtype=np.uint8)
    if str(color_space).lower() != "ycbcr" or channel == 0:
        return _colorize_channel(arr, 0 if channel is None else channel, color_space)

    d = np.sign(signed).astype(np.int16) * arr.astype(np.int16)
    if channel == 1:
        out = np.zeros((*arr.shape, 3), dtype=np.uint8)
        pos = d >= 0
        out[pos, 2] = arr[pos]
        out[~pos, 0] = arr[~pos]
        out[~pos, 1] = arr[~pos]
        return Image.fromarray(out, "RGB")

    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    pos = d >= 0
    out[pos, 0] = arr[pos]
    out[~pos, 1] = arr[~pos]
    out[~pos, 2] = arr[~pos]
    return Image.fromarray(out, "RGB")


def _draw_patch(draw, box, scale, offset, color="red", width=4):
    if box is None:
        return
    x, y, w, h = box
    ox, oy = offset
    rect = [ox + x * scale, oy + y * scale, ox + (x + w) * scale, oy + (y + h) * scale]
    for i in range(width):
        draw.rectangle([rect[0] - i, rect[1] - i, rect[2] + i, rect[3] + i], outline=color)


def _fit(img, max_w, max_h):
    scale = min(max_w / img.width, max_h / img.height)
    out_w = max(1, int(img.width * scale))
    out_h = max(1, int(img.height * scale))
    return img.resize((out_w, out_h), Image.Resampling.NEAREST), scale


def _make_frame(canvas, color_space, patch_info, separated_channels, title, target=None, show_errors=False, output_size=(3840, 2160)):
    arr = np.clip(canvas, 0, 255).astype(np.uint8)
    rgb = Image.fromarray(arr, color_space).convert("RGB")
    error = None if target is None else target.astype(np.int32) - arr.astype(np.int32)

    if not separated_channels:
        panels = [(rgb, False)]
        if show_errors:
            if error is None:
                raise ValueError("show_errors=True requires original_image")
            panels.append((_error_image(error, color_space), True))
        cols, rows = 1, len(panels)
    else:
        panels = [
            (_colorize_channel(arr[:, :, 0], 0, color_space), False),
            (_colorize_channel(arr[:, :, 1], 1, color_space), False),
            (_colorize_channel(arr[:, :, 2], 2, color_space), False),
            (rgb, False),
        ]
        if show_errors:
            if error is None:
                raise ValueError("show_errors=True requires original_image")
            panels.extend([
                (_error_image(error[:, :, 0], color_space, 0), True),
                (_error_image(error[:, :, 1], color_space, 1), True),
                (_error_image(error[:, :, 2], color_space, 2), True),
                (_error_image(error, color_space), True),
            ])
        cols, rows = 4, 2 if show_errors else 1

    frame_w, frame_h = output_size
    frame = Image.new("RGB", output_size, "black")
    draw = ImageDraw.Draw(frame)
    title_h = 120
    gap = 18
    margin = 36
    draw.text((margin, 34), title, fill="white", font=_font(46))

    area_w = frame_w - margin * 2
    area_h = frame_h - title_h - margin
    cell_w = (area_w - gap * (cols - 1)) // cols
    cell_h = (area_h - gap * (rows - 1)) // rows
    active_channel = patch_info[0] if patch_info is not None else None
    patch_box = patch_info[1:5] if patch_info is not None else None

    for i, (panel, is_error) in enumerate(panels):
        col = i % cols
        row = i // cols
        x0 = margin + col * (cell_w + gap)
        y0 = title_h + row * (cell_h + gap)
        fitted, scale = _fit(panel, cell_w, cell_h)
        px = x0 + (cell_w - fitted.width) // 2
        py = y0 + (cell_h - fitted.height) // 2
        frame.paste(fitted, (px, py))

        is_rgb_panel = (separated_channels and col == 3) or (not separated_channels and i == 0)
        is_active_channel_panel = separated_channels and col == active_channel
        if patch_box is not None and not is_error and (is_rgb_panel or is_active_channel_panel):
            _draw_patch(draw, patch_box, scale, (px, py))

    return frame


def _even_rgb_array(frame):
    arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
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

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        for frame in frames:
            writer.append_data(_even_rgb_array(frame))
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
    max_patches=None,
    fallback_to_gif=True,
    show_errors=False,
    original_image=None,
    output_size=(3840, 2160),
):
    if isinstance(data, str):
        with open(data, "rb") as f:
            data = f.read()
    version, body = PBC3._open_body(data)
    br = BitReader(body)
    downsampled, original_w, original_h, w, h, color_space, channels, channel_bits, positive_bias, patch_count, base_values = PBC3._read_header(br, version)
    if channels != 3 and separated_channels:
        separated_channels = False
    if show_errors and original_image is None:
        raise ValueError("show_errors=True requires original_image=...")

    target = _working_target(original_image, color_space, w, h) if show_errors else None
    canvas = np.zeros((h, w, channels), dtype=np.int32)
    for c, base in enumerate(base_values):
        canvas[:, :, c] = base

    frames = []
    limit = patch_count if max_patches is None else min(int(max_patches), patch_count)

    # Note: entropy coding is global, so per-patch byte sizes are reported on the
    # pre-entropy (uncompressed) stream; the final file is smaller after packing.
    current_bytes = br.i
    current_kb = current_bytes / 1024
    frames.append(_make_frame(
        canvas,
        color_space,
        None,
        separated_channels,
        f"Patch 0/{patch_count} | Stream Size: {current_kb:.2f} KB | (+0.00 KB)",
        target,
        show_errors,
        output_size,
    ))

    for i in range(1, limit + 1):
        previous_bytes = current_bytes

        channel, x, y, pw, ph, cell_size, values, mode = PBC3._read_patch(br, channel_bits, positive_bias)

        current_bytes = br.i
        current_kb = current_bytes / 1024
        delta_kb = (current_bytes - previous_bytes) / 1024

        PBC3.apply_grid(canvas[:, :, channel], x, y, pw, ph, cell_size, values)

        mode_name = {PBC3.MODE_RAW: "raw", PBC3.MODE_ZERO_RUN: "zero-run", PBC3.MODE_RLE: "rle"}.get(mode, str(mode))
        title = (
            f"Patch {i}/{patch_count} | Stream Size: {current_kb:.2f} KB "
            f"| (+{delta_kb:.2f} KB) | ch={channel} box=({x},{y},{pw},{ph}) cell={cell_size} grid={mode_name}"
        )
        frames.append(_make_frame(
            canvas,
            color_space,
            (channel, x, y, pw, ph, cell_size),
            separated_channels,
            title,
            target,
            show_errors,
            output_size,
        ))

    return _write_frames(frames, output_path, fps, fallback_to_gif=fallback_to_gif)