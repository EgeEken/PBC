import json
import pathlib
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from run_feasibility import BASE, OUT, VIS, ROOT, IMAGES, SSIMULACRA2, apply_stroke, stroke_pixels


def build_stroke(geometry, shape):
    stroke = dict(geometry)
    yy, xx, coverage = stroke_pixels(stroke, stroke["width"], shape)
    stroke.update(yy=yy, xx=xx, coverage=coverage)
    return stroke


def score(source_path, decoded_path):
    p = subprocess.run([str(SSIMULACRA2), str(source_path), str(decoded_path)], capture_output=True, text=True, check=True)
    return float(p.stdout.strip().splitlines()[-1])


def crop4(name, source, base, one, group, geometry):
    x0, y0, x1, y1 = (geometry[k] for k in ("x0", "y0", "x1", "y1"))
    cx, cy = int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))
    crop_w, crop_h = 384, 256
    left = min(max(0, cx - crop_w // 2), max(0, source.shape[1] - crop_w))
    top = min(max(0, cy - crop_h // 2), max(0, source.shape[0] - crop_h))
    right, bottom = min(source.shape[1], left + crop_w), min(source.shape[0], top + crop_h)
    labels = ["Source", "Canonical V3.1 base", "Single-stroke correction", "Eight-stroke local group"]
    images = [source, base, one, group]
    composite = Image.new("RGB", (crop_w * 4, (crop_h + 24)), "white")
    draw = ImageDraw.Draw(composite)
    for i, (array, label) in enumerate(zip(images, labels)):
        crop = Image.fromarray(array[top:bottom, left:right])
        composite.paste(crop, (i * crop_w, 24))
        draw.text((i * crop_w + 4, 4), label, fill="black")
    geometry_panel = Image.fromarray(base[top:bottom, left:right]).copy()
    draw_geo = ImageDraw.Draw(geometry_panel)
    draw_geo.line((x0-left, y0-top, x1-left, y1-top), fill=(255, 0, 0), width=geometry["width"])
    geometry_sheet = Image.new("RGB", (crop_w, crop_h + 24), "white")
    geometry_sheet.paste(geometry_panel, (0, 24))
    ImageDraw.Draw(geometry_sheet).text((4, 4), "Top stroke geometry in base image", fill="black")
    composite.save(VIS / f"{pathlib.Path(name).stem}_local_comparison.png")
    geometry_sheet.save(VIS / f"{pathlib.Path(name).stem}_local_geometry.png")


def main():
    VIS.mkdir(parents=True, exist_ok=True)
    rows = []
    data = json.loads((OUT / "feasibility_results.json").read_text())
    for row in data["images"]:
        name = row["image"]
        if not row["candidate_results"]:
            continue
        stem = pathlib.Path(name).stem
        source_path = BASE / f"{stem}.png"
        source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
        base = np.asarray(Image.open(BASE / f"{stem}_pbc_compression.png").convert("RGB"), dtype=np.uint8)
        selected = [build_stroke(c["geometry"], source.shape[:2]) for c in row["candidate_results"]]
        top1 = apply_stroke(base, selected[0])
        group = base
        for stroke in selected:
            group = apply_stroke(group, stroke)
        one_path = BASE / f"{stem}_stroke_top1.png"
        group_path = BASE / f"{stem}_stroke_group8.png"
        Image.fromarray(top1).save(one_path)
        Image.fromarray(group).save(group_path)
        group_score = score(source_path, group_path)
        crop4(name, source, base, top1, group, selected[0])
        rows.append({"image": name, "single_stroke_ssimulacra2": score(source_path, one_path), "eight_stroke_group_ssimulacra2": group_score, "eight_stroke_group_decoded_rgb_sha256": __import__("hashlib").sha256(group.tobytes()).hexdigest()})
    (OUT / "visual_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
