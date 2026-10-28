"""Synthetic generation, grouped splits, YOLO labels and COCO conversion."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def alpha_bbox(image: Image.Image, threshold: int = 10):
    """Return a half-open bbox of nontransparent pixels, or None."""
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    ys, xs = np.where(alpha > threshold)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _class_objects(root: Path, classes: list[str]):
    objects = []
    for class_id, name in enumerate(classes):
        folder = root / name
        files = sorted(p for p in folder.glob("*.png") if p.is_file()) if folder.is_dir() else []
        # The legacy flat alvo/ layout maps all PNGs to class 0.
        if len(classes) == 1 and not files:
            files = sorted(p for p in root.glob("*.png") if p.is_file())
        objects.extend((p, class_id) for p in files)
    return objects


def generate(background_dir: Path, object_dir: Path, output: Path, *, classes: list[str],
             count: int, train_fraction: float, negative_fraction: float,
             seed: int, overwrite: bool = False, min_scale: float = .08,
             max_scale: float = .6, min_objects: int = 1, max_objects: int = 4,
             max_rotation_degrees: int = 40, placement: str = "random",
             allow_overlap: bool = True):
    """Generate a YOLO dataset, assigning whole background sources to one split."""
    if count < 1 or not 0 < train_fraction < 1 or not 0 <= negative_fraction < 1:
        raise ValueError("count >= 1, train_fraction in (0,1), negative_fraction in [0,1) required")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be a nonempty list of unique names")
    if not 0 < min_scale < max_scale or min_objects < 1 or max_objects < min_objects:
        raise ValueError("Invalid scale or object-count range")
    if max_rotation_degrees < 0 or placement not in {"random", "center"}:
        raise ValueError("Invalid rotation or placement mode")
    backgrounds = sorted(p for p in background_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    objects = _class_objects(object_dir, classes)
    if not backgrounds or not objects:
        raise ValueError("Need background images and transparent PNG objects")
    if len({class_id for _, class_id in objects}) != len(classes):
        raise ValueError("Every configured class must have at least one PNG in its class directory")
    if len(backgrounds) < 2:
        raise ValueError("At least two distinct backgrounds are required for train/validation separation")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Output is not empty: {output}; choose another output or pass --overwrite")
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)
    # Split by original background file so derivatives cannot leak between partitions.
    shuffled = backgrounds[:]
    rng.shuffle(shuffled)
    cut = min(max(1, round(len(shuffled) * train_fraction)), len(shuffled) - 1) if len(shuffled) > 1 else 1
    split_by_bg = {p: ("train" if i < cut else "val") for i, p in enumerate(shuffled)}
    for split in ("train", "val"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)
    metadata = {"seed": seed, "background_split": {p.name: split_by_bg[p] for p in backgrounds}}
    for i in range(count):
        bg_path = rng.choice(backgrounds)
        split = split_by_bg[bg_path]
        try:
            background = Image.open(bg_path).convert("RGB")
        except (OSError, ValueError):
            continue
        bw, bh = background.size
        labels = []
        # Keep masks to compute the visible extent after later objects occlude earlier ones.
        visible_masks = []
        coverage = np.zeros((bh, bw), dtype=np.float32)
        if rng.random() >= negative_fraction:
            for _ in range(rng.randint(min_objects, max_objects)):
                obj_path, class_id = rng.choice(objects)
                try:
                    obj = Image.open(obj_path).convert("RGBA")
                except (OSError, ValueError):
                    continue
                ow, oh = obj.size
                scale_max = min(bw / ow, bh / oh, max_scale)
                if scale_max <= min_scale:
                    continue
                scale = rng.uniform(min_scale, scale_max)
                obj = obj.resize((max(1, int(ow * scale)), max(1, int(oh * scale))), Image.Resampling.LANCZOS)
                obj = obj.rotate(rng.randint(-max_rotation_degrees, max_rotation_degrees), expand=True, resample=Image.Resampling.BICUBIC)
                if rng.random() < .5: obj = ImageEnhance.Brightness(obj).enhance(rng.uniform(.7, 1.4))
                if rng.random() < .4: obj = ImageEnhance.Contrast(obj).enhance(rng.uniform(.7, 1.4))
                if rng.random() < .3: obj = obj.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.5)))
                box = alpha_bbox(obj)
                if box is None: continue
                x0, y0, x1, y1 = box
                if x1 - x0 < 2 or y1 - y0 < 2: continue
                # Fit full transformed object to image; no bbox clipping is needed.
                if obj.width > bw or obj.height > bh: continue
                alpha = np.asarray(obj.getchannel("A"), dtype=np.float32) / 255.0
                x = y = 0
                accepted = False
                for _attempt in range(50):
                    if placement == "center":
                        candidate_x, candidate_y = (bw - obj.width) // 2, (bh - obj.height) // 2
                    else:
                        candidate_x = rng.randint(0, bw - obj.width)
                        candidate_y = rng.randint(0, bh - obj.height)
                    roi = coverage[candidate_y:candidate_y + obj.height, candidate_x:candidate_x + obj.width]
                    overlap_fraction = float(np.minimum(alpha, roi).sum() / max(float(alpha.sum()), 1e-6))
                    if allow_overlap or overlap_fraction <= .01:
                        x, y = candidate_x, candidate_y
                        accepted = True
                        break
                if not accepted: continue
                background.paste(obj, (x, y), obj)
                region = coverage[y:y + obj.height, x:x + obj.width]
                region[:] = alpha + region * (1.0 - alpha)
                canvas = np.zeros((bh, bw), dtype=np.float32)
                canvas[y:y + obj.height, x:x + obj.width] = alpha
                # Later opaque pixels occlude earlier object pixels. Retain a conservative
                # visible rectangle, suitable for box labels (not instance segmentation).
                for prior, _, _ in visible_masks:
                    prior *= (1.0 - canvas)
                visible_masks.append((canvas, class_id, (x + x0, y + y0, x + x1, y + y1)))
            for canvas, class_id, object_box in visible_masks:
                ys, xs = np.where(canvas > (10.0 / 255.0))
                if len(xs):
                    ox0, oy0, ox1, oy1 = object_box
                    ax0, ax1 = max(int(xs.min()), ox0), min(int(xs.max()) + 1, ox1)
                    ay0, ay1 = max(int(ys.min()), oy0), min(int(ys.max()) + 1, oy1)
                    if ax1 > ax0 and ay1 > ay0:
                        cx, cy = (ax0 + ax1) / 2 / bw, (ay0 + ay1) / 2 / bh
                        labels.append(f"{class_id} {cx:.6f} {cy:.6f} {(ax1-ax0)/bw:.6f} {(ay1-ay0)/bh:.6f}")
        stem = f"synth_{i:06d}"
        image_path = output / split / "images" / f"{stem}.jpg"
        label_path = output / split / "labels" / f"{stem}.txt"
        background.save(image_path, quality=95)
        label_path.write_text("\n".join(labels), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "dataset.yaml").write_text(
        f"path: {output.resolve().as_posix()}\ntrain: train/images\nval: val/images\nnames:\n" +
        "".join(f"  {i}: {name}\n" for i, name in enumerate(classes)), encoding="utf-8")


def validate_dataset(root: Path, classes: list[str]):
    errors = []
    sets = {}
    for split in ("train", "val", "test"):
        image_dir, label_dir = root / split / "images", root / split / "labels"
        if not image_dir.exists():
            if split == "test": continue
            errors.append(f"missing {image_dir}"); continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        sets[split] = {p.stem for p in images}
        for image in images:
            label = label_dir / (image.stem + ".txt")
            if not label.is_file(): errors.append(f"missing label: {label}"); continue
            try:
                with Image.open(image) as im: w, h = im.size
                for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                    parts = line.split()
                    if len(parts) != 5: raise ValueError(f"line {line_no}: expected 5 fields")
                    cls = int(parts[0]); x, y, bw, bh = map(float, parts[1:])
                    if not 0 <= cls < len(classes) or not all(np.isfinite(v) for v in (x,y,bw,bh)):
                        raise ValueError(f"line {line_no}: invalid class or nonfinite coordinate")
                    if bw <= 0 or bh <= 0 or not (0 <= x <= 1 and 0 <= y <= 1 and bw <= 1 and bh <= 1):
                        raise ValueError(f"line {line_no}: normalized box outside valid bounds")
                    if x-bw/2 < -1e-5 or y-bh/2 < -1e-5 or x+bw/2 > 1.00001 or y+bh/2 > 1.00001:
                        raise ValueError(f"line {line_no}: box extends outside image")
            except Exception as e: errors.append(f"{label}: {e}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in sets and b in sets and sets[a] & sets[b]: errors.append(f"duplicate stems across {a}/{b}")
    manifest = root / "split_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        train_bg = {k for k,v in data.get("background_split", {}).items() if v == "train"}
        val_bg = {k for k,v in data.get("background_split", {}).items() if v == "val"}
        if train_bg & val_bg: errors.append("background source leakage between train and val")
    return errors


def yolo_to_coco(root: Path, split: str, classes: list[str]):
    image_dir, label_dir = root / split / "images", root / split / "labels"
    images, annotations = [], []
    ann_id = 1
    for image_id, image in enumerate(sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS), 1):
        with Image.open(image) as im: width, height = im.size
        images.append({"id": image_id, "file_name": image.name, "width": width, "height": height})
        label = label_dir / (image.stem + ".txt")
        if not label.exists(): continue
        for line in label.read_text(encoding="utf-8").splitlines():
            cls, xc, yc, bw, bh = line.split(); cls = int(cls); xc, yc, bw, bh = map(float, (xc,yc,bw,bh))
            wp, hp = bw * width, bh * height
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": cls + 1,
                "bbox": [xc*width-wp/2, yc*height-hp/2, wp, hp], "area": wp*hp, "iscrowd": 0})
            ann_id += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i+1, "name": n} for i,n in enumerate(classes)]}


def write_coco(root: Path, classes: list[str]):
    ann_dir = root / "annotations"; ann_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        if (root / split / "images").exists():
            path = ann_dir / f"instances_{split}2017.json"
            path.write_text(json.dumps(yolo_to_coco(root, split, classes), indent=2), encoding="utf-8")


def import_real(images: Path, labels: Path, root: Path, split: str):
    """Copy labeled real photos into held-out val/test without editing source data."""
    if split not in {"val", "test"}:
        raise ValueError("Real images can only be imported to val or test")
    destination_images = root / split / "images"
    destination_labels = root / split / "labels"
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files: raise ValueError(f"No image files found in {images}")
    for image in files:
        label = labels / f"{image.stem}.txt"
        if not label.is_file(): raise ValueError(f"Missing YOLO label for {image.name}: {label}")
        for target in (destination_images / image.name, destination_labels / label.name):
            if target.exists(): raise FileExistsError(f"Refusing to overwrite {target}")
        shutil.copy2(image, destination_images / image.name)
        shutil.copy2(label, destination_labels / label.name)
