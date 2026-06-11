"""Compare GDINO-prompted vs full-image-box SAM segmentation.

Usage:
    diagnostic-masks <input_folder> <output_folder> [--n N]

For each CR3 or raster image (up to N, default 20), runs SAM twice:
  1. With Grounding DINO bounding box as prompt  (left panel)
  2. With the full image as the bounding box     (right panel)

Overlay colours:
  Blue  — person mask (subject)
  Red   — background
  Yellow rectangle — bounding box used as SAM prompt
  Green dots       — face keypoints above confidence threshold

Output: one JPEG per input image saved to <output_folder>.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .main import (
    apply_orientation,
    extract_preview_image,
    get_orientation,
    load_ml_models,
)
from .ml_crop import (
    FACE_KP_INDICES,
    FACE_KP_THRESHOLD,
    _resize_for_inference,
    _run_sam,
    _run_vitpose,
)

_CR3_SUFFIXES = {".cr3"}
_RASTER_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
OVERLAY_ALPHA = 0.40


def _load_image(path: Path):
    suffix = path.suffix.lower()
    if suffix in _CR3_SUFFIXES:
        orientation = get_orientation(path)
        return apply_orientation(extract_preview_image(path), orientation)
    if suffix in _RASTER_SUFFIXES:
        return Image.open(path).convert("RGB")
    return None


def _make_overlay(image_np, mask, bbox=None, face_kps=None):
    """Return a PIL Image with blue/red overlay, optional bbox and keypoints."""
    h, w = image_np.shape[:2]
    colour_layer = np.zeros_like(image_np)
    colour_layer[mask] = [30, 100, 255]    # blue — subject
    colour_layer[~mask] = [220, 40, 40]    # red  — background
    blended = (
        image_np.astype(float) * (1 - OVERLAY_ALPHA)
        + colour_layer.astype(float) * OVERLAY_ALPHA
    ).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(blended)
    draw = ImageDraw.Draw(img)
    if bbox is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 220, 0), width=3)
    if face_kps is not None:
        for x, y in face_kps:
            r = max(3, w // 200)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 230, 80))
    return img


def _label(img, text, colour=(255, 255, 255)):
    """Add a small text label in the top-left corner."""
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0, 180))
    draw.text((4, 3), text, fill=colour)
    return img


def _run_one(image_inf, models, bbox):
    """Run SAM + ViTPose for a given bbox. Returns (mask_np, face_kps) or (None, None)."""
    sam_processor = models[2]
    sam_model = models[3]
    vitpose_processor = models[4]
    vitpose_model = models[5]
    device = next(models[1].parameters()).device

    try:
        mask = _run_sam(image_inf, bbox, sam_processor, sam_model, device)
    except Exception as e:
        print(f"    SAM failed: {e}", file=sys.stderr)
        return None, None

    try:
        kps, scores = _run_vitpose(image_inf, bbox, vitpose_processor, vitpose_model, device)
    except Exception as e:
        print(f"    ViTPose failed: {e}", file=sys.stderr)
        kps, scores = None, None

    face_kps = None
    if kps is not None and scores is not None:
        indices = [i for i in FACE_KP_INDICES if scores[i] > FACE_KP_THRESHOLD]
        if indices:
            face_kps = kps[indices].tolist()

    return mask, face_kps


def _side_by_side(left, right, gap=8):
    """Join two PIL images horizontally with a thin dark gap."""
    w = left.width + gap + right.width
    h = max(left.height, right.height)
    out = Image.new("RGB", (w, h), (30, 30, 30))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gap, 0))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Produce diagnostic mask overlays comparing GDINO-prompted vs full-image SAM."
    )
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--n", type=int, default=20, help="Max images to process (default 20)")
    args = parser.parse_args()

    root = args.input_folder.expanduser().resolve()
    out_dir = args.output_folder.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suffixes = _CR3_SUFFIXES | _RASTER_SUFFIXES
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in suffixes]
    if not all_files:
        raise SystemExit(f"No supported image files found under: {root}")
    files = all_files[: args.n]
    print(f"Found {len(all_files)} files, processing {len(files)}.")

    print("Loading models...")
    models = load_ml_models()
    gdino_processor = models[0]
    gdino_model = models[1]

    from .main import detect_people_with_masks

    for path in files:
        print(f"  {path.name}")
        image = _load_image(path)
        if image is None:
            print(f"    Skipped (unsupported format)")
            continue

        image_inf, _ = _resize_for_inference(image)
        w_inf, h_inf = image_inf.size
        image_np = np.array(image_inf)
        full_bbox = (0, 0, w_inf, h_inf)

        # Grounding DINO detection
        gdino_bbox = None
        try:
            boxes, _, _, _ = detect_people_with_masks(
                (gdino_processor, gdino_model), image_inf
            )
            if boxes:
                if len(boxes) > 1:
                    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                    gdino_bbox = boxes[int(np.argmax(areas))]
                    print(f"    GDINO: {len(boxes)} people, using largest")
                else:
                    gdino_bbox = boxes[0]
            else:
                print(f"    GDINO: no person detected")
        except Exception as e:
            print(f"    GDINO failed: {e}", file=sys.stderr)

        # Left panel: GDINO bbox (or blank if no detection)
        if gdino_bbox is not None:
            with torch.no_grad():
                mask_gdino, kps_gdino = _run_one(image_inf, models, gdino_bbox)
        else:
            mask_gdino, kps_gdino = None, None

        if mask_gdino is not None:
            left = _make_overlay(image_np, mask_gdino, bbox=gdino_bbox, face_kps=kps_gdino)
        else:
            left = image_inf.copy()
            ImageDraw.Draw(left).text((4, 3), "No detection", fill=(255, 80, 80))
        _label(left, "GDINO bbox")

        # Right panel: full-image bbox
        with torch.no_grad():
            mask_full, kps_full = _run_one(image_inf, models, full_bbox)

        if mask_full is not None:
            right = _make_overlay(image_np, mask_full, bbox=full_bbox, face_kps=kps_full)
        else:
            right = image_inf.copy()
        _label(right, "Full-image bbox")

        out_path = out_dir / (path.stem + "_diagnostic.jpg")
        result = _side_by_side(left, right)
        result.save(out_path, format="JPEG", quality=88)
        print(f"    → {out_path.name}")

    print(f"\nDone. {len(files)} images written to {out_dir}")


if __name__ == "__main__":
    main()
