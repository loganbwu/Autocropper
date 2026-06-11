"""Compare person segmentation across five strategies.

Usage:
    diagnostic-masks <input_folder> <output_folder> [--n N] [--yolo-padding P]

Produces a five-panel side-by-side JPEG for each image:

  Panel 1 — GDINO bbox → SAM 2.1            (current pipeline, reference)
  Panel 2 — YOLOv8-pose tight bbox → SAM 2.1  (baseline YOLO)
  Panel 3 — YOLOv8-pose + padding → SAM 2.1   (bbox expanded by --yolo-padding %)
  Panel 4 — YOLOv8-pose + keypoints union → SAM 2.1  (bbox expanded to cover all kps)
  Panel 5 — YOLOv8-seg mask directly          (no SAM)

Overlay colours:
  Blue   — person mask (subject)
  Red    — background
  Yellow — bounding box used as SAM prompt
  Green  — face keypoints above confidence threshold
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .main import (
    apply_orientation,
    detect_people_with_masks,
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

_CR3_SUFFIXES    = {".cr3"}
_RASTER_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
OVERLAY_ALPHA    = 0.40
YOLO_PERSON_CLASS = 0
KPS_CONF_THRESHOLD = 0.1   # lower threshold to catch feet/hands for bbox expansion


def _load_image(path: Path):
    suffix = path.suffix.lower()
    if suffix in _CR3_SUFFIXES:
        return apply_orientation(extract_preview_image(path), get_orientation(path))
    if suffix in _RASTER_SUFFIXES:
        return Image.open(path).convert("RGB")
    return None


def _make_overlay(image_np, mask, bbox=None, face_kps=None):
    h, w = image_np.shape[:2]
    colour_layer = np.zeros_like(image_np)
    colour_layer[mask]  = [30,  100, 255]
    colour_layer[~mask] = [220,  40,  40]
    blended = (
        image_np.astype(float) * (1 - OVERLAY_ALPHA)
        + colour_layer.astype(float) * OVERLAY_ALPHA
    ).clip(0, 255).astype(np.uint8)
    img  = Image.fromarray(blended)
    draw = ImageDraw.Draw(img)
    if bbox is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 220, 0), width=3)
    if face_kps is not None:
        r = max(3, w // 200)
        for x, y in face_kps:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 230, 80))
    return img


def _label(img, text):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return img


def _no_detection_panel(image_inf, label):
    img  = image_inf.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 80, 80))
    return img


def _yolo_select_person(results):
    """Return (bbox_xyxy, conf, index) for the largest-area person, or None."""
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return None
    person = boxes.cls.cpu() == YOLO_PERSON_CLASS
    if not person.any():
        return None
    xyxy  = boxes.xyxy.cpu()[person]
    confs = boxes.conf.cpu()[person]
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx   = int(areas.argmax())
    original_indices = torch.where(person)[0]
    return xyxy[idx].numpy(), float(confs[idx]), int(original_indices[idx])


def _yolo_face_kps(results, original_idx):
    """Extract face keypoints from a YOLOv8-pose result for a specific detection index."""
    if results.keypoints is None:
        return None
    kps   = results.keypoints.xy.cpu().numpy()[original_idx]
    confs = results.keypoints.conf.cpu().numpy()[original_idx]
    indices = [i for i in FACE_KP_INDICES if confs[i] > FACE_KP_THRESHOLD]
    return kps[indices].tolist() if indices else None


def _yolo_seg_mask(results, original_idx, target_size):
    """Return resized boolean mask from YOLOv8-seg result, or None."""
    if results.masks is None:
        return None
    mask_tensor = results.masks.data[original_idx]
    mask_img    = Image.fromarray((mask_tensor.cpu().numpy() * 255).astype(np.uint8))
    return np.array(mask_img.resize(target_size, Image.NEAREST)) > 127


def _pad_bbox(bbox, pad_frac, w, h):
    """Expand bbox symmetrically by pad_frac (e.g. 0.15 = 15%), clipped to image."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw = (x2 - x1) * (1 + pad_frac)
    bh = (y2 - y1) * (1 + pad_frac)
    return (
        max(0.0, cx - bw / 2),
        max(0.0, cy - bh / 2),
        min(float(w), cx + bw / 2),
        min(float(h), cy + bh / 2),
    )


def _yolo_expand_with_kps(bbox, results, original_idx, w, h):
    """Expand bbox to include all high-confidence keypoints (all 17, not just face)."""
    if results.keypoints is None:
        return bbox
    kps   = results.keypoints.xy.cpu().numpy()[original_idx]    # (17, 2)
    confs = results.keypoints.conf.cpu().numpy()[original_idx]  # (17,)
    valid = kps[confs > KPS_CONF_THRESHOLD]
    if len(valid) == 0:
        return bbox
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (
        min(x1, float(valid[:, 0].min())),
        min(y1, float(valid[:, 1].min())),
        max(x2, float(valid[:, 0].max())),
        max(y2, float(valid[:, 1].max())),
    )


def _stack(*panels, gap=8):
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    h = max(p.height for p in panels)
    out = Image.new("RGB", (total_w, h), (30, 30, 30))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Five-panel diagnostic: GDINO vs YOLOv8 bbox strategies with SAM 2.1."
    )
    parser.add_argument("input_folder",  type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--n", type=int, default=20, help="Max images to process (default 20)")
    parser.add_argument("--yolo-padding", type=float, default=0.15, metavar="P",
                        help="Fraction to expand YOLO bbox for panel 3 (default 0.15 = 15%%)")
    args = parser.parse_args()

    root    = args.input_folder.expanduser().resolve()
    out_dir = args.output_folder.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suffixes  = _CR3_SUFFIXES | _RASTER_SUFFIXES
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in suffixes]
    if not all_files:
        raise SystemExit(f"No supported image files found under: {root}")
    files = all_files[: args.n]
    print(f"Found {len(all_files)} files, processing {len(files)}.")
    print(f"YOLO padding (panel 3): {args.yolo_padding * 100:.0f}%")

    print("Loading GDINO + SAM 2.1 + ViTPose...")
    models = load_ml_models()
    gdino_processor, gdino_model = models[0], models[1]
    device = next(gdino_model.parameters()).device

    print("Loading YOLOv8-pose and YOLOv8-seg...")
    from ultralytics import YOLO
    yolo_pose = YOLO("yolov8n-pose.pt")
    yolo_seg  = YOLO("yolov8n-seg.pt")
    print("All models loaded.\n")

    t_gdino       = []
    t_vitpose     = []
    t_sam_gdino   = []
    t_yolo_pose   = []
    t_sam_tight   = []
    t_sam_padded  = []
    t_sam_kpu     = []
    t_yolo_seg    = []

    for path in files:
        print(f"  {path.name}")
        image = _load_image(path)
        if image is None:
            print(f"    Skipped (unsupported format)")
            continue

        image_inf, _ = _resize_for_inference(image)
        w_inf, h_inf  = image_inf.size
        image_np      = np.array(image_inf)

        # ── GDINO detection ──────────────────────────────────────────────
        gdino_bbox, gdino_kps = None, None
        try:
            t0 = time.perf_counter()
            boxes, _, _, _ = detect_people_with_masks(
                (gdino_processor, gdino_model), image_inf
            )
            t_gdino.append(time.perf_counter() - t0)
            if boxes:
                if len(boxes) > 1:
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                    gdino_bbox = boxes[int(np.argmax(areas))]
                    print(f"    GDINO: {len(boxes)} people, using largest")
                else:
                    gdino_bbox = boxes[0]
                t0 = time.perf_counter()
                with torch.no_grad():
                    kps, scores = _run_vitpose(image_inf, gdino_bbox, models[4], models[5], device)
                t_vitpose.append(time.perf_counter() - t0)
                if kps is not None:
                    idx = [i for i in FACE_KP_INDICES if scores[i] > FACE_KP_THRESHOLD]
                    gdino_kps = kps[idx].tolist() if idx else None
            else:
                print(f"    GDINO: no person detected")
        except Exception as e:
            print(f"    GDINO failed: {e}", file=sys.stderr)

        # ── Panel 1: GDINO → SAM 2.1 ─────────────────────────────────────
        if gdino_bbox is not None:
            try:
                t0 = time.perf_counter()
                with torch.no_grad():
                    mask1 = _run_sam(image_inf, gdino_bbox, models[2], models[3], device)
                t_sam_gdino.append(time.perf_counter() - t0)
                panel1 = _make_overlay(image_np, mask1, bbox=gdino_bbox, face_kps=gdino_kps)
                panel1 = _label(panel1, "GDINO → SAM 2.1")
            except Exception as e:
                print(f"    SAM (panel 1) failed: {e}", file=sys.stderr)
                panel1 = _no_detection_panel(image_inf, "GDINO → SAM 2.1  [error]")
        else:
            panel1 = _no_detection_panel(image_inf, "GDINO → SAM 2.1  [no detection]")

        # ── YOLO-pose detection (shared by panels 2–4) ───────────────────
        yolo_bbox, yolo_kps, res_pose, orig_idx = None, None, None, None
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                res_pose = yolo_pose(image_np, verbose=False)[0]
            t_yolo_pose.append(time.perf_counter() - t0)
            sel = _yolo_select_person(res_pose)
            if sel is not None:
                yolo_bbox, _, orig_idx = sel
                yolo_kps = _yolo_face_kps(res_pose, orig_idx)
                if int((res_pose.boxes.cls.cpu() == 0).sum()) > 1:
                    print(f"    YOLO-pose: {int((res_pose.boxes.cls.cpu()==0).sum())} people, using largest")
            else:
                print(f"    YOLO-pose: no person detected")
        except Exception as e:
            print(f"    YOLO-pose failed: {e}", file=sys.stderr)

        def _sam_panel(bbox, label, t_list):
            try:
                t0 = time.perf_counter()
                with torch.no_grad():
                    m = _run_sam(image_inf, bbox, models[2], models[3], device)
                t_list.append(time.perf_counter() - t0)
                p = _make_overlay(image_np, m, bbox=bbox, face_kps=yolo_kps)
                return _label(p, label)
            except Exception as e:
                print(f"    SAM failed for {label}: {e}", file=sys.stderr)
                return _no_detection_panel(image_inf, f"{label}  [error]")

        # ── Panel 2: YOLO tight bbox → SAM 2.1 ──────────────────────────
        if yolo_bbox is not None:
            panel2 = _sam_panel(yolo_bbox, "YOLO tight → SAM 2.1", t_sam_tight)
        else:
            panel2 = _no_detection_panel(image_inf, "YOLO tight → SAM 2.1  [no detection]")

        # ── Panel 3: YOLO + padding → SAM 2.1 ───────────────────────────
        if yolo_bbox is not None:
            padded_bbox = _pad_bbox(yolo_bbox, args.yolo_padding, w_inf, h_inf)
            panel3 = _sam_panel(padded_bbox, f"YOLO +{args.yolo_padding*100:.0f}% → SAM 2.1", t_sam_padded)
        else:
            panel3 = _no_detection_panel(image_inf, "YOLO padded → SAM 2.1  [no detection]")

        # ── Panel 4: YOLO + keypoints union → SAM 2.1 ───────────────────
        if yolo_bbox is not None and res_pose is not None:
            kpu_bbox = _yolo_expand_with_kps(yolo_bbox, res_pose, orig_idx, w_inf, h_inf)
            panel4 = _sam_panel(kpu_bbox, "YOLO kp-union → SAM 2.1", t_sam_kpu)
        else:
            panel4 = _no_detection_panel(image_inf, "YOLO kp-union → SAM 2.1  [no detection]")

        # ── Panel 5: YOLOv8-seg (no SAM) ────────────────────────────────
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                res_seg = yolo_seg(image_np, verbose=False)[0]
            t_yolo_seg.append(time.perf_counter() - t0)
            sel_seg = _yolo_select_person(res_seg)
            if sel_seg is not None:
                seg_bbox, _, seg_idx = sel_seg
                seg_mask = _yolo_seg_mask(res_seg, seg_idx, (w_inf, h_inf))
                if seg_mask is not None:
                    panel5 = _make_overlay(image_np, seg_mask, bbox=seg_bbox)
                    panel5 = _label(panel5, "YOLOv8-seg")
                else:
                    panel5 = _no_detection_panel(image_inf, "YOLOv8-seg  [no mask]")
            else:
                panel5 = _no_detection_panel(image_inf, "YOLOv8-seg  [no detection]")
        except Exception as e:
            print(f"    YOLO-seg failed: {e}", file=sys.stderr)
            panel5 = _no_detection_panel(image_inf, "YOLOv8-seg  [error]")

        out_path = out_dir / (path.stem + "_diagnostic.jpg")
        _stack(panel1, panel2, panel3, panel4, panel5).save(out_path, format="JPEG", quality=88)
        print(f"    → {out_path.name}")

    def _mean(ts):
        return sum(ts) / len(ts) if ts else float("nan")

    print(f"\nDone. {len(files)} images written to {out_dir}")
    print(f"\nMean inference times (n={len(files)} images):")
    print(f"  {'Component':<35}  {'mean (s)':>8}  {'n':>4}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*4}")
    for label, ts in [
        ("GDINO detection",               t_gdino),
        ("ViTPose",                       t_vitpose),
        ("SAM 2.1 (GDINO bbox)",          t_sam_gdino),
        ("YOLOv8-pose detection",         t_yolo_pose),
        ("SAM 2.1 (YOLO tight bbox)",     t_sam_tight),
        (f"SAM 2.1 (YOLO +{args.yolo_padding*100:.0f}% pad)", t_sam_padded),
        ("SAM 2.1 (YOLO kp-union bbox)",  t_sam_kpu),
        ("YOLOv8-seg",                    t_yolo_seg),
    ]:
        print(f"  {label:<35}  {_mean(ts):>8.3f}  {len(ts):>4}")
    print()
    print(f"  {'Pipeline':<35}  {'mean (s)':>8}")
    print(f"  {'-'*35}  {'-'*8}")
    for label, total in [
        ("GDINO + ViTPose → SAM 2.1",
            _mean(t_gdino) + _mean(t_vitpose) + _mean(t_sam_gdino)),
        ("YOLO tight → SAM 2.1",
            _mean(t_yolo_pose) + _mean(t_sam_tight)),
        (f"YOLO +{args.yolo_padding*100:.0f}% pad → SAM 2.1",
            _mean(t_yolo_pose) + _mean(t_sam_padded)),
        ("YOLO kp-union → SAM 2.1",
            _mean(t_yolo_pose) + _mean(t_sam_kpu)),
        ("YOLOv8-seg",
            _mean(t_yolo_seg)),
    ]:
        print(f"  {label:<35}  {total:>8.3f}")


if __name__ == "__main__":
    main()
