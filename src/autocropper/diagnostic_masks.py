"""Compare person segmentation across four strategies.

Usage:
    diagnostic-masks <input_folder> <output_folder> [--n N]

Produces a four-panel side-by-side JPEG for each image:

  Panel 1 — GDINO bbox → MobileSAM          (current pipeline)
  Panel 2 — YOLOv8-pose bbox → MobileSAM    (faster detector, same SAM)
  Panel 3 — YOLOv8-seg mask directly         (no SAM)
  Panel 4 — GDINO bbox → SAM 2 tiny          (newer SAM, same detector)

Overlay colours:
  Blue   — person mask (subject)
  Red    — background
  Yellow — bounding box used as SAM prompt
  Green  — face keypoints above confidence threshold
"""

import argparse
import sys
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

SAM2_MODEL = "facebook/sam2.1-hiera-tiny"

_CR3_SUFFIXES = {".cr3"}
_RASTER_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
OVERLAY_ALPHA = 0.40
YOLO_PERSON_CLASS = 0


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
    colour_layer[mask]  = [30,  100, 255]   # blue  — subject
    colour_layer[~mask] = [220,  40,  40]   # red   — background
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
    img = image_inf.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 80, 80))
    return img


def _sam_panel(image_inf, image_np, models, bbox, face_kps, label):
    """Run MobileSAM with bbox prompt; return labelled overlay panel."""
    sam_processor = models[2]
    sam_model     = models[3]
    device = next(models[1].parameters()).device
    try:
        with torch.no_grad():
            mask = _run_sam(image_inf, bbox, sam_processor, sam_model, device)
        panel = _make_overlay(image_np, mask, bbox=bbox, face_kps=face_kps)
    except Exception as e:
        print(f"    SAM failed: {e}", file=sys.stderr)
        panel = image_inf.copy()
    return _label(panel, label)


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
    # Map back to original indices for mask lookup
    original_indices = torch.where(person)[0]
    return xyxy[idx].numpy(), float(confs[idx]), int(original_indices[idx])


def _yolo_face_kps(results, original_idx):
    """Extract face keypoints from a YOLOv8-pose result for a specific detection index."""
    if results.keypoints is None:
        return None
    kps   = results.keypoints.xy.cpu().numpy()[original_idx]    # (17, 2)
    confs = results.keypoints.conf.cpu().numpy()[original_idx]  # (17,)
    indices = [i for i in FACE_KP_INDICES if confs[i] > FACE_KP_THRESHOLD]
    return kps[indices].tolist() if indices else None


def _yolo_seg_mask(results, original_idx, target_size):
    """Return resized boolean mask from YOLOv8-seg result, or None."""
    if results.masks is None:
        return None
    mask_tensor = results.masks.data[original_idx]   # (H, W) float32
    mask_img = Image.fromarray((mask_tensor.cpu().numpy() * 255).astype(np.uint8))
    mask_resized = mask_img.resize(target_size, Image.NEAREST)
    return np.array(mask_resized) > 127


def _load_sam2(device):
    """Load SAM 2.1 tiny from HuggingFace transformers."""
    from transformers import Sam2Model, Sam2Processor
    processor = Sam2Processor.from_pretrained(SAM2_MODEL)
    model = Sam2Model.from_pretrained(SAM2_MODEL).to(device).eval()
    return processor, model


def _run_sam2(image_pil, bbox, sam2_processor, sam2_model, device):
    """Return a boolean mask (H, W) using SAM 2.1 with a bbox prompt."""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    inputs = sam2_processor(
        images=image_pil,
        input_boxes=[[[x1, y1, x2, y2]]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = sam2_model(**inputs)
    masks, scores, _ = sam2_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )
    # masks[0]: (1, num_masks, H, W); pick highest-scored mask
    mask_batch = masks[0][0]           # (num_masks, H, W)
    score_batch = scores[0][0]         # (num_masks,)
    best = int(score_batch.argmax())
    return mask_batch[best].cpu().numpy().astype(bool)


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
        description="Four-panel diagnostic comparing GDINO/YOLOv8 detectors with MobileSAM/SAM2.1."
    )
    parser.add_argument("input_folder",  type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--n", type=int, default=20, help="Max images to process (default 20)")
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

    print("Loading GDINO + MobileSAM + ViTPose...")
    models = load_ml_models()
    gdino_processor, gdino_model = models[0], models[1]
    device = next(gdino_model.parameters()).device

    print("Loading YOLOv8-pose and YOLOv8-seg...")
    from ultralytics import YOLO
    yolo_pose = YOLO("yolov8n-pose.pt")
    yolo_seg  = YOLO("yolov8n-seg.pt")

    print(f"Loading SAM 2.1 tiny ({SAM2_MODEL})...")
    sam2_processor, sam2_model = _load_sam2(device)
    print("All models loaded.\n")

    for path in files:
        print(f"  {path.name}")
        image = _load_image(path)
        if image is None:
            print(f"    Skipped (unsupported format)")
            continue

        image_inf, _ = _resize_for_inference(image)
        w_inf, h_inf  = image_inf.size
        image_np      = np.array(image_inf)

        # ── GDINO detection (shared by panels 1 and 4) ──────────────────
        gdino_bbox, gdino_kps = None, None
        try:
            boxes, _, _, _ = detect_people_with_masks(
                (gdino_processor, gdino_model), image_inf
            )
            if boxes:
                if len(boxes) > 1:
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                    gdino_bbox = boxes[int(np.argmax(areas))]
                    print(f"    GDINO: {len(boxes)} people, using largest")
                else:
                    gdino_bbox = boxes[0]
                # ViTPose keypoints
                vp_proc = models[4]
                vp_model = models[5]
                with torch.no_grad():
                    kps, scores = _run_vitpose(image_inf, gdino_bbox, vp_proc, vp_model, device)
                if kps is not None:
                    idx = [i for i in FACE_KP_INDICES if scores[i] > FACE_KP_THRESHOLD]
                    gdino_kps = kps[idx].tolist() if idx else None
            else:
                print(f"    GDINO: no person detected")
        except Exception as e:
            print(f"    GDINO failed: {e}", file=sys.stderr)

        # ── Panel 1: GDINO → MobileSAM ──────────────────────────────────
        if gdino_bbox is not None:
            panel1 = _sam_panel(image_inf, image_np, models, gdino_bbox, gdino_kps,
                                 "GDINO → MobileSAM")
        else:
            panel1 = _no_detection_panel(image_inf, "GDINO → MobileSAM  [no detection]")

        # ── Panel 2: YOLOv8-pose → MobileSAM ───────────────────────────
        yolo_bbox, yolo_kps = None, None
        try:
            with torch.no_grad():
                res_pose = yolo_pose(image_np, verbose=False)[0]
            sel = _yolo_select_person(res_pose)
            if sel is not None:
                yolo_bbox, _, orig_idx = sel
                yolo_kps = _yolo_face_kps(res_pose, orig_idx)
                if len(res_pose.boxes) > 1:
                    print(f"    YOLO-pose: {int((res_pose.boxes.cls==0).sum())} people, using largest")
            else:
                print(f"    YOLO-pose: no person detected")
        except Exception as e:
            print(f"    YOLO-pose failed: {e}", file=sys.stderr)

        if yolo_bbox is not None:
            panel2 = _sam_panel(image_inf, image_np, models, yolo_bbox, yolo_kps,
                                 "YOLOv8-pose → SAM")
        else:
            panel2 = _no_detection_panel(image_inf, "YOLOv8-pose → SAM  [no detection]")

        # ── Panel 3: YOLOv8-seg (mask only, no SAM) ─────────────────────
        try:
            with torch.no_grad():
                res_seg = yolo_seg(image_np, verbose=False)[0]
            sel_seg = _yolo_select_person(res_seg)
            if sel_seg is not None:
                seg_bbox, _, seg_idx = sel_seg
                seg_mask = _yolo_seg_mask(res_seg, seg_idx, (w_inf, h_inf))
                if seg_mask is not None:
                    panel3 = _make_overlay(image_np, seg_mask, bbox=seg_bbox)
                    panel3 = _label(panel3, "YOLOv8-seg")
                else:
                    panel3 = _no_detection_panel(image_inf, "YOLOv8-seg  [no mask]")
            else:
                print(f"    YOLO-seg: no person detected")
                panel3 = _no_detection_panel(image_inf, "YOLOv8-seg  [no detection]")
        except Exception as e:
            print(f"    YOLO-seg failed: {e}", file=sys.stderr)
            panel3 = _no_detection_panel(image_inf, "YOLOv8-seg  [error]")

        # ── Panel 4: GDINO → SAM 2.1 tiny ───────────────────────────────
        if gdino_bbox is not None:
            try:
                mask4 = _run_sam2(image_inf, gdino_bbox, sam2_processor, sam2_model, device)
                panel4 = _make_overlay(image_np, mask4, bbox=gdino_bbox, face_kps=gdino_kps)
                panel4 = _label(panel4, "GDINO → SAM 2.1 tiny")
            except Exception as e:
                print(f"    SAM 2.1 failed: {e}", file=sys.stderr)
                panel4 = _no_detection_panel(image_inf, "GDINO → SAM 2.1  [error]")
        else:
            panel4 = _no_detection_panel(image_inf, "GDINO → SAM 2.1  [no detection]")

        out_path = out_dir / (path.stem + "_diagnostic.jpg")
        _stack(panel1, panel2, panel3, panel4).save(out_path, format="JPEG", quality=88)
        print(f"    → {out_path.name}")

    print(f"\nDone. {len(files)} images written to {out_dir}")


if __name__ == "__main__":
    main()
