"""Case-based ML crop prediction.

Workflow:
  1. Build a training dataset from already-cropped CR3 files using build_training_dataset.py.
  2. Optionally optimise the alpha weighting using optimize_weights.py.
  3. Use predict_ml_crop() in the web UI or CLI as an alternative to compute_crop().
"""

import io
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ANSI colour helpers — disabled automatically when output is not a terminal.
_C      = sys.stdout.isatty()
_GREEN  = "\033[32m" if _C else ""
_YELLOW = "\033[33m" if _C else ""
_RED    = "\033[31m" if _C else ""
_CYAN   = "\033[36m" if _C else ""
_DIM    = "\033[2m"  if _C else ""
_RESET  = "\033[0m"  if _C else ""

import numpy as np
import torch
from PIL import Image

FACE_KP_INDICES = [0, 1, 2, 3, 4]   # COCO: nose, left_eye, right_eye, left_ear, right_ear
FACE_KP_THRESHOLD = 0.3
MASK_SIZE = 1024
KNN_COMPARE_SIZE = 128                # masks are downsampled to this resolution for k-NN comparison
AR_EPSILON = 0.05                     # aspect-ratio tolerance for similarity matching
DEFAULT_N_NEIGHBORS = 10
MAX_INFERENCE_SIZE = 800              # longest edge fed to models; SAM/GDINO resize internally anyway


# ---- Data structures ----

@dataclass
class TrainingRecord:
    mask: np.ndarray        # bool (MASK_SIZE, MASK_SIZE) — pre-normalised
    face_centroid: tuple    # (x, y) normalised within mask bbox [0, 1], or None
    crop_center: tuple      # (x, y) normalised within mask bbox [0, 1]
    min_margin: float       # min of 4 normalised margins (each normalised by mask dim)
    aspect_ratio: float     # image display width / height
    source_path: str = ""   # absolute path to source image; empty for older records

    def __getstate__(self):
        state = self.__dict__.copy()
        mask = state.pop('mask')
        state['_mask_packed'] = np.packbits(mask.flatten())
        state['_mask_shape'] = mask.shape
        return state

    def __setstate__(self, state):
        if '_mask_packed' in state:
            h, w = state.pop('_mask_shape')
            state['mask'] = np.unpackbits(state.pop('_mask_packed'), count=h * w).reshape(h, w).astype(bool)
        self.__dict__.update(state)


@dataclass
class TrainingDataset:
    records: list = field(default_factory=list)
    alpha: float = 0.5      # weight: 0 = face distance only, 1 = mask overlap only

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path_or_stream):
        if hasattr(path_or_stream, 'read'):
            return pickle.load(path_or_stream)
        with open(path_or_stream, 'rb') as f:
            return pickle.load(f)


# ---- Low-level inference helpers ----

def _resize_for_inference(image):
    """Shrink image so longest edge ≤ MAX_INFERENCE_SIZE. Returns (image, scale).
    Coordinates in inference space must be divided by scale to recover original pixels."""
    w, h = image.size
    scale = min(MAX_INFERENCE_SIZE / max(w, h), 1.0)
    if scale >= 1.0:
        return image, 1.0
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS), scale


def _to_device(v, device):
    """Move tensor to device; falls back to float32 if MPS rejects float64."""
    if not hasattr(v, 'to'):
        return v
    try:
        return v.to(device)
    except TypeError:
        return v.float().to(device)


def _run_sam(image, bbox, sam_processor, sam_model, device):
    """Run SAM with a single bbox prompt; return best binary mask (H, W) bool.

    When sam_model is None, sam_processor is a MobileSAM SamPredictor instance.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if sam_model is None:
        # MobileSAM path
        image_np = np.array(image)  # (H, W, 3) uint8 RGB
        with torch.no_grad():
            sam_processor.set_image(image_np)
            masks, _, _ = sam_processor.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False,
            )
        return masks[0].astype(bool)  # (H, W)
    # Transformers SAM path (kept as fallback)
    inputs = sam_processor(images=image, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
    inputs = {k: _to_device(v, device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks_list = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    mask_t = masks_list[0]
    iou = outputs.iou_scores[0].cpu().numpy().flatten()
    mask_arr = mask_t.reshape(-1, mask_t.shape[-2], mask_t.shape[-1]).numpy()
    return mask_arr[int(np.argmax(iou))].astype(bool)


def _run_vitpose(image, bbox, vitpose_processor, vitpose_model, device):
    """Run ViTPose on a person bbox; return (keypoints (17,2), scores (17,)) or (None, None)."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    inputs = vitpose_processor(images=image, boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
    inputs = {k: _to_device(v, device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = vitpose_model(**inputs)
    poses = vitpose_processor.post_process_pose_estimation(
        outputs, boxes=[[[x1, y1, x2, y2]]]
    )
    if not poses or not poses[0]:
        return None, None
    kp = poses[0][0]
    kps = np.array(kp['keypoints'])   # (17, 2) pixel coords
    sc = np.array(kp['scores'])       # (17,)
    return kps, sc


def _mask_bbox(mask):
    """Return (x1, y1, x2, y2) bounding box of True region in mask, or None if empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    return int(x_idx[0]), int(y_idx[0]), int(x_idx[-1]), int(y_idx[-1])


# ---- Feature extraction ----

def extract_features(image, models, name=None, verbose=True):
    """Extract features from an image for ML crop matching.

    Returns a 4-tuple (mask_cropped, face_centroid, aspect_ratio, mask_bbox) or None.

    mask_cropped:  bool np.ndarray cropped to the mask bounding box in display pixels
    face_centroid: (x, y) normalised within mask bbox [0, 1], or None if no face found
    aspect_ratio:  image display width / height
    mask_bbox:     (mx1, my1, mx2, my2) in full-image display pixels

    Returns None if the image cannot be processed (no person detected, SAM failure,
    or empty mask). Returns face_centroid=None when face keypoints are absent.

    name:    optional filename used in warning messages.
    verbose: if False, suppresses per-image timing output (set False when building dataset).
    """
    from .main import detect_people_with_masks
    label = f" [{name}]" if name else ""

    gdino_processor = models[0]
    gdino_model     = models[1]
    sam_processor   = models[2]
    sam_model       = models[3]
    vitpose_processor = models[4]
    vitpose_model     = models[5]

    device = next(gdino_model.parameters()).device

    t0 = time.perf_counter()
    boxes, _, w, h = detect_people_with_masks((gdino_processor, gdino_model), image)
    t1 = time.perf_counter()
    if len(boxes) == 0:
        print(f"{_YELLOW}  GDINO (person detection){label}: no people detected{_RESET}")
        return None
    if len(boxes) > 1:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        bbox = boxes[idx]
        if verbose:
            print(f"{_CYAN}  GDINO (person detection){label}: found {len(boxes)} people, using largest (box {idx + 1}){_RESET}")
    else:
        bbox = boxes[0]

    try:
        full_mask = _run_sam(image, bbox, sam_processor, sam_model, device)
    except Exception as e:
        print(f"{_RED}  SAM (person segmentation){label} failed: {e}{_RESET}")
        return None
    t2 = time.perf_counter()
    bbox_mask = _mask_bbox(full_mask)
    if bbox_mask is None:
        print(f"{_YELLOW}  SAM (person segmentation){label}: returned an empty mask{_RESET}")
        return None
    mx1, my1, mx2, my2 = bbox_mask
    mask_w = mx2 - mx1
    mask_h = my2 - my1
    if mask_w <= 0 or mask_h <= 0:
        print(f"{_YELLOW}  SAM (person segmentation){label}: degenerate mask bbox ({mask_w}×{mask_h}){_RESET}")
        return None

    try:
        kps, scores = _run_vitpose(image, bbox, vitpose_processor, vitpose_model, device)
    except Exception as e:
        print(f"{_RED}  ViTPose (face keypoint detection){label} failed: {e}{_RESET}")
        return None
    t3 = time.perf_counter()
    if kps is None:
        print(f"{_YELLOW}  ViTPose (face keypoint detection){label}: no poses returned{_RESET}")
        return None

    if verbose:
        print(f"{_DIM}    gdino={t1-t0:.2f}s  sam={t2-t1:.2f}s  vitpose={t3-t2:.2f}s{label}{_RESET}")

    mask_cropped = full_mask[my1:my2 + 1, mx1:mx2 + 1]
    ar = float(w) / float(h)

    face_indices = [i for i in FACE_KP_INDICES if scores[i] > FACE_KP_THRESHOLD]
    if not face_indices:
        if verbose:
            print(f"{_YELLOW}  ViTPose (face keypoint detection){label}: no face keypoints above confidence threshold — mask-only similarity will be used{_RESET}")
        return mask_cropped, None, ar, (mx1, my1, mx2, my2)

    face_xy = kps[face_indices]
    fc_x = float((np.mean(face_xy[:, 0]) - mx1) / mask_w)
    fc_y = float((np.mean(face_xy[:, 1]) - my1) / mask_h)

    return mask_cropped, (fc_x, fc_y), ar, (mx1, my1, mx2, my2)


def build_training_record(image, crop_xyxy_display, models, name=None, source_path=""):
    """Build one TrainingRecord from an image with its known crop in display pixels.

    Returns None if the image cannot be processed.
    name:        optional filename used in warning messages.
    source_path: absolute path to the source image, stored in the record.
    """
    image_inf, inf_scale = _resize_for_inference(image)
    result = extract_features(image_inf, models, name=name, verbose=False)
    if result is None:
        return None
    mask_cropped, face_centroid, aspect_ratio, (mx1, my1, mx2, my2) = result
    if face_centroid is None:
        return None  # training records require a face centroid
    # Scale crop coordinates to inference resolution so all values are in the same space.
    crop_x1, crop_y1, crop_x2, crop_y2 = (c * inf_scale for c in crop_xyxy_display)

    mask_w_px = float(mx2 - mx1)
    mask_h_px = float(my2 - my1)

    crop_cx = (crop_x1 + crop_x2) / 2
    crop_cy = (crop_y1 + crop_y2) / 2
    cc_x = float((crop_cx - mx1) / mask_w_px)
    cc_y = float((crop_cy - my1) / mask_h_px)

    # Margins normalised by mask dimension (positive = crop extends beyond mask edge)
    left_m   = (mx1 - crop_x1) / mask_w_px
    right_m  = (crop_x2 - mx2) / mask_w_px
    top_m    = (my1 - crop_y1) / mask_h_px
    bottom_m = (crop_y2 - my2) / mask_h_px
    min_margin = float(min(left_m, right_m, top_m, bottom_m))

    return TrainingRecord(
        mask=_normalize_mask(mask_cropped),
        face_centroid=face_centroid,
        crop_center=(cc_x, cc_y),
        min_margin=min_margin,
        aspect_ratio=aspect_ratio,
        source_path=source_path,
    )


# ---- Similarity metrics ----

def _normalize_mask(mask):
    """Scale mask to MASK_SIZE px on longest edge, centre-pad to MASK_SIZE×MASK_SIZE."""
    if mask.shape == (MASK_SIZE, MASK_SIZE):
        return mask
    h, w = mask.shape
    scale = MASK_SIZE / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    scaled = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((new_w, new_h), Image.NEAREST)
    ).astype(bool)
    out = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    pad_y = (MASK_SIZE - new_h) // 2
    pad_x = (MASK_SIZE - new_w) // 2
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = scaled
    return out


def _mask_iou(m1, m2):
    union = float((m1 | m2).sum())
    return float((m1 & m2).sum()) / union if union > 0 else 1.0


def _face_dist(c1, c2):
    return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))


# ---- k-NN helpers ----

def _get_ar_candidates(dataset, query_ar):
    """Return (candidates, masks_small, face_centroids) for query_ar, cached on the dataset."""
    if not hasattr(dataset, '_ar_cache'):
        dataset._ar_cache = {}
    # Round to nearest AR_EPSILON so photos with the same AR share a cache entry.
    ar_key = round(query_ar / AR_EPSILON) * AR_EPSILON
    if ar_key not in dataset._ar_cache:
        candidates = [r for r in dataset.records if abs(r.aspect_ratio - query_ar) <= AR_EPSILON]
        step = MASK_SIZE // KNN_COMPARE_SIZE
        if candidates:
            # Precompute stacked downsampled masks — one-time cost, reused for every photo with this AR.
            masks_small = np.stack([r.mask[::step, ::step] for r in candidates])
            face_centroids = np.array([r.face_centroid for r in candidates], dtype=np.float32)
        else:
            masks_small = np.empty((0, KNN_COMPARE_SIZE, KNN_COMPARE_SIZE), dtype=bool)
            face_centroids = np.empty((0, 2), dtype=np.float32)
        dataset._ar_cache[ar_key] = (candidates, masks_small, face_centroids)
        print(f"{_DIM}  k-NN cache built for AR≈{ar_key:.2f}: {len(candidates)} candidates, masks_small={masks_small.nbytes // 1024}KB{_RESET}")
    return dataset._ar_cache[ar_key]


# ---- Prediction ----

def predict_ml_crop(cr3_path, dataset, models, n=DEFAULT_N_NEIGHBORS, _inference_lock=None):
    """Predict crop for cr3_path using case-based reasoning.

    Returns the same dict format as compute_crop(), with extra key 'ml_crop': True.
    Returns None if the image cannot be processed or fewer than n matching neighbours exist.
    """
    import contextlib
    from .main import (
        apply_orientation, enforce_aspect_ratio, extract_preview_image,
        get_orientation, limit_zoom,
    )

    t_start = time.perf_counter()

    orientation = get_orientation(cr3_path)
    image = apply_orientation(extract_preview_image(cr3_path), orientation)
    w_img, h_img = image.size
    image_inf, inf_scale = _resize_for_inference(image)
    t_io = time.perf_counter()

    lock_ctx = _inference_lock if _inference_lock is not None else contextlib.nullcontext()
    t_lock_start = time.perf_counter()
    with lock_ctx:
        t_models_start = time.perf_counter()
        result = extract_features(image_inf, models, name=cr3_path.name)
    t_inf = time.perf_counter()
    t_lock_wait = t_models_start - t_lock_start
    t_models = t_inf - t_models_start
    if result is None:
        return None

    query_mask, query_fc, query_ar, (mx1, my1, mx2, my2) = result
    # Scale mask bbox from inference resolution back to original image coordinates.
    if inf_scale < 1.0:
        mx1 = int(round(mx1 / inf_scale))
        my1 = int(round(my1 / inf_scale))
        mx2 = int(round(mx2 / inf_scale))
        my2 = int(round(my2 / inf_scale))
    query_norm = _normalize_mask(query_mask)

    candidates, masks_small, face_centroids_arr = _get_ar_candidates(dataset, query_ar)
    if len(candidates) < n:
        print(f"{_YELLOW}  ML [{cr3_path.name}]: only {len(candidates)} training records match this aspect ratio (need {n}){_RESET}")
        return None

    # Vectorised distance computation — no Python loop over training records.
    step = MASK_SIZE // KNN_COMPARE_SIZE
    q_small = query_norm[::step, ::step]  # strided view, no copy
    intersections = (masks_small & q_small).sum(axis=(1, 2)).astype(np.float32)
    unions = (masks_small | q_small).sum(axis=(1, 2)).astype(np.float32)
    ious = np.where(unions > 0, intersections / unions, 1.0)

    if query_fc is None:
        # No face detected — compare on mask shape only
        dists = 1.0 - ious
    else:
        qfc_arr = np.array(query_fc, dtype=np.float32)
        face_dists = np.sqrt(((face_centroids_arr - qfc_arr) ** 2).sum(axis=1))
        dists = dataset.alpha * (1.0 - ious) + (1.0 - dataset.alpha) * face_dists

    top_idx = np.argpartition(dists, n)[:n]
    neighbors = [candidates[i] for i in top_idx]
    t_knn = time.perf_counter()

    print(f"{_DIM}  ML timing [{cr3_path.name}]: io={t_io-t_start:.2f}s  lock_wait={t_lock_wait:.2f}s  models={t_models:.2f}s  knn(n={len(candidates)})={t_knn-t_inf:.2f}s  total={t_knn-t_start:.2f}s{_RESET}")

    pred_cc_x = float(np.mean([r.crop_center[0] for r in neighbors]))
    pred_cc_y = float(np.mean([r.crop_center[1] for r in neighbors]))
    pred_margin = float(np.mean([r.min_margin for r in neighbors]))

    mask_w = float(mx2 - mx1)
    mask_h = float(my2 - my1)

    cx_abs = mx1 + pred_cc_x * mask_w
    cy_abs = my1 + pred_cc_y * mask_h
    half_w = mask_w / 2 + pred_margin * mask_w
    half_h = mask_h / 2 + pred_margin * mask_h

    x1 = cx_abs - half_w
    y1 = cy_abs - half_h
    x2 = cx_abs + half_w
    y2 = cy_abs + half_h

    x1, y1, x2, y2 = enforce_aspect_ratio(x1, y1, x2, y2, w_img, h_img)
    person_cx = float(mx1 + mx2) / 2
    x1, y1, x2, y2 = limit_zoom(x1, y1, x2, y2, w_img, h_img, person_cx)
    x1, y1, x2, y2 = enforce_aspect_ratio(x1, y1, x2, y2, w_img, h_img)

    if (x2 - x1) * (y2 - y1) / (w_img * h_img) > 0.96:
        return False  # noop — matches compute_crop() sentinel so producer counts it correctly

    orig_buf = io.BytesIO()
    image.save(orig_buf, format="JPEG", quality=85)

    crop_buf = io.BytesIO()
    image.crop((int(x1), int(y1), int(x2), int(y2))).save(crop_buf, format="JPEG", quality=85)

    return {
        "cr3_path": cr3_path,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "w": w_img, "h": h_img,
        "raw_x1": float(mx1), "raw_y1": float(my1),
        "raw_x2": float(mx2), "raw_y2": float(my2),
        "person_cx": person_cx,
        "img": image,
        "orig_bytes": orig_buf.getvalue(),
        "crop_bytes": crop_buf.getvalue(),
        "ml_crop": True,
    }
