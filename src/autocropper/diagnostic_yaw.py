"""Before/after yaw diagnostic: compare k-NN neighbours with and without face yaw.

Usage:
    diagnostic-yaw <input_folder> <dataset.pkl> <output_folder> [--n N] [--yaw-weight YW] [--cols C]

For each query image, produces a composite JPEG with two stacked rows:
  Top row:    query panel + nearest neighbours using yaw_weight=0 (baseline)
  Bottom row: same query panel + nearest neighbours using yaw_weight=X

--yaw-weight overrides the dataset's stored yaw_weight.  If not supplied and
the dataset stores 0, defaults to 0.3 so the comparison is meaningful.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .main import (
    apply_orientation,
    detect_people_with_masks,
    enforce_aspect_ratio,
    extract_preview_image,
    get_orientation,
    load_ml_models,
)
from .ml_crop import (
    AR_EPSILON,
    DEFAULT_N_NEIGHBORS,
    FACE_KP_INDICES,
    FACE_KP_THRESHOLD,
    MASK_SIZE,
    TrainingDataset,
    _get_ar_candidates,
    _normalize_mask,
    _resize_for_inference,
    _run_sam,
    _run_vitpose,
)

# Re-use rendering helpers from diagnostic_knn (no duplication)
from .diagnostic_knn import (
    OVERLAY_ALPHA,
    THUMB_SIZE,
    _draw_label,
    _load_image,
    _mask_content_bbox,
    _neighbour_thumb,
    _query_panel,
)

THUMB_COLS = 5
BANNER_H   = 28   # px height of the coloured section-header banner


# ── k-NN with optional yaw ─────────────────────────────────────────────────────

def _find_neighbours(query_mask, query_fc, query_yaw, query_ar, dataset, n, yaw_weight):
    """Return [(record, distance)] for the n nearest neighbours at the given yaw_weight."""
    from .ml_crop import KNN_COMPARE_SIZE
    candidates, masks_small, face_centroids_arr, face_yaws_arr = _get_ar_candidates(dataset, query_ar)
    if len(candidates) < n:
        return [(r, 0.0) for r in candidates]

    query_norm = _normalize_mask(query_mask)
    step       = MASK_SIZE // KNN_COMPARE_SIZE
    q_small    = query_norm[::step, ::step]

    intersections = (masks_small & q_small).sum(axis=(1, 2)).astype(np.float32)
    unions        = (masks_small | q_small).sum(axis=(1, 2)).astype(np.float32)
    ious          = np.where(unions > 0, intersections / unions, 1.0)

    if query_fc is None:
        dists = 1.0 - ious
    else:
        qfc_arr    = np.array(query_fc, dtype=np.float32)
        face_dists = np.sqrt(((face_centroids_arr - qfc_arr) ** 2).sum(axis=1))
        if yaw_weight > 0 and query_yaw is not None:
            yaw_dists_raw = np.abs(face_yaws_arr - query_yaw) / 2.0
            yaw_dists     = np.where(np.isnan(yaw_dists_raw), face_dists, yaw_dists_raw)
            face_component = (1.0 - yaw_weight) * face_dists + yaw_weight * yaw_dists
        else:
            face_component = face_dists
        dists = dataset.alpha * (1.0 - ious) + (1.0 - dataset.alpha) * face_component

    top_idx = np.argpartition(dists, n)[:n]
    top_idx = top_idx[np.argsort(dists[top_idx])]
    return [(candidates[i], float(dists[i])) for i in top_idx]


# ── Grid layout ────────────────────────────────────────────────────────────────

def _neighbour_grid(neighbours_with_dists, n_cols=THUMB_COLS, size=THUMB_SIZE):
    n      = len(neighbours_with_dists)
    n_rows = math.ceil(n / n_cols)
    grid   = Image.new("RGB", (n_cols * size, n_rows * size), (15, 15, 15))
    for i, (record, dist) in enumerate(neighbours_with_dists):
        thumb = _neighbour_thumb(record, rank=i + 1, distance=dist, size=size)
        col, row = i % n_cols, i // n_cols
        grid.paste(thumb, (col * size, row * size))
    return grid


# ── Section banner ─────────────────────────────────────────────────────────────

def _banner(width, text, colour):
    img  = Image.new("RGB", (width, BANNER_H), colour)
    ImageDraw.Draw(img).text((6, 6), text, fill=(255, 255, 255))
    return img


# ── One diagnostic row ─────────────────────────────────────────────────────────

def _diagnostic_row(image_inf, mask, face_kps, neighbours, mx1, my1, mx2, my2, n_cols):
    """Return (query_panel, neighbour_grid) for one set of neighbours."""
    mw_px = float(mx2 - mx1)
    mh_px = float(my2 - my1)
    w_inf, h_inf = image_inf.size

    centre_trace = []
    pred_crop    = None
    try:
        for k in range(1, len(neighbours) + 1):
            top_k = neighbours[:k]
            cc_x  = float(np.mean([r.crop_center[0] for r, _ in top_k]))
            cc_y  = float(np.mean([r.crop_center[1] for r, _ in top_k]))
            centre_trace.append((mx1 + cc_x * mw_px, my1 + cc_y * mh_px))
        margin   = float(np.mean([r.min_margin for r, _ in neighbours]))
        cx, cy   = centre_trace[-1]
        hw = mw_px / 2 + margin * mw_px
        hh = mh_px / 2 + margin * mh_px
        pred_crop = enforce_aspect_ratio(cx - hw, cy - hh, cx + hw, cy + hh, w_inf, h_inf)
    except Exception:
        pass

    query_img = _query_panel(image_inf, mask, face_kps, pred_crop, centre_trace)
    grid      = _neighbour_grid(neighbours, n_cols=n_cols)
    return query_img, grid


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare k-NN neighbours before and after face-yaw weighting."
    )
    parser.add_argument("input_folder",  type=Path)
    parser.add_argument("dataset",       type=Path, help="Training dataset .pkl")
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--n",    type=int,   default=DEFAULT_N_NEIGHBORS,
                        help=f"Neighbours to show (default {DEFAULT_N_NEIGHBORS})")
    parser.add_argument("--cols", type=int,   default=THUMB_COLS,
                        help=f"Thumbnail columns (default {THUMB_COLS})")
    parser.add_argument("--yaw-weight", type=float, default=None,
                        help="yaw_weight override (default: dataset value, or 0.3 if dataset stores 0)")
    args = parser.parse_args()

    root    = args.input_folder.expanduser().resolve()
    out_dir = args.output_folder.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _CR3_SUFFIXES    = {".cr3"}
    _RASTER_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    suffixes  = _CR3_SUFFIXES | _RASTER_SUFFIXES
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in suffixes]
    if not all_files:
        raise SystemExit(f"No supported image files found under: {root}")

    print(f"Loading training dataset from {args.dataset}...")
    dataset = TrainingDataset.load(args.dataset)
    stored_yw = getattr(dataset, 'yaw_weight', 0.0)
    yaw_weight = args.yaw_weight if args.yaw_weight is not None else (stored_yw or 0.3)
    print(f"  {len(dataset.records)} records, alpha={dataset.alpha:.3f}, "
          f"stored yaw_weight={stored_yw:.3f}, using yaw_weight={yaw_weight:.3f}")

    print("Loading ML models...")
    models = load_ml_models()
    gdino_processor, gdino_model = models[0], models[1]
    sam_processor,   sam_model   = models[2], models[3]
    vp_processor,    vp_model    = models[4], models[5]
    device = next(gdino_model.parameters()).device
    print(f"  Ready on {device}\n")

    for path in all_files:
        print(f"  {path.name}")
        image = _load_image(path)
        if image is None:
            continue

        image_inf, inf_scale = _resize_for_inference(image)
        w_inf, h_inf = image_inf.size

        # Person detection
        try:
            boxes, _, _, _ = detect_people_with_masks(
                (gdino_processor, gdino_model), image_inf
            )
        except Exception as e:
            print(f"    GDINO failed: {e}")
            continue
        if not boxes:
            print(f"    No person detected — skipping")
            continue
        bbox = boxes[int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes]))]
        if len(boxes) > 1:
            print(f"    {len(boxes)} people, using largest")

        # Segmentation
        try:
            with torch.no_grad():
                mask = _run_sam(image_inf, bbox, sam_processor, sam_model, device)
        except Exception as e:
            print(f"    SAM failed: {e}")
            continue

        # Pose / face keypoints + yaw
        face_kps, query_fc, query_yaw = None, None, None
        try:
            with torch.no_grad():
                kps, scores = _run_vitpose(image_inf, bbox, vp_processor, vp_model, device)
            if kps is not None:
                idx = [i for i in FACE_KP_INDICES if scores[i] > FACE_KP_THRESHOLD]
                if idx:
                    face_kps = kps[idx].tolist()
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if rows.any():
                        mx1_f = int(np.where(cols)[0][0]);  mx2_f = int(np.where(cols)[0][-1])
                        my1_f = int(np.where(rows)[0][0]);  my2_f = int(np.where(rows)[0][-1])
                        mw, mh = mx2_f - mx1_f, my2_f - my1_f
                        if mw > 0 and mh > 0:
                            query_fc = (
                                float((np.mean([p[0] for p in face_kps]) - mx1_f) / mw),
                                float((np.mean([p[1] for p in face_kps]) - my1_f) / mh),
                            )
                    # Yaw: require nose + both eyes above threshold
                    if (scores[0] > FACE_KP_THRESHOLD and
                            scores[1] > FACE_KP_THRESHOLD and
                            scores[2] > FACE_KP_THRESHOLD):
                        nose_x       = kps[0, 0]
                        eye_center_x = (kps[1, 0] + kps[2, 0]) / 2.0
                        inter_eye    = abs(kps[1, 0] - kps[2, 0])
                        if inter_eye > 1.0:
                            query_yaw = float(max(-1.0, min(1.0,
                                (nose_x - eye_center_x) / (inter_eye / 2.0))))
        except Exception as e:
            print(f"    ViTPose failed: {e}")

        # Mask bbox
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            print(f"    Empty mask — skipping")
            continue
        mx1 = int(np.where(cols)[0][0]);  mx2 = int(np.where(cols)[0][-1])
        my1 = int(np.where(rows)[0][0]);  my2 = int(np.where(rows)[0][-1])
        mask_crop = mask[my1:my2+1, mx1:mx2+1]

        query_ar = float(image.width) / float(image.height)

        # k-NN: without yaw and with yaw
        nbrs_no_yaw   = _find_neighbours(mask_crop, query_fc, query_yaw, query_ar, dataset, args.n, 0.0)
        nbrs_with_yaw = _find_neighbours(mask_crop, query_fc, query_yaw, query_ar, dataset, args.n, yaw_weight)

        if not nbrs_no_yaw:
            print(f"    No matching training records for AR≈{query_ar:.2f}")
            continue

        yaw_str = f"{query_yaw:+.2f}" if query_yaw is not None else "n/a"
        print(f"    face_yaw={yaw_str}  top_dist_no_yaw={nbrs_no_yaw[0][1]:.3f}  "
              f"top_dist_yaw={nbrs_with_yaw[0][1]:.3f}")

        # ── Render two rows ────────────────────────────────────────────────
        q_no_yaw,   g_no_yaw   = _diagnostic_row(
            image_inf, mask, face_kps, nbrs_no_yaw,   mx1, my1, mx2, my2, args.cols)
        q_with_yaw, g_with_yaw = _diagnostic_row(
            image_inf, mask, face_kps, nbrs_with_yaw, mx1, my1, mx2, my2, args.cols)

        gap      = 6
        row_w    = q_no_yaw.width + gap + g_no_yaw.width
        row_h_no = max(q_no_yaw.height,   g_no_yaw.height)
        row_h_yw = max(q_with_yaw.height, g_with_yaw.height)
        total_h  = BANNER_H + row_h_no + BANNER_H + row_h_yw
        total_w  = max(row_w, q_with_yaw.width + gap + g_with_yaw.width)

        composite = Image.new("RGB", (total_w, total_h), (15, 15, 15))
        y = 0

        # Top row: no yaw
        banner_text = f"Without yaw  (yaw_weight=0)   |   query face_yaw={yaw_str}"
        composite.paste(_banner(total_w, banner_text, (40, 40, 120)), (0, y))
        y += BANNER_H
        composite.paste(q_no_yaw,  (0,                       y))
        composite.paste(g_no_yaw,  (q_no_yaw.width + gap,    y))
        y += row_h_no

        # Bottom row: with yaw
        banner_text = f"With yaw  (yaw_weight={yaw_weight:.2f})   |   query face_yaw={yaw_str}"
        composite.paste(_banner(total_w, banner_text, (40, 90, 60)), (0, y))
        y += BANNER_H
        composite.paste(q_with_yaw, (0,                        y))
        composite.paste(g_with_yaw, (q_with_yaw.width + gap,   y))

        out_path = out_dir / (path.stem + "_yaw_compare.jpg")
        composite.save(out_path, format="JPEG", quality=88)
        print(f"    → {out_path.name}")

    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
