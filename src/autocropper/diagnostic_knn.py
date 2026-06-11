"""K-NN diagnostic: query segmentation alongside nearest training mask neighbours.

Usage:
    diagnostic-knn <input_folder> <dataset.pkl> <output_folder> [--n N]

For each query image, produces a composite JPEG:
  Left panel  — query image with blue/red overlay, green face dot, yellow predicted crop
  Right grid  — n nearest neighbour masks (white silhouette on black), ranked by distance,
                with green face centroid dot, red crop centre dot, and distance label

Training records store only normalised masks (no source images), so neighbours are
displayed as mask silhouettes rather than original photos.
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
    limit_zoom,
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

_CR3_SUFFIXES   = {".cr3"}
_RASTER_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
OVERLAY_ALPHA   = 0.40
THUMB_SIZE      = 220   # px per neighbour thumbnail (square)
THUMB_COLS      = 5     # thumbnails per row


def _load_image(path: Path):
    suffix = path.suffix.lower()
    if suffix in _CR3_SUFFIXES:
        return apply_orientation(extract_preview_image(path), get_orientation(path))
    if suffix in _RASTER_SUFFIXES:
        return Image.open(path).convert("RGB")
    return None


# ── Query panel ────────────────────────────────────────────────────────────────

def _query_panel(image_inf, mask, face_kps, pred_crop_xyxy, centre_trace=None):
    """Render the query image with overlay, face dot, predicted crop box, and centre trace.

    centre_trace: list of (cx, cy) in inference-image pixel coords, ordered k=1..N.
    Drawn as a gradient line from cyan (k=1) to yellow (k=N) so convergence is visible.
    """
    image_np = np.array(image_inf)
    colour   = np.zeros_like(image_np)
    colour[mask]  = [30,  100, 255]
    colour[~mask] = [220,  40,  40]
    blended = (
        image_np.astype(float) * (1 - OVERLAY_ALPHA)
        + colour.astype(float)  * OVERLAY_ALPHA
    ).clip(0, 255).astype(np.uint8)
    img  = Image.fromarray(blended)
    draw = ImageDraw.Draw(img)
    w    = image_inf.width

    if pred_crop_xyxy is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in pred_crop_xyxy)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 220, 0), width=3)

    if centre_trace and len(centre_trace) > 1:
        n = len(centre_trace)
        r = max(3, w // 180)
        # Lines first so dots render on top
        for i in range(len(centre_trace) - 1):
            t  = i / (n - 1)
            cr = int(0   + t * 255)
            cg = int(220 - t * 20)
            cb = int(255 - t * 255)
            x0, y0 = int(round(centre_trace[i][0])),   int(round(centre_trace[i][1]))
            x1, y1 = int(round(centre_trace[i+1][0])), int(round(centre_trace[i+1][1]))
            draw.line([x0, y0, x1, y1], fill=(cr, cg, cb), width=2)
        for i, (cx, cy) in enumerate(centre_trace):
            t  = i / (n - 1)
            cr = int(0   + t * 255)
            cg = int(220 - t * 20)
            cb = int(255 - t * 255)
            cx, cy = int(round(cx)), int(round(cy))
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(cr, cg, cb))
    elif centre_trace and len(centre_trace) == 1:
        r  = max(3, w // 180)
        cx, cy = int(round(centre_trace[0][0])), int(round(centre_trace[0][1]))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0, 220, 255))

    if face_kps is not None:
        r = max(3, w // 150)
        for x, y in face_kps:
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(0, 230, 80))

    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((4, 3), "Query  (cyan=k1 → yellow=kN)", fill=(255, 255, 255))
    return img


# ── Neighbour thumbnail ─────────────────────────────────────────────────────────

def _mask_content_bbox(mask):
    """Return (x1, y1, x2, y2) bounding box of True pixels in mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, 0, mask.shape[1], mask.shape[0]
    y1, y2 = np.where(rows)[0][[0, -1]].tolist()
    x1, x2 = np.where(cols)[0][[0, -1]].tolist()
    return x1, y1, x2, y2


def _draw_label(img, text, alpha=160):
    """Overlay a semi-transparent black bar with white text at the top of img."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([0, 0, img.width, 20], fill=(0, 0, 0, alpha))
    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    ImageDraw.Draw(out).text((3, 3), text, fill=(255, 255, 255))
    return out


def _neighbour_thumb(record, rank, distance, size=THUMB_SIZE):
    """Render a training record as a labelled thumbnail.

    If source_path is set and the file exists, shows the actual image.
    Falls back to a mask silhouette for older records without a path.
    """
    mirrored = getattr(record, "mirrored", False)

    # Attempt to load source image
    source = Path(record.source_path) if getattr(record, "source_path", "") else None
    if source and source.exists():
        try:
            src_img = _load_image(source)
            if src_img is not None:
                if mirrored:
                    src_img = src_img.transpose(Image.FLIP_LEFT_RIGHT)
                src_img.thumbnail((size, size), Image.LANCZOS)
                canvas_img = Image.new("RGB", (size, size), (30, 30, 30))
                x = (size - src_img.width)  // 2
                y = (size - src_img.height) // 2
                canvas_img.paste(src_img, (x, y))
                img = canvas_img
                img = _draw_label(img, f"#{rank}  dist={distance:.3f}")
                return img
        except Exception:
            pass  # fall through to mask rendering

    # Fallback: mask silhouette (record.mask is already flipped for mirrored records)
    mask_small = np.array(
        Image.fromarray(record.mask.astype(np.uint8) * 255)
              .resize((size, size), Image.NEAREST)
    ).astype(bool)

    canvas = np.full((size, size, 3), 30, dtype=np.uint8)
    canvas[mask_small]  = [220, 220, 220]

    img  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    # Locate content region in the scaled mask to position dots correctly.
    mx1, my1, mx2, my2 = _mask_content_bbox(mask_small)
    mw = max(mx2 - mx1, 1)
    mh = max(my2 - my1, 1)
    r  = max(3, size // 60)

    # Green dot: face centroid
    if record.face_centroid is not None:
        fcx = int(mx1 + record.face_centroid[0] * mw)
        fcy = int(my1 + record.face_centroid[1] * mh)
        draw.ellipse([fcx-r, fcy-r, fcx+r, fcy+r], fill=(0, 200, 80))

    # Red dot: crop centre
    ccx = int(mx1 + record.crop_center[0] * mw)
    ccy = int(my1 + record.crop_center[1] * mh)
    draw.ellipse([ccx-r, ccy-r, ccx+r, ccy+r], fill=(220, 60, 60))

    img = _draw_label(img, f"#{rank}  dist={distance:.3f}")
    return img


# ── Grid layout ────────────────────────────────────────────────────────────────

def _neighbour_grid(neighbours_with_dists, n_cols=THUMB_COLS, size=THUMB_SIZE):
    """Arrange neighbour thumbnails into a grid image."""
    n      = len(neighbours_with_dists)
    n_rows = math.ceil(n / n_cols)
    grid   = Image.new("RGB", (n_cols * size, n_rows * size), (15, 15, 15))
    for i, (record, dist) in enumerate(neighbours_with_dists):
        thumb = _neighbour_thumb(record, rank=i+1, distance=dist, size=size)
        col, row = i % n_cols, i // n_cols
        grid.paste(thumb, (col * size, row * size))
    return grid


# ── k-NN distance ──────────────────────────────────────────────────────────────

def _find_neighbours(query_mask, query_fc, query_ar, dataset, n):
    """Return list of (record, distance) for the n nearest neighbours."""
    from .ml_crop import KNN_COMPARE_SIZE
    candidates, masks_small, face_centroids_arr = _get_ar_candidates(dataset, query_ar)
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
        dists      = dataset.alpha * (1.0 - ious) + (1.0 - dataset.alpha) * face_dists

    top_idx = np.argpartition(dists, n)[:n]
    top_idx = top_idx[np.argsort(dists[top_idx])]   # sort by distance ascending
    return [(candidates[i], float(dists[i])) for i in top_idx]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Show query segmentation alongside k-NN training mask neighbours."
    )
    parser.add_argument("input_folder",  type=Path)
    parser.add_argument("dataset",       type=Path, help="Training dataset .pkl")
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--n",    type=int, default=DEFAULT_N_NEIGHBORS,
                        help=f"Neighbours to show (default {DEFAULT_N_NEIGHBORS})")
    parser.add_argument("--cols", type=int, default=THUMB_COLS,
                        help=f"Thumbnail columns (default {THUMB_COLS})")
    args = parser.parse_args()

    root    = args.input_folder.expanduser().resolve()
    out_dir = args.output_folder.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suffixes  = _CR3_SUFFIXES | _RASTER_SUFFIXES
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in suffixes]
    if not all_files:
        raise SystemExit(f"No supported image files found under: {root}")

    print(f"Loading training dataset from {args.dataset}...")
    dataset = TrainingDataset.load(args.dataset)
    print(f"  {len(dataset.records)} records, alpha={dataset.alpha}")

    print("Loading ML models...")
    models = load_ml_models()
    gdino_processor, gdino_model = models[0], models[1]
    sam_processor, sam_model     = models[2], models[3]
    vp_processor,  vp_model      = models[4], models[5]
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

        if len(boxes) > 1:
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            bbox  = boxes[int(np.argmax(areas))]
            print(f"    {len(boxes)} people detected, using largest")
        else:
            bbox = boxes[0]

        # Segmentation
        try:
            with torch.no_grad():
                mask = _run_sam(image_inf, bbox, sam_processor, sam_model, device)
        except Exception as e:
            print(f"    SAM failed: {e}")
            continue

        # Pose / face keypoints
        face_kps, query_fc = None, None
        try:
            with torch.no_grad():
                kps, scores = _run_vitpose(image_inf, bbox, vp_processor, vp_model, device)
            if kps is not None:
                idx = [i for i in FACE_KP_INDICES if scores[i] > FACE_KP_THRESHOLD]
                if idx:
                    face_kps = kps[idx].tolist()
                    # Face centroid normalised within mask bbox
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if rows.any():
                        mx1 = int(np.where(cols)[0][0]);  mx2 = int(np.where(cols)[0][-1])
                        my1 = int(np.where(rows)[0][0]);  my2 = int(np.where(rows)[0][-1])
                        mw, mh = mx2 - mx1, my2 - my1
                        if mw > 0 and mh > 0:
                            fc_x = float((np.mean([p[0] for p in face_kps]) - mx1) / mw)
                            fc_y = float((np.mean([p[1] for p in face_kps]) - my1) / mh)
                            query_fc = (fc_x, fc_y)
        except Exception as e:
            print(f"    ViTPose failed: {e}")

        # Mask bbox (in inference space) → scale back to original image space
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            print(f"    Empty mask — skipping")
            continue
        mx1 = int(np.where(cols)[0][0]);  mx2 = int(np.where(cols)[0][-1])
        my1 = int(np.where(rows)[0][0]);  my2 = int(np.where(rows)[0][-1])
        mask_crop = mask[my1:my2+1, mx1:mx2+1]

        # k-NN
        query_ar = float(image.width) / float(image.height)
        neighbours = _find_neighbours(mask_crop, query_fc, query_ar, dataset, args.n)
        if not neighbours:
            print(f"    No matching training records for AR≈{query_ar:.2f}")
            continue

        # Predicted crop and centre trace (in inference space, for display)
        pred_crop    = None
        centre_trace = []
        mw_px = float(mx2 - mx1)
        mh_px = float(my2 - my1)
        try:
            for k in range(1, len(neighbours) + 1):
                top_k = neighbours[:k]
                cc_x = float(np.mean([r.crop_center[0] for r, _ in top_k]))
                cc_y = float(np.mean([r.crop_center[1] for r, _ in top_k]))
                centre_trace.append((mx1 + cc_x * mw_px, my1 + cc_y * mh_px))
            margin = float(np.mean([r.min_margin for r, _ in neighbours]))
            cx, cy = centre_trace[-1]
            hw = mw_px / 2 + margin * mw_px
            hh = mh_px / 2 + margin * mh_px
            pred_crop = enforce_aspect_ratio(cx-hw, cy-hh, cx+hw, cy+hh, w_inf, h_inf)
        except Exception:
            pass

        # Composite image
        query_img  = _query_panel(image_inf, mask, face_kps, pred_crop, centre_trace)
        neighbour_grid = _neighbour_grid(neighbours, n_cols=args.cols)

        gap = 8
        total_h = max(query_img.height, neighbour_grid.height)
        total_w = query_img.width + gap + neighbour_grid.width
        composite = Image.new("RGB", (total_w, total_h), (15, 15, 15))
        composite.paste(query_img,    (0, 0))
        composite.paste(neighbour_grid, (query_img.width + gap, 0))

        out_path = out_dir / (path.stem + "_knn.jpg")
        composite.save(out_path, format="JPEG", quality=88)
        print(f"    → {out_path.name}  (top dist={neighbours[0][1]:.3f})")

    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
