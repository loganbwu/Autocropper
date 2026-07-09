"""Optimise (alpha, yaw_weight) for ML crop similarity via leave-one-out cross-validation.

Usage:
    optimize-weights <training_dataset.pkl> [--update]

Strategy:
  1. Coarse 2-D grid search over alpha × yaw_weight (6×6 = 36 evaluations).
  2. Gradient descent with central differences from the best grid point.

If --update is passed, the dataset's alpha and yaw_weight are updated in-place and saved.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .ml_crop import (
    AR_EPSILON,
    DEFAULT_N_NEIGHBORS,
    MASK_SIZE,
    KNN_COMPARE_SIZE,
    TrainingDataset,
    _face_dist,
    _normalize_mask,
    _yaw_dist,
)

_STEP = MASK_SIZE // KNN_COMPARE_SIZE


def _cross_val_error(records, alpha, yaw_weight, n, pbar=None):
    """Mean crop-centre prediction error (Euclidean, normalised coords) over all leave-one-out folds.

    pbar: optional tqdm instance to update once per fold.
    """
    errors = []
    # Pre-compute downsampled masks (matches predict_ml_crop vectorised path)
    masks_small = np.array([r.mask[::_STEP, ::_STEP] for r in records], dtype=bool)

    face_centroids = np.array(
        [r.face_centroid if r.face_centroid is not None else (np.nan, np.nan)
         for r in records], dtype=np.float32
    )
    face_yaws = np.array(
        [r.face_yaw if r.face_yaw is not None else np.nan for r in records],
        dtype=np.float32,
    )
    aspect_ratios = np.array([r.aspect_ratio for r in records], dtype=np.float32)

    for i, query in enumerate(records):
        if pbar is not None:
            pbar.update(1)

        mask_match = np.abs(aspect_ratios - query.aspect_ratio) <= AR_EPSILON
        mask_match[i] = False
        cand_idx = np.where(mask_match)[0]
        if len(cand_idx) < n:
            continue

        q_small = masks_small[i]
        c_masks  = masks_small[cand_idx]
        intersections = (c_masks & q_small).sum(axis=(1, 2)).astype(np.float32)
        unions        = (c_masks | q_small).sum(axis=(1, 2)).astype(np.float32)
        ious = np.where(unions > 0, intersections / unions, 1.0)

        if query.face_centroid is None:
            dists = 1.0 - ious
        else:
            qfc = np.array(query.face_centroid, dtype=np.float32)
            face_dists = np.sqrt(((face_centroids[cand_idx] - qfc) ** 2).sum(axis=1))
            if yaw_weight > 0 and query.face_yaw is not None:
                yaw_dists_raw = np.abs(face_yaws[cand_idx] - query.face_yaw) / 2.0
                yaw_dists = np.where(np.isnan(yaw_dists_raw), face_dists, yaw_dists_raw)
                face_comp = (1.0 - yaw_weight) * face_dists + yaw_weight * yaw_dists
            else:
                face_comp = face_dists
            dists = alpha * (1.0 - ious) + (1.0 - alpha) * face_comp

        top_idx = np.argpartition(dists, n)[:n]
        neighbors = [records[cand_idx[k]] for k in top_idx]

        pred_cx = float(np.mean([r.crop_center[0] for r in neighbors]))
        pred_cy = float(np.mean([r.crop_center[1] for r in neighbors]))

        err = np.sqrt((pred_cx - query.crop_center[0]) ** 2 + (pred_cy - query.crop_center[1]) ** 2)
        errors.append(float(err))

    return float(np.mean(errors)) if errors else float('inf')


def _gradient_descent(records, alpha0, yw0, n, lr=0.05, h=0.01, max_iter=50, tol=1e-5):
    """Fine-tune (alpha, yaw_weight) using gradient descent with central differences.

    Returns (alpha, yaw_weight, final_error).
    """
    alpha, yw = float(alpha0), float(yw0)
    prev_err = float('inf')

    for iteration in range(max_iter):
        with tqdm(total=len(records), desc=f"  GD iter {iteration:2d} centre", leave=False) as pb:
            err = _cross_val_error(records, alpha, yw, n, pbar=pb)
        print(f"  GD iter {iteration:2d}: alpha={alpha:.4f}  yaw_weight={yw:.4f}  error={err:.5f}")
        if abs(prev_err - err) < tol:
            break
        prev_err = err

        # Central differences — clamp nudge to stay in [0, 1]
        a_hi = min(1.0, alpha + h)
        a_lo = max(0.0, alpha - h)
        yw_hi = min(1.0, yw + h)
        yw_lo = max(0.0, yw - h)

        with tqdm(total=len(records), desc=f"  GD iter {iteration:2d} grad α+", leave=False) as pb:
            e_a_hi = _cross_val_error(records, a_hi, yw, n, pbar=pb)
        with tqdm(total=len(records), desc=f"  GD iter {iteration:2d} grad α-", leave=False) as pb:
            e_a_lo = _cross_val_error(records, a_lo, yw, n, pbar=pb)
        with tqdm(total=len(records), desc=f"  GD iter {iteration:2d} grad yw+", leave=False) as pb:
            e_yw_hi = _cross_val_error(records, alpha, yw_hi, n, pbar=pb)
        with tqdm(total=len(records), desc=f"  GD iter {iteration:2d} grad yw-", leave=False) as pb:
            e_yw_lo = _cross_val_error(records, alpha, yw_lo, n, pbar=pb)

        grad_alpha = (e_a_hi - e_a_lo) / (a_hi - a_lo)
        grad_yw    = (e_yw_hi - e_yw_lo) / (yw_hi - yw_lo)

        alpha = max(0.0, min(1.0, alpha - lr * grad_alpha))
        yw = max(0.0, min(1.0, yw - lr * grad_yw))

    with tqdm(total=len(records), desc="  GD final eval", leave=False) as pb:
        final_err = _cross_val_error(records, alpha, yw, n, pbar=pb)
    return alpha, yw, final_err


def main():
    parser = argparse.ArgumentParser(
        description="Optimise alpha and yaw_weight for ML crop via leave-one-out cross-validation"
    )
    parser.add_argument("dataset", type=Path, help="Path to training_dataset.pkl")
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N_NEIGHBORS,
        help=f"Number of neighbours (default: {DEFAULT_N_NEIGHBORS})"
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Update the dataset file with optimal alpha and yaw_weight and save it"
    )
    parser.add_argument(
        "--grid-steps", type=int, default=6,
        help="Grid points per axis for coarse search (default: 6, i.e. 0.0, 0.2, ..., 1.0)"
    )
    parser.add_argument(
        "--gd-lr", type=float, default=0.05,
        help="Gradient-descent learning rate (default: 0.05)"
    )
    parser.add_argument(
        "--gd-iter", type=int, default=50,
        help="Max gradient-descent iterations (default: 50)"
    )
    args = parser.parse_args()

    path = args.dataset.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")

    dataset = TrainingDataset.load(path)
    records = dataset.records

    if len(records) < args.n + 1:
        raise SystemExit(
            f"Dataset has only {len(records)} records; need at least {args.n + 1} for leave-one-out with n={args.n}."
        )

    print(f"Dataset: {len(records)} records, n={args.n}")

    # ── Phase 1: coarse 2-D grid search ────────────────────────────────────
    alphas = np.linspace(0.0, 1.0, args.grid_steps)
    yaw_weights = np.linspace(0.0, 1.0, args.grid_steps)
    grid_results = {}

    print(f"\nPhase 1: coarse grid search ({args.grid_steps}×{args.grid_steps} = {args.grid_steps**2} evaluations)")
    total = args.grid_steps ** 2
    done = 0
    with tqdm(total=total, desc="Grid search", position=0) as outer:
        for alpha in alphas:
            for yw in yaw_weights:
                done += 1
                with tqdm(total=len(records),
                          desc=f"  eval {done}/{total}  α={float(alpha):.2f} yw={float(yw):.2f}",
                          position=1, leave=False) as inner:
                    err = _cross_val_error(records, float(alpha), float(yw), args.n, pbar=inner)
                grid_results[(float(alpha), float(yw))] = err
                outer.update(1)

    # Display grid
    print("\n  Grid errors (rows=alpha 0→1, cols=yaw_weight 0→1):")
    header = "alpha \\ yaw_wt  " + "  ".join(f"{yw:.2f}" for yw in yaw_weights)
    print("  " + header)
    for alpha in alphas:
        row = f"  {alpha:.2f}          " + "  ".join(
            f"{grid_results[(float(alpha), float(yw))]:.4f}" for yw in yaw_weights
        )
        print(row)

    best_pair = min(grid_results, key=grid_results.__getitem__)
    best_alpha0, best_yw0 = best_pair
    print(f"\nBest grid point: alpha={best_alpha0:.2f}  yaw_weight={best_yw0:.2f}  error={grid_results[best_pair]:.5f}")

    # ── Phase 2: gradient descent from best grid point ───────────────────
    print(f"\nPhase 2: gradient descent from (alpha={best_alpha0:.2f}, yaw_weight={best_yw0:.2f})")
    opt_alpha, opt_yw, opt_err = _gradient_descent(
        records, best_alpha0, best_yw0, args.n,
        lr=args.gd_lr, max_iter=args.gd_iter,
    )

    print(f"\nOptimal: alpha={opt_alpha:.4f}  yaw_weight={opt_yw:.4f}  mean_error={opt_err:.5f}")
    print("  alpha=0 → all weight on face component; alpha=1 → all weight on mask overlap")
    print("  yaw_weight=0 → face component is centroid distance only; yaw_weight=1 → yaw distance only")

    if args.update:
        dataset.alpha = opt_alpha
        dataset.yaw_weight = opt_yw
        dataset.save(path)
        print(f"\nDataset updated (alpha={opt_alpha:.4f}, yaw_weight={opt_yw:.4f}) and saved to {path}")


if __name__ == "__main__":
    main()
