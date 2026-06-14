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
    TrainingDataset,
    _face_dist,
    _mask_iou,
    _normalize_mask,
    _yaw_dist,
)


def _cross_val_error(records, alpha, yaw_weight, n):
    """Mean crop-centre prediction error (Euclidean, normalised coords) over all leave-one-out folds."""
    errors = []
    norms = [_normalize_mask(r.mask) for r in records]

    for i, query in enumerate(records):
        candidates = [
            (j, r) for j, r in enumerate(records)
            if j != i and abs(r.aspect_ratio - query.aspect_ratio) <= AR_EPSILON
        ]
        if len(candidates) < n:
            continue

        distances = []
        for j, r in candidates:
            iou = _mask_iou(norms[i], norms[j])
            fd = _face_dist(query.face_centroid, r.face_centroid)
            yd = _yaw_dist(query.face_yaw, r.face_yaw, fallback=fd)
            face_comp = (1.0 - yaw_weight) * fd + yaw_weight * yd
            dist = alpha * (1.0 - iou) + (1.0 - alpha) * face_comp
            distances.append((dist, r))

        distances.sort(key=lambda x: x[0])
        neighbors = [r for _, r in distances[:n]]

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
        err = _cross_val_error(records, alpha, yw, n)
        print(f"  GD iter {iteration:2d}: alpha={alpha:.4f}  yaw_weight={yw:.4f}  error={err:.5f}")
        if abs(prev_err - err) < tol:
            break
        prev_err = err

        # Central differences — clamp nudge to stay in [0, 1]
        a_hi = min(1.0, alpha + h)
        a_lo = max(0.0, alpha - h)
        yw_hi = min(1.0, yw + h)
        yw_lo = max(0.0, yw - h)

        grad_alpha = (_cross_val_error(records, a_hi, yw, n) -
                      _cross_val_error(records, a_lo, yw, n)) / (a_hi - a_lo)
        grad_yw = (_cross_val_error(records, alpha, yw_hi, n) -
                   _cross_val_error(records, alpha, yw_lo, n)) / (yw_hi - yw_lo)

        alpha = max(0.0, min(1.0, alpha - lr * grad_alpha))
        yw = max(0.0, min(1.0, yw - lr * grad_yw))

    final_err = _cross_val_error(records, alpha, yw, n)
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
    with tqdm(total=total, desc="Grid search") as pbar:
        for alpha in alphas:
            for yw in yaw_weights:
                err = _cross_val_error(records, float(alpha), float(yw), args.n)
                grid_results[(float(alpha), float(yw))] = err
                pbar.update(1)

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
