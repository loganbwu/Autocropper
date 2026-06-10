"""Optimise the alpha weighting for ML crop similarity via leave-one-out cross-validation.

Usage:
    optimize-weights <training_dataset.pkl> [--update]

For each value of alpha in [0.0, 0.1, ..., 1.0], the script performs
leave-one-out cross-validation: each record is treated as a query and the
remaining records are used as the training set. The predicted crop centre is
compared against the actual crop centre (in normalised mask-bbox coordinates).
The alpha with the lowest mean error is reported.

If --update is passed, the dataset's alpha is updated in-place and saved back.
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
)


def _cross_val_error(records, alpha, n):
    """Mean crop-centre prediction error (Euclidean, normalised coords) over all leave-one-out folds."""
    errors = []
    # Pre-normalise all masks once to avoid re-scaling on every pairwise comparison
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
            dist = alpha * (1.0 - iou) + (1.0 - alpha) * fd
            distances.append((dist, r))

        distances.sort(key=lambda x: x[0])
        neighbors = [r for _, r in distances[:n]]

        pred_cx = float(np.mean([r.crop_center[0] for r in neighbors]))
        pred_cy = float(np.mean([r.crop_center[1] for r in neighbors]))

        err = np.sqrt((pred_cx - query.crop_center[0]) ** 2 + (pred_cy - query.crop_center[1]) ** 2)
        errors.append(float(err))

    return float(np.mean(errors)) if errors else float('inf')


def main():
    parser = argparse.ArgumentParser(
        description="Optimise alpha weighting for ML crop via leave-one-out cross-validation"
    )
    parser.add_argument("dataset", type=Path, help="Path to training_dataset.pkl")
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N_NEIGHBORS,
        help=f"Number of neighbours (default: {DEFAULT_N_NEIGHBORS})"
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Update the dataset file with the optimal alpha and save it"
    )
    parser.add_argument(
        "--steps", type=int, default=11,
        help="Number of alpha values to test (default: 11, i.e. 0.0, 0.1, ..., 1.0)"
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
    print("Running leave-one-out cross-validation...")

    alphas = np.linspace(0.0, 1.0, args.steps)
    results = []

    for alpha in tqdm(alphas, desc="Testing alpha values"):
        err = _cross_val_error(records, float(alpha), args.n)
        results.append((float(alpha), err))
        print(f"  alpha={alpha:.2f}  mean_error={err:.4f}")

    best_alpha, best_err = min(results, key=lambda x: x[1])
    print(f"\nOptimal alpha: {best_alpha:.2f}  (mean error: {best_err:.4f})")
    print("  alpha=0.0 → all weight on face distance")
    print("  alpha=1.0 → all weight on mask overlap")

    if args.update:
        dataset.alpha = best_alpha
        dataset.save(path)
        print(f"Dataset updated with alpha={best_alpha:.2f} and saved to {path}")


if __name__ == "__main__":
    main()
