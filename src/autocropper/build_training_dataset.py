"""Build an ML crop training dataset from a folder of already-cropped CR3 files.

Usage:
    build-training-dataset <input_folder> <output.pkl>

For each CR3 file that has a recorded crop (HasCrop=True in its XMP sidecar),
the script runs person detection, SAM2 segmentation, and ViTPose keypoint
estimation, then stores the result as a TrainingRecord.

Files without a crop, with no detectable person, with multiple people, or with
no detectable face are silently skipped.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from .main import (
    apply_orientation,
    extract_preview_image,
    get_orientation,
    has_existing_crop,
    load_ml_models,
    read_xmp_crop,
)
from .ml_crop import TrainingDataset, build_training_record


def main():
    parser = argparse.ArgumentParser(
        description="Build an ML crop training dataset from already-cropped CR3 files"
    )
    parser.add_argument("input_folder", type=Path, help="Folder containing CR3 files")
    parser.add_argument("output", type=Path, help="Output .pkl file for the training dataset")
    args = parser.parse_args()

    root = args.input_folder.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Input folder does not exist: {root}")

    cr3_files = [p for p in root.rglob("*") if p.suffix.lower() == ".cr3"]
    if not cr3_files:
        raise SystemExit(f"No CR3 files found under: {root}")

    print(f"Found {len(cr3_files)} CR3 files.")
    print("Loading ML models (this may take a moment)...")
    models = load_ml_models()

    records = []
    skipped_no_crop = 0
    skipped_no_xmp = 0
    skipped_detection = 0

    for cr3 in tqdm(cr3_files, desc="Building training dataset", unit="image"):
        if not has_existing_crop(cr3):
            skipped_no_crop += 1
            continue

        crop_xyxy = read_xmp_crop(cr3)
        if crop_xyxy is None:
            skipped_no_xmp += 1
            continue

        try:
            orientation = get_orientation(cr3)
            image = apply_orientation(extract_preview_image(cr3), orientation)
            record = build_training_record(image, crop_xyxy, models)
        except Exception as e:
            print(f"  Warning: {cr3.name} — {e}")
            skipped_detection += 1
            continue

        if record is None:
            skipped_detection += 1
            continue

        records.append(record)

    dataset = TrainingDataset(records=records, alpha=0.5)
    dataset.save(args.output)

    print(f"\nDone.")
    print(f"  Records saved:          {len(records)}")
    print(f"  Skipped (no crop XMP):  {skipped_no_crop}")
    print(f"  Skipped (unreadable):   {skipped_no_xmp}")
    print(f"  Skipped (detection):    {skipped_detection}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
