"""Build an ML crop training dataset from a folder of already-cropped images.

Usage:
    build-training-dataset <input_folder> <output.pkl>

Accepts two kinds of training images:

  CR3 + XMP sidecar (HasCrop=True)
      The crop coordinates come from the XMP file.  Only files whose XMP
      records a positive crop decision are used.

  JPEG / PNG / TIFF (and other PIL-readable formats)
      The entire image is treated as the crop — i.e. delivered images that
      have already been exported at the desired framing are used directly.

For each image the script runs person detection (Grounding DINO), segmentation
(SAM2), and pose estimation (ViTPose), then stores the result as a
TrainingRecord.  Images with no detectable person, multiple people, or no
detectable face are silently skipped.
"""

import argparse
from pathlib import Path

from PIL import Image
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

_CR3_SUFFIXES = {".cr3"}
_JPEG_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _load_image_and_crop(path: Path):
    """Return (PIL image, (x1, y1, x2, y2) crop) for a supported file, or None.

    For CR3 files the crop comes from the XMP sidecar.
    For raster images the entire image frame is the crop.
    Returns None if the file should be skipped.
    """
    suffix = path.suffix.lower()

    if suffix in _CR3_SUFFIXES:
        if not has_existing_crop(path):
            return None, "no_crop"
        crop = read_xmp_crop(path)
        if crop is None:
            return None, "no_xmp"
        orientation = get_orientation(path)
        image = apply_orientation(extract_preview_image(path), orientation)
        return image, crop

    if suffix in _JPEG_SUFFIXES:
        image = Image.open(path).convert("RGB")
        w, h = image.size
        return image, (0.0, 0.0, float(w), float(h))

    return None, "unsupported"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build an ML crop training dataset from already-cropped images. "
            "Accepts CR3 files (crop from XMP sidecar) and JPEG/PNG files "
            "(full frame treated as the crop)."
        )
    )
    parser.add_argument("input_folder", type=Path, help="Folder to scan recursively")
    parser.add_argument("output", type=Path, help="Output .pkl file for the training dataset")
    args = parser.parse_args()

    root = args.input_folder.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Input folder does not exist: {root}")

    all_suffixes = _CR3_SUFFIXES | _JPEG_SUFFIXES
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in all_suffixes]
    if not all_files:
        raise SystemExit(f"No supported image files found under: {root}")

    cr3_count = sum(1 for p in all_files if p.suffix.lower() in _CR3_SUFFIXES)
    jpg_count = len(all_files) - cr3_count
    print(f"Found {len(all_files)} files ({cr3_count} CR3, {jpg_count} raster).")
    print("Loading ML models (this may take a moment)...")
    models = load_ml_models()

    records = []
    skipped_no_crop = 0
    skipped_no_xmp = 0
    skipped_detection = 0
    skipped_unsupported = 0

    for path in tqdm(all_files, desc="Building training dataset", unit="image"):
        try:
            image, crop = _load_image_and_crop(path)
        except Exception as e:
            print(f"  Warning: {path.name} — could not load: {e}")
            skipped_detection += 1
            continue

        if image is None:
            if crop == "no_crop":
                skipped_no_crop += 1
            elif crop == "no_xmp":
                skipped_no_xmp += 1
            else:
                skipped_unsupported += 1
            continue

        try:
            record = build_training_record(image, crop, models)
        except Exception as e:
            print(f"  Warning: {path.name} — detection error: {e}")
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
    if skipped_unsupported:
        print(f"  Skipped (unsupported):  {skipped_unsupported}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
