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
(SAM), and pose estimation (ViTPose), then stores the result as a
TrainingRecord.  Images with no detectable person, multiple people, or no
detectable face are silently skipped.
"""

import argparse
from collections import Counter
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
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore an existing output file and start from scratch")
    parser.add_argument("--checkpoint-every", type=int, default=20, metavar="N",
                        help="Save progress every N images (default: 20)")
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

    # Resume from an existing output file unless --fresh was given.
    if args.output.exists() and not args.fresh:
        dataset = TrainingDataset.load(args.output, augment_mirrors=False)
        if not hasattr(dataset, "processed_paths"):  # datasets saved before resume support
            dataset.processed_paths = []
        done = set(dataset.processed_paths) | {r.source_path for r in dataset.records if r.source_path}
        dataset.processed_paths = sorted(done)
        pending = [p for p in all_files if str(p.resolve()) not in done]
        print(f"Resuming: {len(dataset.records)} records in {args.output}, "
              f"{len(all_files) - len(pending)} files already processed, {len(pending)} remaining.")
    else:
        dataset = TrainingDataset(alpha=0.5)
        pending = all_files

    if not pending:
        print("Nothing to do.")
        return

    print("Loading ML models (this may take a moment)...")
    models = load_ml_models()

    records = dataset.records
    counts = Counter()

    try:
        for i, path in enumerate(tqdm(pending, desc="Building training dataset", unit="image"), 1):
            counts[_process_file(path, models, records)] += 1
            dataset.processed_paths.append(str(path.resolve()))
            if i % args.checkpoint_every == 0:
                dataset.save(args.output)
    except KeyboardInterrupt:
        dataset.save(args.output)
        print(f"\nInterrupted. Progress saved to {args.output} "
              f"({len(records)} records). Re-run the same command to resume.")
        raise SystemExit(130)

    dataset.save(args.output)

    print(f"\nDone.")
    print(f"  New records this run:   {counts['ok']}")
    print(f"  Total records:          {len(records)}")
    print(f"  Skipped (no crop XMP):  {counts['no_crop']}")
    print(f"  Skipped (unreadable):   {counts['no_xmp']}")
    print(f"  Skipped (detection):    {counts['detection']}")
    if counts["unsupported"]:
        print(f"  Skipped (unsupported):  {counts['unsupported']}")
    print(f"  Output: {args.output}")


def _process_file(path: Path, models, records: list) -> str:
    """Run detection on one file, appending a record on success.

    Returns "ok" or the skip reason: "no_crop", "no_xmp", "unsupported", "detection".
    """
    try:
        image, crop = _load_image_and_crop(path)
    except Exception as e:
        print(f"  Warning: {path.name} — could not load: {e}")
        return "detection"

    if image is None:
        return crop

    try:
        record = build_training_record(image, crop, models, name=path.name,
                                       source_path=str(path.resolve()))
    except Exception as e:
        print(f"  Warning: {path.name} — detection error: {e}")
        return "detection"

    if record is None:
        return "detection"

    records.append(record)
    return "ok"


if __name__ == "__main__":
    main()
