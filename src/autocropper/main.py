#!/usr/bin/env python3

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIG ----------------

GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "dancing person."   # Grounding DINO requires a trailing period
CONFIDENCE  = 0.3        # Box and text threshold for Grounding DINO
MARGIN_RATIO = 0.10      # Margin around merged box
INSTAGRAM_RATIO = 5 / 4  # Instagram's widest feed crop (5:4 landscape / 4:5 portrait)
MAX_ZOOM = 0.5           # Don't zoom in more than this fraction of the image width

DEFAULT_ROOT = Path.home() / "Desktop/Test"

# ----------------------------------------


def extract_preview_jpeg(cr3_path: Path, out_jpg: Path):
    subprocess.run(
        [
            "exiftool",
            "-b",
            "-PreviewImage",
            str(cr3_path),
        ],
        stdout=open(out_jpg, "wb"),
        stderr=subprocess.DEVNULL,
        check=True,
    )


def load_models():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading models on {device}...")

    gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL, use_fast=True)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL, dtype=torch.float16
    ).to(device).eval()

    return gdino_processor, gdino_model


def detect_people_with_masks(models, image_path: Path):
    gdino_processor, gdino_model = models
    device = next(gdino_model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    gdino_inputs = gdino_processor(images=image, text=TEXT_PROMPT, return_tensors="pt")
    gdino_inputs = {k: v.to(device) for k, v in gdino_inputs.items()}
    device_type = device.type  # "cuda", "mps", or "cpu"
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16):
        gdino_outputs = gdino_model(**gdino_inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        gdino_outputs,
        gdino_inputs["input_ids"],
        threshold=CONFIDENCE,
        text_threshold=CONFIDENCE,
        target_sizes=[(h, w)],
    )

    boxes_xyxy = results[0]["boxes"].cpu().numpy()

    if len(boxes_xyxy) == 0:
        return [], [], w, h

    # Without SAM 2, treat each box's corners as the "hull" for envelope calculation
    hulls = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) for x1, y1, x2, y2 in boxes_xyxy]

    return list(boxes_xyxy), hulls, w, h


def merged_envelope(boxes, hulls):
    xs = []
    ys = []

    for b in boxes:
        x1, y1, x2, y2 = b
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    for pts in hulls:
        if len(pts):
            xs.extend(pts[:, 0])
            ys.extend(pts[:, 1])

    return min(xs), min(ys), max(xs), max(ys)


def expand_with_margin(x1, y1, x2, y2, w, h):
    bw = x2 - x1
    bh = y2 - y1

    mx = bw * MARGIN_RATIO
    my = bh * MARGIN_RATIO

    x1 -= mx
    x2 += mx
    y1 -= my
    y2 += my

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return x1, y1, x2, y2


def expand_for_instagram_safe_zone(x1, y1, x2, y2, w, h):
    """Expand the crop region so the person fits within Instagram's safe zone.

    Instagram center-crops a 3:2 image to 5:4, removing W/12 from each side.
    Instagram center-crops a 2:3 image to 4:5, removing H/12 from top and bottom.
    In both cases the safe zone is 5/6 of the crop's constrained dimension.

    To guarantee the person sits inside the safe zone, we pre-expand the
    bounding box so that the eventual aspect-ratio enforcement produces a crop
    that is large enough: safe-zone width (or height) >= person extent.
    """
    pw = x2 - x1
    ph = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if w >= h:
        # Landscape: safe-zone width = INSTAGRAM_RATIO * crop_h >= pw
        # Requires crop_h >= pw / INSTAGRAM_RATIO.
        # enforce_aspect_ratio will set crop_h to the bounding-box height,
        # so pre-expand height here if needed.
        min_h = pw / INSTAGRAM_RATIO
        if ph < min_h:
            y1 = cy - min_h / 2
            y2 = cy + min_h / 2
    else:
        # Portrait: safe-zone height = INSTAGRAM_RATIO * crop_w >= ph
        min_w = ph / INSTAGRAM_RATIO
        if pw < min_w:
            x1 = cx - min_w / 2
            x2 = cx + min_w / 2

    return x1, y1, x2, y2


def enforce_aspect_ratio(x1, y1, x2, y2, w, h):
    crop_w = x2 - x1
    crop_h = y2 - y1
    target_ratio = w / h
    crop_ratio = crop_w / crop_h

    # Expand to match aspect ratio
    if crop_ratio > target_ratio:
        # Too wide → expand height
        new_h = crop_w / target_ratio
        delta = (new_h - crop_h) / 2
        y1 -= delta
        y2 += delta
    else:
        # Too tall → expand width
        new_w = crop_h * target_ratio
        delta = (new_w - crop_w) / 2
        x1 -= delta
        x2 += delta

    # --- SHIFT, DON'T SHRINK ---
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= (x2 - w)
        x2 = w
    if y2 > h:
        y1 -= (y2 - h)
        y2 = h

    # Final safety clamp (no geometry change now)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return x1, y1, x2, y2



def limit_zoom(x1, y1, x2, y2, w, h, person_cx):
    """If crop is zoomed in more than MAX_ZOOM, widen to the minimum needed to centre the person."""
    crop_w = x2 - x1
    if crop_w / w >= (1 - MAX_ZOOM):
        return x1, y1, x2, y2

    # Widest crop that can be centred on person_cx within the frame
    min_w = 2 * min(person_cx, w - person_cx)
    new_w = max(crop_w, min_w)

    new_x1 = person_cx - new_w / 2
    new_x2 = person_cx + new_w / 2

    if new_x1 < 0:
        new_x2 -= new_x1
        new_x1 = 0
    if new_x2 > w:
        new_x1 -= (new_x2 - w)
        new_x2 = w

    return new_x1, y1, new_x2, y2


def select_main_person(boxes, keypoints):
    """Select only the largest detected person by bounding box area."""
    if not boxes:
        return boxes, keypoints
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    main_kps = [keypoints[idx]] if idx < len(keypoints) else []
    return [boxes[idx]], main_kps


CROP_TAGS = ('HasCrop', 'CropLeft', 'CropTop', 'CropRight', 'CropBottom')


def has_existing_crop(cr3_path: Path):
    """Check if XMP file exists and already has crop data"""
    xmp_path = cr3_path.with_suffix("").with_suffix(".xmp")

    if not xmp_path.exists():
        return False

    try:
        content = xmp_path.read_text()
        # Match both element form (<crs:HasCrop>True</crs:HasCrop>)
        # and attribute form (crs:HasCrop="True") written by Lightroom
        return bool(re.search(r'crs:HasCrop[=>"\s]*(True|true|1)', content))
    except Exception:
        return False


def write_xmp(cr3_path: Path, x1, y1, x2, y2, w, h):
    xmp_path = cr3_path.with_suffix("").with_suffix(".xmp")

    left = x1 / w
    top = y1 / h
    right = x2 / w
    bottom = y2 / h

    crop_block = (
        f'   <crs:HasCrop>True</crs:HasCrop>\n'
        f'   <crs:CropLeft>{left:.6f}</crs:CropLeft>\n'
        f'   <crs:CropTop>{top:.6f}</crs:CropTop>\n'
        f'   <crs:CropRight>{right:.6f}</crs:CropRight>\n'
        f'   <crs:CropBottom>{bottom:.6f}</crs:CropBottom>\n'
    )

    if xmp_path.exists():
        content = xmp_path.read_text()

        # Strip all existing crop element tags from anywhere in the document
        # (handles duplicates inserted by previous buggy runs)
        for tag in CROP_TAGS:
            content = re.sub(rf'\s*<crs:{tag}>.*?</crs:{tag}>', '', content)

        # Insert crop block once, before the last </rdf:Description> (top-level block)
        last_close = content.rfind('</rdf:Description>')
        if last_close != -1:
            content = content[:last_close] + crop_block + '  ' + content[last_close:]

        xmp_path.write_text(content)

    else:
        # Create new XMP file with crop data
        xmp = f"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
   <crs:HasCrop>True</crs:HasCrop>
   <crs:CropLeft>{left:.6f}</crs:CropLeft>
   <crs:CropTop>{top:.6f}</crs:CropTop>
   <crs:CropRight>{right:.6f}</crs:CropRight>
   <crs:CropBottom>{bottom:.6f}</crs:CropBottom>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

        xmp_path.write_text(xmp)


def process_cr3(models, cr3_path: Path, force: bool = False, all_people: bool = False):
    # Skip if already has a crop (unless force flag is set)
    if not force and has_existing_crop(cr3_path):
        return False

    with tempfile.TemporaryDirectory() as tmp:
        preview = Path(tmp) / "preview.jpg"
        extract_preview_jpeg(cr3_path, preview)

        boxes, hulls, w, h = detect_people_with_masks(models, preview)

        if not boxes and not hulls:
            return False

        if not all_people:
            boxes, hulls = select_main_person(boxes, hulls)

        x1, y1, x2, y2 = merged_envelope(boxes, hulls)
        person_cx = (x1 + x2) / 2
        x1, y1, x2, y2 = expand_with_margin(x1, y1, x2, y2, w, h)
        x1, y1, x2, y2 = expand_for_instagram_safe_zone(x1, y1, x2, y2, w, h)
        x1, y1, x2, y2 = enforce_aspect_ratio(x1, y1, x2, y2, w, h)
        x1, y1, x2, y2 = limit_zoom(x1, y1, x2, y2, w, h, person_cx)
        x1, y1, x2, y2 = enforce_aspect_ratio(x1, y1, x2, y2, w, h)

        write_xmp(cr3_path, x1, y1, x2, y2, w, h)
        return True


def find_cr3_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".cr3")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-crop CR3 files using Grounded SAM 2 segmentation and write Lightroom XMP crops"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_ROOT,
        type=Path,
        help="Folder containing CR3 files (default: ~/Pictures)",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force re-crop even if XMP already has crop data",
    )
    parser.add_argument(
        "-a", "--all-people",
        action="store_true",
        help="Crop to include all detected people (default: crop to main person only)",
    )

    args = parser.parse_args()
    root = args.path.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    cr3_files = find_cr3_files(root)
    if not cr3_files:
        raise SystemExit(f"No CR3 files found under: {root}")

    models = load_models()

    processed = 0
    skipped = 0
    no_people = 0

    for cr3 in tqdm(cr3_files, desc="Auto-cropping CR3s", unit="image"):
        if not args.force and has_existing_crop(cr3):
            skipped += 1
            continue

        result = process_cr3(models, cr3, force=args.force, all_people=args.all_people)
        if result:
            processed += 1
        else:
            no_people += 1

    print(f"\n✓ Processed: {processed}")
    if skipped > 0:
        print(f"⊘ Skipped (already cropped): {skipped}")
    if no_people > 0:
        print(f"⊘ Skipped (no people detected): {no_people}")
    if skipped > 0:
        print(f"\nTip: Use --force to re-crop files that already have crops")


if __name__ == "__main__":
    main()
