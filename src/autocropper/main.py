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

DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"
POSE_MODEL = "usyd-community/vitpose-base-simple"
CONFIDENCE = 0.3        # Person detection confidence threshold
KEYPOINT_SCORE = 0.3    # Minimum keypoint confidence to include
MARGIN_RATIO = 0.30     # 30% margin around merged box

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
    from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading models on {device}...")

    det_processor = AutoProcessor.from_pretrained(DETECTOR_MODEL)
    det_model = RTDetrForObjectDetection.from_pretrained(DETECTOR_MODEL).to(device).eval()

    pose_processor = AutoProcessor.from_pretrained(POSE_MODEL)
    pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL).to(device).eval()

    return det_processor, det_model, pose_processor, pose_model


def detect_people_with_keypoints(models, image_path: Path):
    det_processor, det_model, pose_processor, pose_model = models
    device = next(det_model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # Stage 1: Person detection via RT-DETR
    det_inputs = det_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        det_outputs = det_model(**det_inputs)

    det_results = det_processor.post_process_object_detection(
        det_outputs,
        target_sizes=torch.tensor([(h, w)]),
        threshold=CONFIDENCE,
    )

    person_label = next(
        k for k, v in det_model.config.id2label.items() if v.lower() == "person"
    )
    mask = det_results[0]["labels"] == person_label
    boxes_xyxy = det_results[0]["boxes"][mask].cpu().numpy()

    if len(boxes_xyxy) == 0:
        return [], [], w, h

    # Convert VOC (x1,y1,x2,y2) → COCO (x1,y1,w,h) for ViTPose processor
    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]

    # Stage 2: Pose estimation via ViTPose
    pose_inputs = pose_processor(image, boxes=[boxes_xywh], return_tensors="pt")
    pose_inputs = {k: v.to(device) for k, v in pose_inputs.items()}
    with torch.no_grad():
        pose_outputs = pose_model(**pose_inputs)

    pose_results = pose_processor.post_process_pose_estimation(
        pose_outputs, boxes=[boxes_xywh]
    )

    boxes = list(boxes_xyxy)
    keypoints = []
    for person in pose_results[0]:
        kps = person["keypoints"].cpu().numpy()    # (17, 2) pixel coords
        scores = person["scores"].cpu().numpy()     # (17,) confidence
        kps = kps[scores > KEYPOINT_SCORE]
        keypoints.append(kps if len(kps) else np.zeros((0, 2)))

    return boxes, keypoints, w, h


def merged_envelope(boxes, keypoints):
    xs = []
    ys = []

    for b in boxes:
        x1, y1, x2, y2 = b
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    for kp in keypoints:
        xs.extend(kp[:, 0])
        ys.extend(kp[:, 1])

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

        boxes, keypoints, w, h = detect_people_with_keypoints(models, preview)

        if not boxes and not keypoints:
            return False

        if not all_people:
            boxes, keypoints = select_main_person(boxes, keypoints)

        x1, y1, x2, y2 = merged_envelope(boxes, keypoints)
        x1, y1, x2, y2 = expand_with_margin(x1, y1, x2, y2, w, h)
        x1, y1, x2, y2 = enforce_aspect_ratio(x1, y1, x2, y2, w, h)

        write_xmp(cr3_path, x1, y1, x2, y2, w, h)
        return True


def find_cr3_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".cr3")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-crop CR3 files using ViTPose keypoints and write Lightroom XMP crops"
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
