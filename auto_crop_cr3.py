#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# ---------------- CONFIG ----------------

MODEL_NAME = "yolo11n-pose.pt"
CONFIDENCE = 0.1
MARGIN_RATIO = 0.30   # 30% margin around merged box

DEFAULT_ROOT = Path.home() / "Pictures"

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


def detect_people_with_keypoints(model, image_path: Path):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    results = model(str(image_path), conf=CONFIDENCE, verbose=False)[0]

    boxes = []
    keypoints = []

    if results.boxes is not None:
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # person
                boxes.append(box.xyxy[0].cpu().numpy())

    if results.keypoints is not None:
        for kp in results.keypoints.xy:
            kp = kp.cpu().numpy()
            kp = kp[~np.isnan(kp[:, 0])]
            if len(kp):
                keypoints.append(kp)

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



def write_xmp(cr3_path: Path, x1, y1, x2, y2, w, h):
    xmp_path = cr3_path.with_suffix("").with_suffix(".xmp")

    left = x1 / w
    top = y1 / h
    right = x2 / w
    bottom = y2 / h

    # Define namespaces
    namespaces = {
        'x': 'adobe:ns:meta/',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/'
    }

    # Register namespaces for proper serialization
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)

    if xmp_path.exists():
        # Read and parse existing XMP
        tree = ET.parse(xmp_path)
        root = tree.getroot()

        # Find or create the RDF Description element with crs namespace
        rdf = root.find('.//rdf:RDF', namespaces)
        if rdf is None:
            # Create RDF structure if it doesn't exist
            rdf = ET.SubElement(root, f"{{{namespaces['rdf']}}}RDF")
        
        # Find first Description element (there may be multiple)
        desc = rdf.find('.//rdf:Description', namespaces)
        if desc is None:
            # Create new Description element
            desc = ET.SubElement(rdf, f"{{{namespaces['rdf']}}}Description")
            desc.set(f"{{{namespaces['rdf']}}}about", "")
        
        # Ensure crs namespace is declared on Description element
        desc.set(f"{{http://www.w3.org/2000/xmlns/}}crs", namespaces['crs'])

        # Update or create crop tags
        crop_tags = {
            f"{{{namespaces['crs']}}}HasCrop": "True",
            f"{{{namespaces['crs']}}}CropLeft": f"{left:.6f}",
            f"{{{namespaces['crs']}}}CropTop": f"{top:.6f}",
            f"{{{namespaces['crs']}}}CropRight": f"{right:.6f}",
            f"{{{namespaces['crs']}}}CropBottom": f"{bottom:.6f}"
        }

        for tag, value in crop_tags.items():
            elem = desc.find(f".//{tag}", namespaces)
            if elem is not None:
                desc.remove(elem)
            new_elem = ET.SubElement(desc, tag)
            new_elem.text = value

        # Write back with XML declaration and xpacket wrapper
        tree.write(xmp_path, encoding='utf-8', xml_declaration=True)
        
        # Add xpacket processing instructions
        content = xmp_path.read_text()
        if not content.startswith('<?xpacket'):
            content = f'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n{content}'
        if not content.endswith('<?xpacket end="w"?>'):
            content = f'{content.rstrip()}\n<?xpacket end="w"?>'
        xmp_path.write_text(content)

    else:
        # Create new XMP file with crop data (original behavior)
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


def process_cr3(model, cr3_path: Path):
    with tempfile.TemporaryDirectory() as tmp:
        preview = Path(tmp) / "preview.jpg"
        extract_preview_jpeg(cr3_path, preview)

        boxes, keypoints, w, h = detect_people_with_keypoints(model, preview)

        if not boxes and not keypoints:
            return

        x1, y1, x2, y2 = merged_envelope(boxes, keypoints)
        x1, y1, x2, y2 = expand_with_margin(x1, y1, x2, y2, w, h)
        x1, y1, x2, y2 = enforce_aspect_ratio(x1, y1, x2, y2, w, h)

        write_xmp(cr3_path, x1, y1, x2, y2, w, h)


def find_cr3_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".cr3")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-crop CR3 files using YOLO pose keypoints and write Lightroom XMP crops"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_ROOT,
        type=Path,
        help="Folder containing CR3 files (default: ~/Pictures)",
    )

    args = parser.parse_args()
    root = args.path.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    cr3_files = find_cr3_files(root)
    if not cr3_files:
        raise SystemExit(f"No CR3 files found under: {root}")

    model = YOLO(MODEL_NAME)

    for cr3 in tqdm(cr3_files, desc="Auto-cropping CR3s", unit="image"):
        process_cr3(model, cr3)


if __name__ == "__main__":
    main()
