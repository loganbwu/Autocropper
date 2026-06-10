#!/usr/bin/env python3

import argparse
import contextlib
import io
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rawpy
import torch
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIG ----------------

GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "dancing person."   # Grounding DINO requires a trailing period
CONFIDENCE  = 0.3        # Box and text threshold for Grounding DINO
MARGIN_RATIO = 0.20      # Margin around merged box
INSTAGRAM_RATIO = 5 / 4  # Instagram's widest feed crop (5:4 landscape / 4:5 portrait)
MAX_ZOOM = 0.5           # Don't zoom in more than this fraction of the image width

DEFAULT_ROOT = Path.home() / "Desktop/Test"

# ----------------------------------------


def extract_preview_image(cr3_path: Path) -> Image.Image:
    """Extract the largest embedded JPEG preview from a CR3 file using libraw."""
    with rawpy.imread(str(cr3_path)) as raw:
        thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            return Image.open(io.BytesIO(bytes(thumb.data))).convert("RGB")
        return Image.fromarray(thumb.data).convert("RGB")


def load_models():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading models on {device}...")

    kwargs = {"local_files_only": True}
    try:
        gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL, use_fast=True, **kwargs)
        gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_MODEL, dtype=torch.float16, **kwargs
        ).to(device).eval()
    except Exception:
        # Not cached yet — download and cache
        kwargs = {}
        gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL, use_fast=True)
        gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_MODEL, dtype=torch.float16
        ).to(device).eval()

    return gdino_processor, gdino_model


def load_ml_models():
    """Load all models needed for ML crop mode: Grounding DINO + SAM2 + ViTPose."""
    from transformers import AutoProcessor, AutoModelForMaskGeneration, AutoModel

    gdino_processor, gdino_model = load_models()

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    SAM2_MODEL = "facebook/sam2-hiera-tiny"
    VITPOSE_MODEL = "usyd-community/vitpose-base-simple"

    kwargs = {"local_files_only": True}
    try:
        sam_processor = AutoProcessor.from_pretrained(SAM2_MODEL, **kwargs)
        sam_model = AutoModelForMaskGeneration.from_pretrained(SAM2_MODEL, **kwargs).to(device).eval()
        vitpose_processor = AutoProcessor.from_pretrained(VITPOSE_MODEL, **kwargs)
        vitpose_model = AutoModel.from_pretrained(VITPOSE_MODEL, **kwargs).to(device).eval()
    except Exception:
        sam_processor = AutoProcessor.from_pretrained(SAM2_MODEL)
        sam_model = AutoModelForMaskGeneration.from_pretrained(SAM2_MODEL).to(device).eval()
        vitpose_processor = AutoProcessor.from_pretrained(VITPOSE_MODEL)
        vitpose_model = AutoModel.from_pretrained(VITPOSE_MODEL).to(device).eval()

    print("ML models (SAM2 + ViTPose) loaded.")
    return gdino_processor, gdino_model, sam_processor, sam_model, vitpose_processor, vitpose_model


def detect_people_with_masks(models, image: Image.Image):
    gdino_processor, gdino_model = models
    device = next(gdino_model.parameters()).device

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


def expand_with_margin(x1, y1, x2, y2, w, h, margin_ratio=MARGIN_RATIO):
    bw = x2 - x1
    bh = y2 - y1

    mx = bw * margin_ratio
    my = bh * margin_ratio

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


CROP_TAGS = ('HasCrop', 'CropLeft', 'CropTop', 'CropRight', 'CropBottom', 'CropAngle')


def has_existing_crop(cr3_path: Path):
    """True if an XMP with HasCrop=True exists (crop was applied)."""
    xmp_path = cr3_path.with_suffix("").with_suffix(".xmp")
    if not xmp_path.exists():
        return False
    try:
        content = xmp_path.read_text()
        return bool(re.search(r'crs:HasCrop[=>"\s]*(True|true|1)', content))
    except Exception:
        return False


def has_been_reviewed(cr3_path: Path):
    """True if a review decision has already been recorded (crop applied OR declined)."""
    xmp_path = cr3_path.with_suffix("").with_suffix(".xmp")
    if not xmp_path.exists():
        return False
    try:
        content = xmp_path.read_text()
        return bool(re.search(r'crs:HasCrop[=>"\s]*(True|False|true|false|1|0)', content))
    except Exception:
        return False


def write_decline_marker(cr3_path: Path):
    """Record that the crop was reviewed and declined (HasCrop=False).

    Written so the file is skipped on restart without re-reviewing it.
    Lightroom treats HasCrop=False as 'no crop', which is correct.
    """
    xmp_path = cr3_path.with_suffix("").with_suffix(".xmp")
    decline_tag = '   <crs:HasCrop>False</crs:HasCrop>\n'

    if xmp_path.exists():
        content = xmp_path.read_text()
        for tag in CROP_TAGS:
            content = re.sub(rf'\s*<crs:{tag}>.*?</crs:{tag}>', '', content)
        for tag in CROP_TAGS:
            content = re.sub(rf'\s*crs:{tag}="[^"]*"', '', content)
        last_close = content.rfind('</rdf:Description>')
        if last_close != -1:
            content = content[:last_close] + decline_tag + '  ' + content[last_close:]
        xmp_path.write_text(content)
    else:
        xmp_path.write_text(
            '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
            ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
            '  <rdf:Description rdf:about=""\n'
            '    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">\n'
            + decline_tag +
            '  </rdf:Description>\n'
            ' </rdf:RDF>\n'
            '</x:xmpmeta>\n'
            '<?xpacket end="w"?>'
        )


def _display_to_sensor_crop(left, top, right, bottom, orientation):
    """Rotate crop fractions from display (post-rotation) to sensor (pre-rotation) space.

    Lightroom interprets CropLeft/Top/Right/Bottom relative to the sensor image
    before any EXIF rotation is applied.
    """
    if orientation == 6:    # 90° CW
        return top, 1-right, bottom, 1-left
    elif orientation == 8:  # 90° CCW
        return 1-bottom, left, 1-top, right
    elif orientation == 3:  # 180°
        return 1-right, 1-bottom, 1-left, 1-top
    return left, top, right, bottom


def _sensor_to_display_crop(left, top, right, bottom, orientation):
    """Inverse of _display_to_sensor_crop: sensor space → display space."""
    if orientation == 6:    # 90° CW
        return 1-bottom, left, 1-top, right
    elif orientation == 8:  # 90° CCW
        return top, 1-right, bottom, 1-left
    elif orientation == 3:  # 180°
        return 1-right, 1-bottom, 1-left, 1-top
    return left, top, right, bottom


def read_xmp_crop(cr3_path: Path):
    """Return existing XMP crop as (x1, y1, x2, y2) in display pixels, or None."""
    xmp_path = cr3_path.with_suffix("").with_suffix(".xmp")
    if not xmp_path.exists():
        return None
    try:
        content = xmp_path.read_text()
    except Exception:
        return None

    if not re.search(r'crs:HasCrop[=>"\s]*(True|true|1)', content):
        return None

    def _extract(tag):
        m = re.search(rf'crs:{tag}[>="]*\s*([0-9.]+)', content)
        return float(m.group(1)) if m else None

    vals = [_extract(t) for t in ('CropLeft', 'CropTop', 'CropRight', 'CropBottom')]
    if any(v is None for v in vals):
        return None

    left, top, right, bottom = vals
    orientation = get_orientation(cr3_path)
    left, top, right, bottom = _sensor_to_display_crop(left, top, right, bottom, orientation)

    img = apply_orientation(extract_preview_image(cr3_path), orientation)
    w, h = img.size
    return left * w, top * h, right * w, bottom * h


def write_xmp(cr3_path: Path, x1, y1, x2, y2, w, h):
    xmp_path = cr3_path.with_suffix("").with_suffix(".xmp")

    orientation = get_orientation(cr3_path)
    left, top, right, bottom = _display_to_sensor_crop(
        x1 / w, y1 / h, x2 / w, y2 / h, orientation
    )

    crop_block = (
        f'   <crs:HasCrop>True</crs:HasCrop>\n'
        f'   <crs:CropLeft>{left:.6f}</crs:CropLeft>\n'
        f'   <crs:CropTop>{top:.6f}</crs:CropTop>\n'
        f'   <crs:CropRight>{right:.6f}</crs:CropRight>\n'
        f'   <crs:CropBottom>{bottom:.6f}</crs:CropBottom>\n'
        f'   <crs:CropAngle>0</crs:CropAngle>\n'
    )

    if xmp_path.exists():
        content = xmp_path.read_text()

        # Strip element-form crop tags
        for tag in CROP_TAGS:
            content = re.sub(rf'\s*<crs:{tag}>.*?</crs:{tag}>', '', content)
        # Strip attribute-form crop tags (written by Lightroom)
        for tag in CROP_TAGS:
            content = re.sub(rf'\s*crs:{tag}="[^"]*"', '', content)

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
   <crs:CropAngle>0</crs:CropAngle>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

        xmp_path.write_text(xmp)


def compute_crop(models, cr3_path: Path, all_people: bool = False, _inference_lock=None,
                 margin_ratio: float = MARGIN_RATIO):
    """Compute crop coordinates and return preview image bytes. Returns dict or None.

    _inference_lock: optional threading.Lock to serialise GPU/MPS model calls when
    multiple worker threads are used (concurrent inference corrupts MPS state).
    """
    orientation = get_orientation(cr3_path)
    img = apply_orientation(extract_preview_image(cr3_path), orientation)
    w, h = img.size

    lock_ctx = _inference_lock if _inference_lock is not None else contextlib.nullcontext()
    with lock_ctx:
        boxes, hulls, w, h = detect_people_with_masks(models, img)

    if not boxes and not hulls:
        return None

    if not all_people:
        boxes, hulls = select_main_person(boxes, hulls)

    raw_x1, raw_y1, raw_x2, raw_y2 = merged_envelope(boxes, hulls)
    person_cx = (raw_x1 + raw_x2) / 2

    x1, y1, x2, y2 = _run_geometry(raw_x1, raw_y1, raw_x2, raw_y2, w, h, person_cx, margin_ratio)

    # Skip if the crop is effectively the full frame (no meaningful difference)
    if (x2 - x1) * (y2 - y1) / (w * h) > 0.96:
        return None

    orig_buf = io.BytesIO()
    img.save(orig_buf, format="JPEG", quality=85)

    crop_buf = io.BytesIO()
    img.crop((int(x1), int(y1), int(x2), int(y2))).save(crop_buf, format="JPEG", quality=85)

    return {
        "cr3_path": cr3_path,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "w": w, "h": h,
        "raw_x1": raw_x1, "raw_y1": raw_y1, "raw_x2": raw_x2, "raw_y2": raw_y2,
        "person_cx": person_cx,
        "img": img,
        "orig_bytes": orig_buf.getvalue(),
        "crop_bytes": crop_buf.getvalue(),
    }


def _run_geometry(raw_x1, raw_y1, raw_x2, raw_y2, w, h, person_cx, margin_ratio):
    x1, y1, x2, y2 = expand_with_margin(raw_x1, raw_y1, raw_x2, raw_y2, w, h, margin_ratio)
    x1, y1, x2, y2 = expand_for_instagram_safe_zone(x1, y1, x2, y2, w, h)
    x1, y1, x2, y2 = enforce_aspect_ratio(x1, y1, x2, y2, w, h)
    x1, y1, x2, y2 = limit_zoom(x1, y1, x2, y2, w, h, person_cx)
    x1, y1, x2, y2 = enforce_aspect_ratio(x1, y1, x2, y2, w, h)
    return x1, y1, x2, y2


def recompute_crop(d: dict, margin_ratio: float) -> None:
    """Re-run crop geometry with a new margin, updating d in-place."""
    x1, y1, x2, y2 = _run_geometry(
        d["raw_x1"], d["raw_y1"], d["raw_x2"], d["raw_y2"],
        d["w"], d["h"], d["person_cx"], margin_ratio,
    )
    crop_buf = io.BytesIO()
    d["img"].crop((int(x1), int(y1), int(x2), int(y2))).save(crop_buf, format="JPEG", quality=85)
    d["x1"], d["y1"], d["x2"], d["y2"] = x1, y1, x2, y2
    d["crop_bytes"] = crop_buf.getvalue()


def process_cr3(models, cr3_path: Path, force: bool = False, all_people: bool = False):
    if not force and has_existing_crop(cr3_path):
        return False

    result = compute_crop(models, cr3_path, all_people)
    if result is None:
        return False

    write_xmp(cr3_path, result["x1"], result["y1"], result["x2"], result["y2"], result["w"], result["h"])
    return True


_DTO_TAG        = 36867  # ExifIFD.DateTimeOriginal
_ORIENTATION_TAG = 274   # IFD0.Orientation
_CANON_UUID = bytes.fromhex('85c0b687820f11e08111f4ce462b6a48')

# EXIF Orientation → PIL transpose operation
_ORIENTATION_TO_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def _iter_isobmff_boxes(buf, start, end):
    end = min(end, len(buf))  # don't iterate past the buffer
    off = start
    while off + 8 <= end:
        size = struct.unpack_from('>I', buf, off)[0]
        btype = buf[off + 4:off + 8]
        payload = off + 8
        if size == 1:
            if off + 16 > len(buf):
                break
            size = struct.unpack_from('>Q', buf, off + 8)[0]
            payload = off + 16
        if size == 0:
            size = end - off
        yield btype, payload, off + size
        off += size


def _cr3_cmt_box(data: bytes, box_name: bytes):
    """Extract a named CMT box payload from a Canon CR3 ISOBMFF file."""
    moov_start = moov_end = None
    for btype, s, e in _iter_isobmff_boxes(data, 0, len(data)):
        if btype == b'moov':
            moov_start, moov_end = s, e
            break
    if moov_start is None:
        return None
    for btype, s, e in _iter_isobmff_boxes(data, moov_start, moov_end):
        if btype == b'uuid' and data[s:s + 16] == _CANON_UUID:
            for btype2, s2, e2 in _iter_isobmff_boxes(data, s + 16, e):
                if btype2 == box_name:
                    return data[s2:e2]
    return None


def _read_tiff_tag(tiff: bytes, tag: int):
    """Return the value of a tag from a raw TIFF IFD block."""
    if len(tiff) < 8:
        return None
    endian = '<' if tiff[:2] == b'II' else '>'
    ifd_off = struct.unpack_from(endian + 'I', tiff, 4)[0]
    n = struct.unpack_from(endian + 'H', tiff, ifd_off)[0]
    for i in range(n):
        off = ifd_off + 2 + i * 12
        if off + 12 > len(tiff):
            break
        t, typ, count = struct.unpack_from(endian + 'HHI', tiff, off)
        if t != tag:
            continue
        raw = tiff[off + 8:off + 12]
        if typ == 3 and count == 1:
            return struct.unpack_from(endian + 'H', raw)[0]
        if typ == 4 and count == 1:
            return struct.unpack_from(endian + 'I', raw)[0]
        if typ == 2:
            if count > 4:
                val_off = struct.unpack_from(endian + 'I', raw)[0]
                return tiff[val_off:val_off + count].rstrip(b'\x00').decode('ascii', 'replace')
            return raw[:count].rstrip(b'\x00').decode('ascii', 'replace')
    return None


def _read_cr3_header(cr3_path: Path, max_bytes: int = 12_000_000) -> bytes:
    """Read only the first max_bytes of a CR3 file.

    The moov/CMT boxes appear near the start of Canon CR3 files before the
    large CRAW image-data box. 12 MB covers the moov box even for high-res
    cameras (EOS R3, R5) whose embedded preview images are several MB.
    """
    with open(cr3_path, 'rb') as f:
        return f.read(max_bytes)


def get_capture_time(cr3_path: Path) -> str:
    """Return DateTimeOriginal string ('YYYY:MM:DD HH:MM:SS') or '' on failure."""
    try:
        cmt2 = _cr3_cmt_box(_read_cr3_header(cr3_path), b'CMT2')
        if cmt2 is not None:
            ts = _read_tiff_tag(cmt2, _DTO_TAG)
            if ts:
                return str(ts)
    except Exception:
        pass
    return ''


def get_orientation(cr3_path: Path) -> int:
    """Return EXIF Orientation (1–8) from IFD0/CMT1, or 1 (normal) on failure."""
    try:
        cmt1 = _cr3_cmt_box(_read_cr3_header(cr3_path), b'CMT1')
        if cmt1 is not None:
            val = _read_tiff_tag(cmt1, _ORIENTATION_TAG)
            if val is not None:
                return int(val)
    except Exception:
        pass
    return 1


def apply_orientation(img: Image.Image, orientation: int) -> Image.Image:
    op = _ORIENTATION_TO_TRANSPOSE.get(orientation)
    return img.transpose(op) if op else img


def find_cr3_files(root: Path):
    files = [p for p in root.rglob("*") if p.suffix.lower() == ".cr3"]
    workers = min(8, len(files)) if files else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        times = list(pool.map(get_capture_time, files))
    return [f for _, f in sorted(zip(times, files), key=lambda x: (x[0], x[1].name))]


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

    print(f"Done: {processed} cropped, {skipped} skipped, {no_people} no person detected")


if __name__ == "__main__":
    main()
