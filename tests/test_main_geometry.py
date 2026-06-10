"""Tests for pure geometry and XMP functions in main.py.

No models or CR3 files are required — CR3-reading calls are mocked where needed.
"""

import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from autocropper.main import (
    _display_to_sensor_crop,
    _sensor_to_display_crop,
    enforce_aspect_ratio,
    expand_for_instagram_safe_zone,
    expand_with_margin,
    limit_zoom,
    merged_envelope,
    read_xmp_crop,
)

# ── Coordinate transform round-trips ───────────────────────────────────────

ORIENTATIONS = [1, 3, 6, 8]


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_sensor_display_round_trip(orientation):
    """_sensor_to_display_crop must be the exact inverse of _display_to_sensor_crop."""
    original = (0.1, 0.2, 0.8, 0.9)
    sensor = _display_to_sensor_crop(*original, orientation)
    recovered = _sensor_to_display_crop(*sensor, orientation)
    for a, b in zip(original, recovered):
        assert abs(a - b) < 1e-12, f"orientation={orientation}: {original} → {sensor} → {recovered}"


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_display_sensor_round_trip(orientation):
    """_display_to_sensor_crop must be the exact inverse of _sensor_to_display_crop."""
    original = (0.05, 0.15, 0.85, 0.95)
    display = _sensor_to_display_crop(*original, orientation)
    recovered = _display_to_sensor_crop(*display, orientation)
    for a, b in zip(original, recovered):
        assert abs(a - b) < 1e-12, f"orientation={orientation}: {original} → {display} → {recovered}"


def test_identity_orientation():
    """Orientation 1 (no rotation) must be a no-op for both transforms."""
    vals = (0.1, 0.2, 0.8, 0.9)
    assert _display_to_sensor_crop(*vals, 1) == vals
    assert _sensor_to_display_crop(*vals, 1) == vals


# ── XMP crop parsing ───────────────────────────────────────────────────────

def _write_xmp(path, *, has_crop, left=0.1, top=0.2, right=0.8, bottom=0.9, element_form=True):
    if element_form:
        crop_block = (
            f'   <crs:HasCrop>{"True" if has_crop else "False"}</crs:HasCrop>\n'
            f'   <crs:CropLeft>{left:.6f}</crs:CropLeft>\n'
            f'   <crs:CropTop>{top:.6f}</crs:CropTop>\n'
            f'   <crs:CropRight>{right:.6f}</crs:CropRight>\n'
            f'   <crs:CropBottom>{bottom:.6f}</crs:CropBottom>\n'
            f'   <crs:CropAngle>0</crs:CropAngle>\n'
        )
    else:
        # Attribute form as written by Lightroom
        crop_block = (
            f'   crs:HasCrop="{"True" if has_crop else "False"}"\n'
            f'   crs:CropLeft="{left:.6f}"\n'
            f'   crs:CropTop="{top:.6f}"\n'
            f'   crs:CropRight="{right:.6f}"\n'
            f'   crs:CropBottom="{bottom:.6f}"\n'
            f'   crs:CropAngle="0"\n'
        )
    content = (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">\n'
        + crop_block +
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )
    path.write_text(content)


def _make_cr3_with_xmp(tmp_path, **xmp_kwargs):
    """Return a fake CR3 path whose XMP sidecar contains the given crop."""
    cr3 = tmp_path / "IMG_0001.CR3"
    cr3.touch()
    xmp = tmp_path / "IMG_0001.xmp"
    _write_xmp(xmp, **xmp_kwargs)
    return cr3


def _mock_orientation_and_image(orientation=1, width=3000, height=2000):
    """Return a mock image with known dimensions for use in read_xmp_crop patches."""
    img = Image.new("RGB", (width, height))
    return orientation, img


@pytest.mark.parametrize("element_form", [True, False], ids=["element-form", "attr-form"])
def test_read_xmp_crop_identity_orientation(tmp_path, element_form):
    """read_xmp_crop returns correct pixel coords for orientation=1 (no rotation)."""
    w, h = 3000, 2000
    left, top, right, bottom = 0.1, 0.2, 0.8, 0.9
    cr3 = _make_cr3_with_xmp(
        tmp_path, has_crop=True,
        left=left, top=top, right=right, bottom=bottom,
        element_form=element_form,
    )
    mock_img = Image.new("RGB", (w, h))
    with (
        patch("autocropper.main.get_orientation", return_value=1),
        patch("autocropper.main.extract_preview_image", return_value=mock_img),
        patch("autocropper.main.apply_orientation", return_value=mock_img),
    ):
        result = read_xmp_crop(cr3)

    assert result is not None
    x1, y1, x2, y2 = result
    assert abs(x1 - left * w) < 1
    assert abs(y1 - top * h) < 1
    assert abs(x2 - right * w) < 1
    assert abs(y2 - bottom * h) < 1


def test_read_xmp_crop_no_crop_returns_none(tmp_path):
    """read_xmp_crop returns None when HasCrop is False."""
    cr3 = _make_cr3_with_xmp(tmp_path, has_crop=False)
    with (
        patch("autocropper.main.get_orientation", return_value=1),
        patch("autocropper.main.extract_preview_image", return_value=Image.new("RGB", (10, 10))),
        patch("autocropper.main.apply_orientation", return_value=Image.new("RGB", (10, 10))),
    ):
        assert read_xmp_crop(cr3) is None


def test_read_xmp_crop_no_xmp_returns_none(tmp_path):
    """read_xmp_crop returns None when no XMP sidecar exists."""
    cr3 = tmp_path / "IMG_0001.CR3"
    cr3.touch()
    assert read_xmp_crop(cr3) is None


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_read_xmp_crop_rotated_round_trips(tmp_path, orientation):
    """Crops written by write_xmp and read back by read_xmp_crop must match."""
    from autocropper.main import write_xmp

    w, h = 3000, 2000
    x1, y1, x2, y2 = 300, 400, 2700, 1600

    # write_xmp needs the real CR3 path only for get_orientation; mock it
    cr3 = tmp_path / "IMG_0001.CR3"
    cr3.touch()
    mock_img = Image.new("RGB", (w, h))

    with patch("autocropper.main.get_orientation", return_value=orientation):
        write_xmp(cr3, x1, y1, x2, y2, w, h)

    with (
        patch("autocropper.main.get_orientation", return_value=orientation),
        patch("autocropper.main.extract_preview_image", return_value=mock_img),
        patch("autocropper.main.apply_orientation", return_value=mock_img),
    ):
        result = read_xmp_crop(cr3)

    assert result is not None
    rx1, ry1, rx2, ry2 = result
    assert abs(rx1 - x1) < 2, f"x1 mismatch: {rx1} vs {x1}"
    assert abs(ry1 - y1) < 2, f"y1 mismatch: {ry1} vs {y1}"
    assert abs(rx2 - x2) < 2, f"x2 mismatch: {rx2} vs {x2}"
    assert abs(ry2 - y2) < 2, f"y2 mismatch: {ry2} vs {y2}"


# ── Geometry: expand_with_margin ──────────────────────────────────────────

def test_expand_with_margin_basic():
    x1, y1, x2, y2 = expand_with_margin(100, 200, 400, 500, w=1000, h=800, margin_ratio=0.1)
    assert x1 < 100
    assert y1 < 200
    assert x2 > 400
    assert y2 > 500


def test_expand_with_margin_clamps_to_image():
    # Box already at edges; any margin must be clamped to [0, w/h]
    x1, y1, x2, y2 = expand_with_margin(0, 0, 1000, 800, w=1000, h=800, margin_ratio=0.2)
    assert x1 == 0
    assert y1 == 0
    assert x2 == 1000
    assert y2 == 800


def test_expand_with_margin_zero():
    original = (100, 150, 500, 600)
    result = expand_with_margin(*original, w=1000, h=800, margin_ratio=0.0)
    for a, b in zip(original, result):
        assert a == b


# ── Geometry: enforce_aspect_ratio ────────────────────────────────────────

def test_enforce_aspect_ratio_preserves_ratio():
    """Output crop must have the same aspect ratio as the image."""
    w, h = 3000, 2000
    # Start with a crop that is too narrow (would produce a wrong ratio)
    x1, y1, x2, y2 = enforce_aspect_ratio(500, 200, 800, 1000, w, h)
    crop_ratio = (x2 - x1) / (y2 - y1)
    assert abs(crop_ratio - w / h) < 1e-6


def test_enforce_aspect_ratio_stays_in_bounds():
    w, h = 3000, 2000
    x1, y1, x2, y2 = enforce_aspect_ratio(0, 0, 3000, 2000, w, h)
    assert x1 >= 0 and y1 >= 0
    assert x2 <= w and y2 <= h


def test_enforce_aspect_ratio_shifts_not_shrinks():
    """If the expanded crop would go out of bounds it should shift, not shrink."""
    w, h = 3000, 2000
    # Crop near right edge — expansion to the right would overflow
    x1, y1, x2, y2 = enforce_aspect_ratio(2900, 0, 3000, 2000, w, h)
    assert x1 >= 0
    assert x2 <= w


# ── Geometry: limit_zoom ──────────────────────────────────────────────────

def test_limit_zoom_no_change_when_wide_enough():
    """A crop that already covers ≥ MAX_ZOOM fraction should be unchanged."""
    w, h = 3000, 2000
    # Crop covering 60% of width → no change (MAX_ZOOM = 0.5 → threshold is 50%)
    x1, y1, x2, y2 = limit_zoom(300, 0, 2100, 2000, w, h, person_cx=1200)
    assert x1 == 300 and x2 == 2100


def test_limit_zoom_widens_narrow_crop():
    """A very tight crop must be widened."""
    w, h = 3000, 2000
    # Crop covering only 10% of width — should be widened
    orig_w = 300  # 10% of 3000
    x1, y1, x2, y2 = limit_zoom(1350, 0, 1650, 2000, w, h, person_cx=1500)
    assert (x2 - x1) > orig_w


# ── Geometry: merged_envelope ─────────────────────────────────────────────

def test_merged_envelope_single_box():
    import numpy as np
    boxes = [[100.0, 200.0, 400.0, 600.0]]
    hulls = [np.array([[100, 200], [400, 200], [400, 600], [100, 600]])]
    x1, y1, x2, y2 = merged_envelope(boxes, hulls)
    assert x1 == 100 and y1 == 200 and x2 == 400 and y2 == 600


def test_merged_envelope_multiple_boxes():
    import numpy as np
    boxes = [[0, 0, 100, 100], [200, 200, 300, 300]]
    hulls = [
        np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
        np.array([[200, 200], [300, 200], [300, 300], [200, 300]]),
    ]
    x1, y1, x2, y2 = merged_envelope(boxes, hulls)
    assert x1 == 0 and y1 == 0 and x2 == 300 and y2 == 300


# ── Geometry: expand_for_instagram_safe_zone ──────────────────────────────

def test_instagram_safe_zone_portrait_expansion():
    """In portrait mode, a very tall thin box should be widened to maintain the safe zone."""
    w, h = 1000, 1500  # portrait
    # Person is very tall relative to width → need wider crop
    x1, y1, x2, y2 = expand_for_instagram_safe_zone(400, 0, 600, 1500, w, h)
    person_h = 1500
    from autocropper.main import INSTAGRAM_RATIO
    min_w = person_h / INSTAGRAM_RATIO
    assert (x2 - x1) >= min_w - 1
