"""Tests for web.py — filmstrip thumbnail endpoint and related state.

No real CR3 files or AI models are required; image extraction is mocked.
"""

import io
import queue
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from autocropper.web import ReviewState, create_app


@contextmanager
def _mock_extraction(source_img, orientation=1):
    """Patch the three image-reading calls used by get_thumbnail."""
    with patch("autocropper.main.extract_preview_image", return_value=source_img), \
         patch("autocropper.main.get_orientation", return_value=orientation), \
         patch("autocropper.main.apply_orientation", side_effect=lambda img, ori: img):
        yield


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_rgb_image(w=180, h=270, color=(80, 120, 160)):
    return Image.new("RGB", (w, h), color=color)


def _minimal_state(files):
    """Return a ReviewState with __init__ bypassed — only thumbnail-related attrs set."""
    state = object.__new__(ReviewState)
    state.files = list(files)
    state._file_index = {f: i for i, f in enumerate(state.files)}
    state._thumb_cache = {}
    state._lock = threading.Lock()
    state._prefetch_q = queue.Queue()
    state.status = "loading"
    state.current = None
    state.total_eligible = len(state.files)
    state.accepted = 0
    state.rejected = 0
    state.pre_skipped = 0
    state.no_person_skipped = 0
    state.noop_skipped = 0
    state.producer_processed = 0
    state.margin = 0.20
    state.prefetch = 10
    state.ml_mode = False
    state.ml_dataset = None
    return state


@pytest.fixture
def app():
    a = create_app()
    a.config["TESTING"] = True
    return a


@pytest.fixture
def client(app):
    return app.test_client()


# ── /api/thumbnail — HTTP layer ───────────────────────────────────────────────

def test_thumbnail_no_session_returns_404(client):
    """Endpoint returns 404 when no review session is active."""
    response = client.get("/api/thumbnail/0")
    assert response.status_code == 404


def test_thumbnail_negative_index_returns_404(app, client):
    """Flask routing converts negative path segments differently; ensure no crash."""
    # Flask won't even match <int:file_idx> for negative values in most versions,
    # but guard against any edge case.
    response = client.get("/api/thumbnail/-1")
    assert response.status_code in (404, 405)


def test_thumbnail_out_of_range_returns_404(app, client):
    files = [Path("/fake/a.cr3"), Path("/fake/b.cr3")]
    state = _minimal_state(files)
    state.get_thumbnail = MagicMock(return_value=None)
    app.config["review_state"] = state

    response = client.get("/api/thumbnail/5")
    assert response.status_code == 404
    state.get_thumbnail.assert_not_called()


def test_thumbnail_valid_returns_jpeg(app, client):
    """Valid file_idx with a successful thumbnail returns 200 image/jpeg."""
    img = _make_rgb_image(60, 90)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    jpeg_bytes = buf.getvalue()

    files = [Path("/fake/a.cr3"), Path("/fake/b.cr3")]
    state = _minimal_state(files)
    state.get_thumbnail = MagicMock(return_value=jpeg_bytes)
    app.config["review_state"] = state

    response = client.get("/api/thumbnail/1")
    assert response.status_code == 200
    assert response.content_type == "image/jpeg"
    assert response.data == jpeg_bytes
    state.get_thumbnail.assert_called_once_with(1)


def test_thumbnail_extraction_failure_returns_404(app, client):
    """If get_thumbnail returns None (e.g. corrupt file), respond 404."""
    files = [Path("/fake/a.cr3")]
    state = _minimal_state(files)
    state.get_thumbnail = MagicMock(return_value=None)
    app.config["review_state"] = state

    response = client.get("/api/thumbnail/0")
    assert response.status_code == 404


# ── ReviewState.get_thumbnail ─────────────────────────────────────────────────

def test_get_thumbnail_resizes_to_90px_height(tmp_path):
    """Thumbnail must be exactly 90 px tall regardless of source dimensions."""
    cr3 = tmp_path / "photo.cr3"
    cr3.touch()
    state = _minimal_state([cr3])

    source = _make_rgb_image(w=4000, h=3000)
    with _mock_extraction(source):
        result = state.get_thumbnail(0)

    assert result is not None
    out = Image.open(io.BytesIO(result))
    assert out.height == 90


def test_get_thumbnail_preserves_aspect_ratio(tmp_path):
    """Width should scale proportionally with the 90 px height."""
    cr3 = tmp_path / "photo.cr3"
    cr3.touch()
    state = _minimal_state([cr3])

    source = _make_rgb_image(w=3000, h=2000)  # 3:2 landscape
    with _mock_extraction(source):
        result = state.get_thumbnail(0)

    out = Image.open(io.BytesIO(result))
    assert out.height == 90
    expected_w = round(3000 * 90 / 2000)
    assert abs(out.width - expected_w) <= 1  # allow 1px rounding


def test_get_thumbnail_is_cached(tmp_path):
    """Second call must return the same bytes without re-extracting."""
    cr3 = tmp_path / "photo.cr3"
    cr3.touch()
    state = _minimal_state([cr3])

    source = _make_rgb_image()
    with patch("autocropper.main.extract_preview_image", return_value=source) as mock_extract, \
         patch("autocropper.main.get_orientation", return_value=1), \
         patch("autocropper.main.apply_orientation", side_effect=lambda img, ori: img):
        result1 = state.get_thumbnail(0)
        result2 = state.get_thumbnail(0)

    assert result1 is result2
    assert mock_extract.call_count == 1


def test_get_thumbnail_out_of_range_returns_none(tmp_path):
    """Index beyond files list must return None without raising."""
    cr3 = tmp_path / "photo.cr3"
    cr3.touch()
    state = _minimal_state([cr3])

    result = state.get_thumbnail(99)
    assert result is None


def test_get_thumbnail_extract_failure_returns_none(tmp_path):
    """Exception in extract_preview_image must be caught; return None."""
    cr3 = tmp_path / "photo.cr3"
    cr3.touch()
    state = _minimal_state([cr3])

    with patch("autocropper.main.extract_preview_image", side_effect=OSError("no preview")), \
         patch("autocropper.main.get_orientation", return_value=1):
        result = state.get_thumbnail(0)

    assert result is None


def test_get_thumbnail_failed_not_cached(tmp_path):
    """A failed extraction must not poison the cache — next call should retry."""
    cr3 = tmp_path / "photo.cr3"
    cr3.touch()
    state = _minimal_state([cr3])

    with patch("autocropper.main.extract_preview_image", side_effect=OSError("no preview")), \
         patch("autocropper.main.get_orientation", return_value=1):
        result1 = state.get_thumbnail(0)

    assert result1 is None
    assert 0 not in state._thumb_cache  # failure must not be cached


def test_get_thumbnail_returns_valid_jpeg(tmp_path):
    """Output must be a decodable JPEG."""
    cr3 = tmp_path / "photo.cr3"
    cr3.touch()
    state = _minimal_state([cr3])

    source = _make_rgb_image(w=200, h=300)
    with _mock_extraction(source):
        result = state.get_thumbnail(0)

    assert result is not None
    assert result[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_get_thumbnail_applies_orientation(tmp_path):
    """apply_orientation must be called with the file's EXIF orientation value."""
    cr3 = tmp_path / "photo.cr3"
    cr3.touch()
    state = _minimal_state([cr3])

    source = _make_rgb_image(w=300, h=200)
    with patch("autocropper.main.extract_preview_image", return_value=source), \
         patch("autocropper.main.get_orientation", return_value=6) as mock_orient, \
         patch("autocropper.main.apply_orientation", side_effect=lambda img, ori: img) as mock_apply:
        state.get_thumbnail(0)

    mock_orient.assert_called_once_with(cr3)
    mock_apply.assert_called_once_with(source, 6)


# ── get_state includes filmstrip fields ───────────────────────────────────────

def test_get_state_loading_includes_total_files():
    """Loading state must expose total_files for the filmstrip to pre-build."""
    files = [Path(f"/fake/{i}.cr3") for i in range(7)]
    state = _minimal_state(files)

    result = state.get_state()
    assert result["total_files"] == 7


def test_get_state_ready_includes_file_idx_and_total_files():
    """Ready state must expose file_idx so the filmstrip knows which slot to highlight."""
    cr3 = Path("/fake/0.cr3")
    files = [cr3, Path("/fake/1.cr3"), Path("/fake/2.cr3")]
    state = _minimal_state(files)
    state.status = "ready"
    state.current = {
        "cr3_path": cr3,
        "x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0,
        "w": 100, "h": 100,
        "file_idx": 0,
        "orig_bytes": b"\xff\xd8\xff\xe0" + b"\x00" * 16,  # minimal JPEG stub
        "ml_crop": False,
    }

    result = state.get_state()
    assert result["file_idx"] == 0
    assert result["total_files"] == 3


def test_get_state_ready_file_idx_reflects_position():
    """file_idx must match the file's position in state.files, not the review counter."""
    cr3_a = Path("/fake/a.cr3")
    cr3_b = Path("/fake/b.cr3")
    cr3_c = Path("/fake/c.cr3")
    files = [cr3_a, cr3_b, cr3_c]
    state = _minimal_state(files)
    state.status = "ready"
    state.accepted = 1  # one image already reviewed
    state.current = {
        "cr3_path": cr3_c,   # third file in the list
        "x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0,
        "w": 100, "h": 100,
        "file_idx": 2,        # 0-based index in state.files
        "orig_bytes": b"\xff\xd8\xff\xe0" + b"\x00" * 16,
        "ml_crop": False,
    }

    result = state.get_state()
    assert result["file_idx"] == 2  # third file → index 2


# ── file_idx tracking in _file_index ─────────────────────────────────────────

def test_file_index_built_correctly():
    """_file_index must map each file path to its 0-based position in files."""
    files = [Path(f"/fake/{i}.cr3") for i in range(5)]
    state = _minimal_state(files)

    for i, f in enumerate(files):
        assert state._file_index[f] == i


def test_file_index_lookup_for_remaining_subset():
    """When producer restarts with a subset (remaining), file_idx still uses original order."""
    files = [Path(f"/fake/{i}.cr3") for i in range(5)]
    state = _minimal_state(files)

    # Simulate looking up the third file in a remaining-subset scenario
    assert state._file_index.get(files[2]) == 2
    assert state._file_index.get(files[4]) == 4
