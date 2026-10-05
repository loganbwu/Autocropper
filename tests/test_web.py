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
    state.all_files = state.files
    state._file_index = {f: i for i, f in enumerate(state.files)}
    state._all_index = state._file_index
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
    state.skip_noop = False
    state.skip_no_person = False
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


# ── Manual fallback when the next auto-crop isn't ready ──────────────────────

class _GatedCrop:
    """Fake compute_crop: each photo blocks until its gate is released, so a
    test controls exactly which auto-crops are 'ready'."""

    def __init__(self):
        self.gates = {}
        self.started = []

    def gate(self, cr3):
        return self.gates.setdefault(cr3, threading.Event())

    def __call__(self, models, cr3, all_people, margin_ratio=0.2):
        self.started.append(cr3)
        assert self.gate(cr3).wait(5)
        return {
            "cr3_path": cr3,
            "x1": 10, "y1": 10, "x2": 90, "y2": 90, "w": 100, "h": 100,
            "raw_x1": 30, "raw_y1": 30, "raw_x2": 70, "raw_y2": 70, "person_cx": 50,
            "orig_bytes": b"jpeg", "noop": False,
        }


def _wait_for(pred, timeout=5):
    deadline = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if pred():
            return True
        deadline.wait(0.01)
    return False


@contextmanager
def _review_session(n=3):
    files = [Path(f"/fake/{i}.cr3") for i in range(n)]
    fake = _GatedCrop()
    xmp = MagicMock()
    with patch("autocropper.web.compute_crop", fake), \
         patch("autocropper.web.recompute_crop"), \
         patch("autocropper.web.extract_preview_image", return_value=_make_rgb_image(120, 80)), \
         patch("autocropper.web.get_orientation", return_value=1), \
         patch("autocropper.web.apply_orientation", side_effect=lambda img, ori: img), \
         patch("autocropper.web.write_xmp", xmp), \
         patch("autocropper.web.write_decline_marker"):
        state = ReviewState(files, models=None)
        yield state, files, fake, xmp
        for g in fake.gates.values():
            g.set()  # let the producer thread finish


def _current_path(state):
    with state._lock:
        return state.current["cr3_path"] if state.current else None


def test_first_photo_waits_for_auto_crop():
    """Before the user has moved on, the consumer waits rather than going manual."""
    with _review_session() as (state, files, fake, _):
        assert _wait_for(lambda: files[0] in fake.started)
        threading.Event().wait(0.3)
        assert state.get_state()["status"] == "loading"
        fake.gate(files[0]).set()
        assert _wait_for(lambda: _current_path(state) == files[0])
        assert not state.current.get("manual")


def test_unready_next_photo_shown_manually_and_auto_result_discarded():
    with _review_session() as (state, files, fake, _):
        fake.gate(files[0]).set()
        assert _wait_for(lambda: _current_path(state) == files[0])
        assert _wait_for(lambda: files[1] in fake.started)

        state.decide("skip")  # photo 1's auto-crop is still running
        assert _wait_for(lambda: _current_path(state) == files[1])
        s = state.get_state()
        assert s["manual"] is True
        assert s["crop_coords"] == {"x1": 0, "y1": 0, "x2": 120, "y2": 80}
        assert s["file_idx"] == 1

        # The in-flight auto-crop for photo 1 finishes: discarded, not queued.
        fake.gate(files[1]).set()
        assert _wait_for(lambda: files[2] in fake.started)
        assert all(item["cr3_path"] != files[1] for item in list(state._prefetch_q.queue) if item)

        # Photo 2's auto-crop arrives normally once the manual one is decided.
        fake.gate(files[2]).set()
        assert _wait_for(lambda: state._prefetch_q.qsize() >= 1)
        state.decide("skip")
        assert _wait_for(lambda: _current_path(state) == files[2])
        assert not state.current.get("manual")


def test_claimed_photo_not_yet_started_is_never_auto_cropped():
    with _review_session(n=4) as (state, files, fake, _):
        fake.gate(files[0]).set()
        assert _wait_for(lambda: _current_path(state) == files[0])
        state.decide("skip")                     # photo 1 claimed (in flight)
        assert _wait_for(lambda: _current_path(state) == files[1])
        state.decide("skip")                     # photo 2 claimed (not started)
        assert _wait_for(lambda: _current_path(state) == files[2])
        assert state.current.get("manual")

        fake.gate(files[1]).set()
        fake.gate(files[3]).set()
        assert _wait_for(lambda: files[3] in fake.started)
        assert files[2] not in fake.started


def test_manual_crop_writes_manual_keyword():
    with _review_session() as (state, files, fake, xmp):
        fake.gate(files[0]).set()
        assert _wait_for(lambda: _current_path(state) == files[0])
        state.decide("skip")
        assert _wait_for(lambda: _current_path(state) == files[1])
        state.decide("crop", {"x1": 5, "y1": 5, "x2": 65, "y2": 45}, 0)
        assert _wait_for(lambda: xmp.called)
        args, kwargs = xmp.call_args
        assert args[:5] == (files[1], 5, 5, 65, 45)
        assert kwargs["keywords"] == ["AutoCropper", "AutoCropper_Manual"]


def test_navigate_to_unbuffered_photo_shows_it_manually():
    with _review_session(n=5) as (state, files, fake, _):
        fake.gate(files[0]).set()
        assert _wait_for(lambda: _current_path(state) == files[0])
        assert state.navigate(3)
        assert _wait_for(lambda: _current_path(state) == files[3])
        assert state.current.get("manual")
        fake.gate(files[1]).set()  # old generation's in-flight photo
        fake.gate(files[4]).set()
        assert _wait_for(lambda: files[4] in fake.started)
        assert files[3] not in fake.started
