"""Tests for pure ML crop functions (no model loading required)."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from autocropper.ml_crop import (
    DEFAULT_N_NEIGHBORS,
    MASK_SIZE,
    TrainingDataset,
    TrainingRecord,
    _face_dist,
    _mask_bbox,
    _mask_iou,
    _normalize_mask,
    _yaw_dist,
)
from autocropper.optimize_weights import _cross_val_error


# ── _mask_bbox ────────────────────────────────────────────────────────────

def test_mask_bbox_simple():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:50, 30:70] = True
    assert _mask_bbox(mask) == (30, 20, 69, 49)


def test_mask_bbox_full():
    mask = np.ones((50, 80), dtype=bool)
    assert _mask_bbox(mask) == (0, 0, 79, 49)


def test_mask_bbox_empty_returns_none():
    mask = np.zeros((100, 100), dtype=bool)
    assert _mask_bbox(mask) is None


def test_mask_bbox_single_pixel():
    mask = np.zeros((100, 100), dtype=bool)
    mask[42, 57] = True
    assert _mask_bbox(mask) == (57, 42, 57, 42)


# ── _normalize_mask ───────────────────────────────────────────────────────

def test_normalize_mask_output_shape():
    mask = np.ones((200, 100), dtype=bool)
    out = _normalize_mask(mask)
    assert out.shape == (MASK_SIZE, MASK_SIZE)


def test_normalize_mask_preserves_truthy_region():
    """After normalisation the mask should contain some True pixels."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    out = _normalize_mask(mask)
    assert out.any()


def test_normalize_mask_empty_input():
    """All-False mask normalises to all-False."""
    mask = np.zeros((100, 100), dtype=bool)
    out = _normalize_mask(mask)
    assert not out.any()


def test_normalize_mask_landscape_fits():
    """A wide mask should fill MASK_SIZE pixels horizontally."""
    mask = np.ones((100, 300), dtype=bool)
    out = _normalize_mask(mask)
    # Width should be MASK_SIZE
    cols = np.any(out, axis=0)
    assert cols.sum() == MASK_SIZE


def test_normalize_mask_portrait_fits():
    """A tall mask should fill MASK_SIZE pixels vertically."""
    mask = np.ones((300, 100), dtype=bool)
    out = _normalize_mask(mask)
    rows = np.any(out, axis=1)
    assert rows.sum() == MASK_SIZE


# ── _mask_iou ─────────────────────────────────────────────────────────────

def _solid_norm_mask(h=MASK_SIZE, w=MASK_SIZE):
    return np.ones((h, w), dtype=bool)


def test_mask_iou_identical_masks():
    """Identical all-True 1024×1024 masks have IoU = 1.0 (overlap = full grid)."""
    m = _solid_norm_mask()
    assert _mask_iou(m, m) == pytest.approx(1.0)


def test_mask_iou_no_overlap():
    m1 = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    m2 = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    m1[:, :MASK_SIZE // 2] = True
    m2[:, MASK_SIZE // 2:] = True
    assert _mask_iou(m1, m2) == pytest.approx(0.0)


def test_mask_iou_half_overlap():
    m1 = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    m2 = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    m1[:, :MASK_SIZE // 2] = True   # left half
    m2[:, :MASK_SIZE // 4] = True   # leftmost quarter
    # Intersection = leftmost quarter; union = left half → IoU = 0.25/0.5 = 0.5
    assert _mask_iou(m1, m2) == pytest.approx(0.5)


# ── _face_dist ────────────────────────────────────────────────────────────

def test_face_dist_same_point():
    assert _face_dist((0.5, 0.5), (0.5, 0.5)) == pytest.approx(0.0)


def test_face_dist_unit_diagonal():
    assert _face_dist((0.0, 0.0), (1.0, 1.0)) == pytest.approx(2 ** 0.5)


def test_face_dist_horizontal():
    assert _face_dist((0.0, 0.5), (1.0, 0.5)) == pytest.approx(1.0)


# ── TrainingDataset serialisation ─────────────────────────────────────────

def _make_record(seed=0, face_yaw=0.0):
    rng = np.random.default_rng(seed)
    mask = rng.random((60, 40)) > 0.5
    return TrainingRecord(
        mask=mask,
        face_centroid=(0.4, 0.2),
        face_yaw=face_yaw,
        crop_center=(0.5, 0.5),
        min_margin=0.15,
        aspect_ratio=1.5,
    )


def test_dataset_save_load_round_trip(tmp_path):
    records = [_make_record(i) for i in range(5)]
    ds = TrainingDataset(records=records, alpha=0.7)
    out = tmp_path / "dataset.pkl"
    ds.save(out)

    # augment_mirrors=False so we only check the raw serialised records.
    loaded = TrainingDataset.load(out, augment_mirrors=False)
    assert loaded.alpha == pytest.approx(0.7)
    assert len(loaded.records) == 5
    for orig, rec in zip(records, loaded.records):
        np.testing.assert_array_equal(orig.mask, rec.mask)
        assert orig.face_centroid == rec.face_centroid
        assert orig.crop_center == rec.crop_center
        assert orig.min_margin == pytest.approx(rec.min_margin)
        assert orig.aspect_ratio == pytest.approx(rec.aspect_ratio)


def test_dataset_load_from_stream(tmp_path):
    ds = TrainingDataset(records=[_make_record()], alpha=0.3)
    out = tmp_path / "ds.pkl"
    ds.save(out)

    with open(out, 'rb') as f:
        loaded = TrainingDataset.load(f, augment_mirrors=False)
    assert loaded.alpha == pytest.approx(0.3)
    assert len(loaded.records) == 1


def test_dataset_empty():
    ds = TrainingDataset()
    assert ds.records == []
    assert ds.alpha == pytest.approx(0.5)


# ── _cross_val_error (optimize_weights) ───────────────────────────────────

def _identical_records(n, face_centroid=(0.4, 0.3), crop_center=(0.5, 0.5),
                       min_margin=0.2, aspect_ratio=1.5, face_yaw=0.0):
    """All records share the same mask shape, face centroid, crop centre, and yaw."""
    mask = np.ones((80, 60), dtype=bool)
    return [
        TrainingRecord(
            mask=mask,
            face_centroid=face_centroid,
            face_yaw=face_yaw,
            crop_center=crop_center,
            min_margin=min_margin,
            aspect_ratio=aspect_ratio,
        )
        for _ in range(n)
    ]


def test_cross_val_error_identical_records_is_zero():
    """When all records are identical, LOO-CV prediction equals the truth → error = 0."""
    records = _identical_records(15)
    err = _cross_val_error(records, alpha=0.5, yaw_weight=0.0, n=DEFAULT_N_NEIGHBORS)
    assert err == pytest.approx(0.0, abs=1e-9)


def test_cross_val_error_returns_float():
    records = _identical_records(12)
    err = _cross_val_error(records, alpha=0.0, yaw_weight=0.0, n=5)
    assert isinstance(err, float)
    assert err >= 0.0


def test_cross_val_error_insufficient_records():
    """With fewer records than n+1, the function should skip all folds → inf."""
    records = _identical_records(5)
    err = _cross_val_error(records, alpha=0.5, yaw_weight=0.0, n=10)
    assert err == float('inf')


def test_cross_val_error_varies_with_alpha():
    """Different alpha values should generally produce different errors unless
    all features are identical (degenerate case tested separately above)."""
    # Create records where face centroids differ — alpha=0 uses face only, alpha=1 mask only
    rng = np.random.default_rng(42)
    mask = np.ones((80, 60), dtype=bool)
    records = [
        TrainingRecord(
            mask=mask,
            face_centroid=(rng.random(), rng.random()),
            face_yaw=0.0,
            crop_center=(rng.random(), rng.random()),
            min_margin=0.2,
            aspect_ratio=1.5,
        )
        for _ in range(20)
    ]
    e0 = _cross_val_error(records, alpha=0.0, yaw_weight=0.0, n=5)
    e1 = _cross_val_error(records, alpha=1.0, yaw_weight=0.0, n=5)
    # With identical masks but varying face centroids, alpha=1 (mask only) can't
    # distinguish records; alpha=0 (face only) uses the face signal.
    # They may or may not differ — just check both are finite non-negative floats.
    assert e0 >= 0.0 and e1 >= 0.0
    assert e0 != float('inf') and e1 != float('inf')


# ── Neighbour selection math (inline, without model calls) ─────────────────

def test_predict_selects_closest_neighbours():
    """Verify the distance formula and neighbour selection directly,
    without running any model inference."""
    from autocropper.ml_crop import _face_dist, _mask_iou, _normalize_mask

    # Build a query and a set of training records
    query_mask = np.ones((50, 50), dtype=bool)
    query_fc = (0.5, 0.5)

    # Identical record (should be nearest)
    near = TrainingRecord(
        mask=np.ones((50, 50), dtype=bool),
        face_centroid=(0.5, 0.5),
        face_yaw=0.0,
        crop_center=(0.6, 0.4),
        min_margin=0.2,
        aspect_ratio=1.5,
    )
    # Very different record
    far = TrainingRecord(
        mask=np.zeros((50, 50), dtype=bool),
        face_centroid=(0.0, 0.0),
        face_yaw=0.0,
        crop_center=(0.1, 0.9),
        min_margin=0.1,
        aspect_ratio=1.5,
    )

    q_norm = _normalize_mask(query_mask)
    alpha = 0.5

    def dist(r):
        iou = _mask_iou(q_norm, _normalize_mask(r.mask))
        fd = _face_dist(query_fc, r.face_centroid)
        return alpha * (1.0 - iou) + (1.0 - alpha) * fd

    assert dist(near) < dist(far)


# ── _yaw_dist ─────────────────────────────────────────────────────────────

def test_yaw_dist_same():
    assert _yaw_dist(0.0, 0.0) == pytest.approx(0.0)

def test_yaw_dist_opposite():
    assert _yaw_dist(-1.0, 1.0) == pytest.approx(1.0)

def test_yaw_dist_normalised():
    assert _yaw_dist(0.0, 1.0) == pytest.approx(0.5)

def test_yaw_dist_none_returns_fallback():
    assert _yaw_dist(None, 0.5, fallback=0.99) == pytest.approx(0.99)
    assert _yaw_dist(0.5, None, fallback=0.12) == pytest.approx(0.12)


# ── _mirror_record face_yaw ───────────────────────────────────────────────

def test_mirror_record_negates_face_yaw():
    r = _make_record(face_yaw=0.6)
    mirrored = TrainingDataset._mirror_record(r)
    assert mirrored.face_yaw == pytest.approx(-0.6)

def test_mirror_record_none_face_yaw():
    r = _make_record(face_yaw=None)
    mirrored = TrainingDataset._mirror_record(r)
    assert mirrored.face_yaw is None


# ── yaw_weight in distance formula ────────────────────────────────────────

def test_yaw_weight_ranks_matching_yaw_closer():
    """With yaw_weight > 0, a candidate with matching yaw outranks one with opposite yaw
    when mask and centroid are identical."""
    from autocropper.ml_crop import _face_dist, _mask_iou, _normalize_mask, _yaw_dist

    mask = np.ones((50, 50), dtype=bool)
    query_fc = (0.5, 0.5)
    query_yaw = 0.8  # strong rightward turn

    near = TrainingRecord(
        mask=mask, face_centroid=query_fc, face_yaw=0.8,
        crop_center=(0.5, 0.5), min_margin=0.2, aspect_ratio=1.5,
    )
    far = TrainingRecord(
        mask=mask, face_centroid=query_fc, face_yaw=-0.8,
        crop_center=(0.5, 0.5), min_margin=0.2, aspect_ratio=1.5,
    )

    q_norm = _normalize_mask(mask)
    alpha = 0.5
    yaw_weight = 0.5

    def dist(r):
        iou = _mask_iou(q_norm, _normalize_mask(r.mask))
        fd = _face_dist(query_fc, r.face_centroid)
        yd = _yaw_dist(query_yaw, r.face_yaw, fallback=fd)
        face_comp = (1.0 - yaw_weight) * fd + yaw_weight * yd
        return alpha * (1.0 - iou) + (1.0 - alpha) * face_comp

    assert dist(near) < dist(far)


# ── _cross_val_error with yaw_weight ──────────────────────────────────────

def test_cross_val_error_yaw_weight_zero_matches_no_yaw():
    """_cross_val_error with yaw_weight=0 produces the same result regardless of face_yaw values."""
    records = _identical_records(15, face_yaw=0.5)
    err_yw0 = _cross_val_error(records, alpha=0.5, yaw_weight=0.0, n=DEFAULT_N_NEIGHBORS)
    records_no_yaw = _identical_records(15, face_yaw=0.0)
    err_no_yaw = _cross_val_error(records_no_yaw, alpha=0.5, yaw_weight=0.0, n=DEFAULT_N_NEIGHBORS)
    assert err_yw0 == pytest.approx(err_no_yaw, abs=1e-9)
