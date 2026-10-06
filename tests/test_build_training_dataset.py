"""Tests for resumable training dataset builds (ML models mocked out)."""

import sys

import numpy as np
import pytest
from PIL import Image

from autocropper import build_training_dataset as btd
from autocropper.ml_crop import MASK_SIZE, TrainingDataset, TrainingRecord


def _fake_record(image, crop, models, name=None, source_path=""):
    # Images named "empty_*" simulate a photo with no detectable person.
    if name.startswith("empty_"):
        return None
    return TrainingRecord(
        mask=np.ones((MASK_SIZE, MASK_SIZE), dtype=bool),
        face_centroid=(0.5, 0.3), crop_center=(0.5, 0.5),
        min_margin=0.1, aspect_ratio=1.0, source_path=source_path,
    )


@pytest.fixture
def photos(tmp_path, monkeypatch):
    folder = tmp_path / "photos"
    folder.mkdir()
    for name in ["a.jpg", "b.jpg", "c.jpg", "empty_d.jpg", "e.jpg"]:
        Image.new("RGB", (10, 10)).save(folder / name)
    monkeypatch.setattr(btd, "load_ml_models", lambda: None)
    return folder


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["build-training-dataset", *map(str, argv)])
    btd.main()


def test_interrupt_then_resume(photos, tmp_path, monkeypatch):
    out = tmp_path / "out.pkl"
    calls = []

    def interrupting(image, crop, models, name=None, source_path=""):
        if len(calls) == 3:
            raise KeyboardInterrupt
        calls.append(name)
        return _fake_record(image, crop, models, name, source_path)

    monkeypatch.setattr(btd, "build_training_record", interrupting)
    with pytest.raises(SystemExit):
        _run(monkeypatch, photos, out, "--checkpoint-every", "100")

    partial = TrainingDataset.load(out, augment_mirrors=False)
    assert len(partial.processed_paths) == 3

    resumed = []

    def tracking(image, crop, models, name=None, source_path=""):
        resumed.append(name)
        return _fake_record(image, crop, models, name, source_path)

    monkeypatch.setattr(btd, "build_training_record", tracking)
    _run(monkeypatch, photos, out)

    # Only the two unprocessed files are run on resume.
    assert len(resumed) == 2
    assert set(calls) | set(resumed) == {"a.jpg", "b.jpg", "c.jpg", "empty_d.jpg", "e.jpg"}
    final = TrainingDataset.load(out, augment_mirrors=False)
    assert len(final.records) == 4
    assert len(final.processed_paths) == 5


def test_rerun_skips_everything_including_detection_failures(photos, tmp_path, monkeypatch):
    out = tmp_path / "out.pkl"
    monkeypatch.setattr(btd, "build_training_record", _fake_record)
    _run(monkeypatch, photos, out)

    def fail(*a, **k):
        raise AssertionError("should not re-process")

    monkeypatch.setattr(btd, "build_training_record", fail)
    _run(monkeypatch, photos, out)
    assert len(TrainingDataset.load(out, augment_mirrors=False).records) == 4


def test_fresh_ignores_existing_output(photos, tmp_path, monkeypatch):
    out = tmp_path / "out.pkl"
    monkeypatch.setattr(btd, "build_training_record", _fake_record)
    _run(monkeypatch, photos, out)
    _run(monkeypatch, photos, out, "--fresh")
    assert len(TrainingDataset.load(out, augment_mirrors=False).records) == 4


def test_resume_from_dataset_without_processed_paths(photos, tmp_path, monkeypatch):
    """Datasets built before resume support only carry source_path on records."""
    out = tmp_path / "out.pkl"
    ds = TrainingDataset(records=[_fake_record(None, None, None, "a.jpg", str((photos / "a.jpg").resolve()))])
    del ds.processed_paths
    ds.save(out)

    resumed = []

    def tracking(image, crop, models, name=None, source_path=""):
        resumed.append(name)
        return _fake_record(image, crop, models, name, source_path)

    monkeypatch.setattr(btd, "build_training_record", tracking)
    _run(monkeypatch, photos, out)
    assert "a.jpg" not in resumed
    assert len(resumed) == 4
