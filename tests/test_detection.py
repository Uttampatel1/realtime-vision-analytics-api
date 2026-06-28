from __future__ import annotations

import numpy as np

from src.config import Settings
from src.detection import BlobDetector, Detection, build_detector
from src.synthetic import generate_sequence


def _settings() -> Settings:
    return Settings(frame_width=320, frame_height=240, seed=7, min_blob_area=80)


def test_detection_centroid_and_area():
    d = Detection(x=10, y=20, w=30, h=40, score=0.9)
    assert d.centroid == (25.0, 40.0)
    assert d.area == 1200


def test_blob_detector_finds_known_object_count():
    settings = _settings()
    # Objects can overlap near the left edge at t=0; once they spread out the
    # detector should resolve all 3. Check the max count across the sequence.
    frames = generate_sequence(n_frames=30, n_objects=3, settings=settings)
    detector = BlobDetector(settings)
    counts = [len(detector.detect(f)) for f in frames]
    assert max(counts) == 3
    for d in detector.detect(frames[-1]):
        assert d.w > 0 and d.h > 0
        assert 0.0 <= d.score <= 1.0


def test_blob_detector_empty_on_blank_frame():
    settings = _settings()
    blank = np.zeros((240, 320), dtype=np.uint8)
    assert BlobDetector(settings).detect(blank) == []


def test_build_detector_returns_blob_by_default():
    det = build_detector(_settings())
    assert det.name == "blob"
