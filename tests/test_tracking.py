from __future__ import annotations

from src.config import Settings
from src.detection import Detection
from src.tracking import CentroidTracker


def _settings() -> Settings:
    return Settings(max_track_distance=50, max_disappeared=3)


def _det(cx: float, cy: float, size: int = 10) -> Detection:
    return Detection(x=int(cx - size / 2), y=int(cy - size / 2), w=size, h=size, score=1.0)


def test_registers_new_objects():
    tracker = CentroidTracker(_settings())
    tracks = tracker.update([_det(10, 10), _det(100, 100)])
    assert len(tracks) == 2


def test_keeps_stable_id_across_small_motion():
    tracker = CentroidTracker(_settings())
    tracker.update([_det(10, 10)])
    first_id = next(iter(tracker.tracks))
    tracker.update([_det(20, 12)])  # small move -> same id
    assert list(tracker.tracks) == [first_id]
    assert tracker.tracks[first_id].centroid == (20.0, 12.0)


def test_large_jump_creates_new_object():
    tracker = CentroidTracker(_settings())
    tracker.update([_det(10, 10)])
    tracker.update([_det(300, 300)])  # beyond max_track_distance
    # old track aged (still alive), new one registered
    assert len(tracker.tracks) == 2


def test_stale_track_is_dropped():
    tracker = CentroidTracker(_settings())
    tracker.update([_det(10, 10)])
    for _ in range(_settings().max_disappeared + 1):
        tracker.update([])  # no detections
    assert tracker.tracks == {}
