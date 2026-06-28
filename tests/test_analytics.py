from __future__ import annotations

from src.analytics import LineCrossingCounter, SequenceAnalytics
from src.config import Settings
from src.pipeline import VisionPipeline
from src.synthetic import generate_sequence
from src.tracking import Track


def _track(oid: int, xs: list[float]) -> Track:
    history = [(x, 50.0) for x in xs]
    return Track(object_id=oid, centroid=history[-1], history=history)


def test_counts_left_to_right_crossing():
    counter = LineCrossingCounter(line_x=100.0)
    counter.update({0: _track(0, [90.0, 110.0])})
    assert counter.left_to_right == 1
    assert counter.right_to_left == 0


def test_counts_right_to_left_crossing():
    counter = LineCrossingCounter(line_x=100.0)
    counter.update({0: _track(0, [120.0, 80.0])})
    assert counter.right_to_left == 1


def test_object_counted_only_once():
    counter = LineCrossingCounter(line_x=100.0)
    t = _track(0, [90.0, 110.0])
    counter.update({0: t})
    t.history.append((130.0, 50.0))
    counter.update({0: t})
    assert counter.total == 1


def test_sequence_analytics_summary_keys():
    sa = SequenceAnalytics(line_x=100.0)
    sa.update(2, {0: _track(0, [90.0, 110.0]), 1: _track(1, [10.0, 20.0])})
    summary = sa.summary()
    for key in ("frames", "unique_objects", "crossings_total", "avg_objects_per_frame"):
        assert key in summary
    assert summary["frames"] == 1
    assert summary["crossings_total"] == 1


def test_full_pipeline_counts_objects_crossing_line():
    settings = Settings(frame_width=320, frame_height=240, seed=1, min_blob_area=80)
    frames = generate_sequence(n_frames=40, n_objects=3, settings=settings)
    pipeline = VisionPipeline(settings)
    pipeline.process_sequence(frames)
    summary = pipeline.summary()
    assert summary["frames"] == 40
    assert summary["unique_objects"] >= 3
    # objects start left and move right across the centre line
    assert summary["crossings_left_to_right"] >= 1
