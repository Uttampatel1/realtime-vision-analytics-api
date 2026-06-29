"""Tests for polygon-zone analytics, point-in-polygon, and speed estimation."""
from __future__ import annotations

from src.analytics import (
    PolygonZone,
    SequenceAnalytics,
    SpeedEstimator,
    point_in_polygon,
)
from src.tracking import Track


def _track(oid, positions):
    """Build a Track whose history is a list of (x, y) points."""
    pts = [(float(x), float(y)) for x, y in positions]
    t = Track(object_id=oid, centroid=pts[-1])
    t.history = list(pts)
    return t


def test_point_in_polygon_square():
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon((50, 50), square)
    assert not point_in_polygon((150, 50), square)


def test_point_in_polygon_concave_triangle():
    tri = [(0, 0), (100, 0), (50, 100)]
    assert point_in_polygon((50, 10), tri)
    assert not point_in_polygon((5, 90), tri)  # outside the apex


def test_polygon_zone_occupancy_and_dwell():
    zone = PolygonZone(name="lane", points=[(0, 0), (100, 0), (100, 100), (0, 100)])
    for _ in range(3):
        zone.update({1: _track(1, [(50, 50), (50, 50)])})
    zone.update({1: _track(1, [(50, 50), (500, 500)])})  # leaves
    assert zone.dwell_frames[1] == 3
    assert zone.max_dwell == 3
    assert zone.unique_visitors == 1
    assert zone.peak_occupancy == 1


def test_speed_estimator_pixels_per_frame():
    est = SpeedEstimator()
    # moves 3,4 -> 5 px/frame, then 6,8 -> 10 px/frame
    est.update({1: _track(1, [(0, 0), (3, 4)])})
    est.update({1: _track(1, [(3, 4), (9, 12)])})
    assert round(est.per_track_max[1], 2) == 10.0
    assert round(est.avg_speed(1), 2) == 7.5
    assert round(est.max_speed, 2) == 10.0


def test_speed_estimator_metres_per_second_when_calibrated():
    est = SpeedEstimator(fps=30.0, pixels_per_meter=100.0)
    est.update({1: _track(1, [(0, 0), (10, 0)])})  # 10 px/frame
    s = est.summary()
    # 10 px/frame * 30 fps / 100 px/m = 3.0 m/s
    assert s["max_m_per_s"] == 3.0
    assert s["tracks_measured"] == 1


def test_sequence_analytics_includes_polygon_and_speed_blocks():
    seq = SequenceAnalytics(
        line_x=320.0,
        polygons=[{"name": "doorway", "points": [(0, 0), (100, 0), (100, 100), (0, 100)]}],
        track_speed=True,
    )
    seq.update(1, {0: _track(0, [(40, 40), (50, 50)])})
    summary = seq.summary()
    assert "polygon_zones" in summary
    assert summary["polygon_zones"][0]["name"] == "doorway"
    assert summary["polygon_zones"][0]["peak_occupancy"] == 1
    assert "speed" in summary
    assert summary["speed"]["tracks_measured"] == 1
