from __future__ import annotations

from src.analytics import SequenceAnalytics, ZoneCounter
from src.tracking import Track


def _track(oid: int, x: float, y: float) -> Track:
    return Track(object_id=oid, centroid=(x, y), history=[(x, y)])


def test_zone_counts_only_objects_inside():
    zone = ZoneCounter(10, 10, 100, 100)
    zone.update({0: _track(0, 50, 50), 1: _track(1, 200, 200)})  # 1 is outside
    assert zone.current_occupancy == 1
    assert zone.peak_occupancy == 1


def test_zone_tracks_peak_occupancy():
    zone = ZoneCounter(0, 0, 100, 100)
    zone.update({0: _track(0, 10, 10), 1: _track(1, 20, 20)})    # 2 inside
    zone.update({0: _track(0, 500, 500)})                        # all leave
    assert zone.current_occupancy == 0
    assert zone.peak_occupancy == 2


def test_zone_accumulates_dwell_per_object():
    zone = ZoneCounter(0, 0, 100, 100)
    for _ in range(3):
        zone.update({7: _track(7, 50, 50)})
    zone.update({7: _track(7, 999, 999)})   # leaves on frame 4
    assert zone.dwell_frames[7] == 3
    assert zone.max_dwell == 3
    assert zone.unique_visitors == 1


def test_sequence_analytics_includes_zone_block_when_configured():
    seq = SequenceAnalytics(line_x=320.0, zone=(0, 0, 100, 100))
    seq.update(1, {0: _track(0, 50, 50)})
    summary = seq.summary()
    assert "zone" in summary
    assert summary["zone"]["peak_occupancy"] == 1


def test_sequence_analytics_omits_zone_when_not_configured():
    seq = SequenceAnalytics(line_x=320.0)
    seq.update(1, {0: _track(0, 50, 50)})
    assert "zone" not in seq.summary()
