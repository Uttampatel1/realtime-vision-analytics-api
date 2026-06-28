"""Frame-level and sequence-level analytics built on tracked objects.

The headline analytic is a **line-crossing counter**: a vertical virtual line is
placed across the scene and we count objects whose centroid crosses it
left-to-right or right-to-left. This is the core of footfall / vehicle-flow
counting. We also expose per-frame object counts and the number of unique objects
seen across the whole sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings, get_settings
from .tracking import Track


@dataclass
class LineCrossingCounter:
    """Count tracked centroids crossing a vertical line at ``line_x`` (pixels)."""

    line_x: float
    left_to_right: int = 0
    right_to_left: int = 0
    _counted: set[int] = field(default_factory=set)

    def update(self, tracks: dict[int, Track]) -> None:
        for oid, track in tracks.items():
            if len(track.history) < 2 or oid in self._counted:
                continue
            prev_x = track.history[-2][0]
            curr_x = track.history[-1][0]
            if prev_x < self.line_x <= curr_x:
                self.left_to_right += 1
                self._counted.add(oid)
            elif prev_x > self.line_x >= curr_x:
                self.right_to_left += 1
                self._counted.add(oid)

    @property
    def total(self) -> int:
        return self.left_to_right + self.right_to_left


@dataclass
class ZoneCounter:
    """Region-of-interest analytics: occupancy + per-object dwell time.

    A rectangular zone ``(x1, y1, x2, y2)`` in pixels models a checkout queue, a
    shelf, a no-go area, etc. Each frame we count how many tracked objects sit
    inside the zone (occupancy), remember the peak, and accumulate per-track
    *dwell* (number of frames spent in the zone) — the basis for queue-length and
    loitering analytics.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    current_occupancy: int = 0
    peak_occupancy: int = 0
    dwell_frames: dict[int, int] = field(default_factory=dict)

    def _contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def update(self, tracks: dict[int, Track]) -> None:
        inside = 0
        for oid, track in tracks.items():
            if self._contains(track.centroid):
                inside += 1
                self.dwell_frames[oid] = self.dwell_frames.get(oid, 0) + 1
        self.current_occupancy = inside
        self.peak_occupancy = max(self.peak_occupancy, inside)

    @property
    def unique_visitors(self) -> int:
        return len(self.dwell_frames)

    @property
    def max_dwell(self) -> int:
        return max(self.dwell_frames.values(), default=0)


@dataclass
class SequenceAnalytics:
    line_x: float
    zone: tuple[float, float, float, float] | None = None
    per_frame_counts: list[int] = field(default_factory=list)
    counter: LineCrossingCounter = field(init=False)
    zone_counter: ZoneCounter | None = field(init=False, default=None)
    _seen_ids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.counter = LineCrossingCounter(self.line_x)
        if self.zone is not None:
            self.zone_counter = ZoneCounter(*self.zone)

    def update(self, n_detections: int, tracks: dict[int, Track]) -> None:
        self.per_frame_counts.append(n_detections)
        self._seen_ids.update(tracks.keys())
        self.counter.update(tracks)
        if self.zone_counter is not None:
            self.zone_counter.update(tracks)

    def summary(self) -> dict:
        counts = self.per_frame_counts
        out = {
            "frames": len(counts),
            "unique_objects": len(self._seen_ids),
            "max_objects_in_frame": max(counts) if counts else 0,
            "avg_objects_per_frame": (
                round(sum(counts) / len(counts), 2) if counts else 0.0
            ),
            "crossings_left_to_right": self.counter.left_to_right,
            "crossings_right_to_left": self.counter.right_to_left,
            "crossings_total": self.counter.total,
        }
        if self.zone_counter is not None:
            out["zone"] = {
                "peak_occupancy": self.zone_counter.peak_occupancy,
                "current_occupancy": self.zone_counter.current_occupancy,
                "unique_visitors": self.zone_counter.unique_visitors,
                "max_dwell_frames": self.zone_counter.max_dwell,
            }
        return out


def line_x_pixels(settings: Settings | None = None) -> float:
    settings = settings or get_settings()
    return settings.counting_line_x * settings.frame_width
