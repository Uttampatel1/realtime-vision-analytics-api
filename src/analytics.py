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
class SequenceAnalytics:
    line_x: float
    per_frame_counts: list[int] = field(default_factory=list)
    counter: LineCrossingCounter = field(init=False)
    _seen_ids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.counter = LineCrossingCounter(self.line_x)

    def update(self, n_detections: int, tracks: dict[int, Track]) -> None:
        self.per_frame_counts.append(n_detections)
        self._seen_ids.update(tracks.keys())
        self.counter.update(tracks)

    def summary(self) -> dict:
        counts = self.per_frame_counts
        return {
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


def line_x_pixels(settings: Settings | None = None) -> float:
    settings = settings or get_settings()
    return settings.counting_line_x * settings.frame_width
