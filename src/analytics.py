"""Frame-level and sequence-level analytics built on tracked objects.

The headline analytic is a **line-crossing counter**: a vertical virtual line is
placed across the scene and we count objects whose centroid crosses it
left-to-right or right-to-left. This is the core of footfall / vehicle-flow
counting. We also expose per-frame object counts and the number of unique objects
seen across the whole sequence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .tracking import Track

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray-casting point-in-polygon test for an arbitrary (convex or concave) ROI.

    Counts how many polygon edges a rightward ray from ``point`` crosses; an odd
    count means the point is inside. Works for any simple polygon, unlike the
    axis-aligned rectangle test, so zones can follow lanes, doorways or aisles.
    """
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


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
class PolygonZone:
    """Occupancy + dwell analytics for an arbitrary polygon region of interest.

    Like :class:`ZoneCounter` but the region is any simple polygon (a lane, a
    doorway funnel, an L-shaped aisle), and each zone carries a ``name`` so a
    scene can track several at once.
    """

    name: str
    points: list[Point]
    current_occupancy: int = 0
    peak_occupancy: int = 0
    dwell_frames: dict[int, int] = field(default_factory=dict)

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.points)

    def update(self, tracks: dict[int, Track]) -> None:
        inside = 0
        for oid, track in tracks.items():
            if self.contains(track.centroid):
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

    def summary(self) -> dict:
        return {
            "name": self.name,
            "peak_occupancy": self.peak_occupancy,
            "current_occupancy": self.current_occupancy,
            "unique_visitors": self.unique_visitors,
            "max_dwell_frames": self.max_dwell,
        }


@dataclass
class SpeedEstimator:
    """Per-track speed from centroid displacement between consecutive frames.

    Speed is measured in **pixels/frame** (calibration-free). If both ``fps`` and
    ``pixels_per_meter`` are provided, summaries also report **metres/second** —
    the basis for speed-limit and flow-rate analytics.
    """

    fps: float = 0.0
    pixels_per_meter: float = 0.0
    per_track_max: dict[int, float] = field(default_factory=dict)
    _sum: dict[int, float] = field(default_factory=dict)
    _count: dict[int, int] = field(default_factory=dict)

    def update(self, tracks: dict[int, Track]) -> None:
        for oid, track in tracks.items():
            hist = track.history
            if len(hist) < 2:
                continue
            (x0, y0), (x1, y1) = hist[-2], hist[-1]
            speed = math.hypot(x1 - x0, y1 - y0)  # pixels/frame
            self.per_track_max[oid] = max(self.per_track_max.get(oid, 0.0), speed)
            self._sum[oid] = self._sum.get(oid, 0.0) + speed
            self._count[oid] = self._count.get(oid, 0) + 1

    def avg_speed(self, oid: int) -> float:
        return self._sum[oid] / self._count[oid] if self._count.get(oid) else 0.0

    @property
    def max_speed(self) -> float:
        return max(self.per_track_max.values(), default=0.0)

    @property
    def mean_speed(self) -> float:
        per = [self._sum[o] / self._count[o] for o in self._count]
        return sum(per) / len(per) if per else 0.0

    def _to_mps(self, px_per_frame: float) -> float | None:
        if self.fps > 0 and self.pixels_per_meter > 0:
            return px_per_frame * self.fps / self.pixels_per_meter
        return None

    def summary(self) -> dict:
        out = {
            "max_px_per_frame": round(self.max_speed, 2),
            "avg_px_per_frame": round(self.mean_speed, 2),
            "tracks_measured": len(self._count),
        }
        mx = self._to_mps(self.max_speed)
        if mx is not None:
            out["max_m_per_s"] = round(mx, 2)
            out["avg_m_per_s"] = round(self._to_mps(self.mean_speed), 2)
        return out


@dataclass
class SequenceAnalytics:
    line_x: float
    zone: tuple[float, float, float, float] | None = None
    polygons: list[dict] | None = None
    track_speed: bool = True
    fps: float = 0.0
    pixels_per_meter: float = 0.0
    per_frame_counts: list[int] = field(default_factory=list)
    counter: LineCrossingCounter = field(init=False)
    zone_counter: ZoneCounter | None = field(init=False, default=None)
    polygon_zones: list[PolygonZone] = field(init=False, default_factory=list)
    speed_estimator: SpeedEstimator | None = field(init=False, default=None)
    _seen_ids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.counter = LineCrossingCounter(self.line_x)
        if self.zone is not None:
            self.zone_counter = ZoneCounter(*self.zone)
        for spec in self.polygons or []:
            self.polygon_zones.append(
                PolygonZone(name=spec["name"], points=[tuple(p) for p in spec["points"]])
            )
        if self.track_speed:
            self.speed_estimator = SpeedEstimator(
                fps=self.fps, pixels_per_meter=self.pixels_per_meter
            )

    def update(self, n_detections: int, tracks: dict[int, Track]) -> None:
        self.per_frame_counts.append(n_detections)
        self._seen_ids.update(tracks.keys())
        self.counter.update(tracks)
        if self.zone_counter is not None:
            self.zone_counter.update(tracks)
        for pz in self.polygon_zones:
            pz.update(tracks)
        if self.speed_estimator is not None:
            self.speed_estimator.update(tracks)

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
        if self.polygon_zones:
            out["polygon_zones"] = [pz.summary() for pz in self.polygon_zones]
        if self.speed_estimator is not None:
            out["speed"] = self.speed_estimator.summary()
        return out


def line_x_pixels(settings: Settings | None = None) -> float:
    settings = settings or get_settings()
    return settings.counting_line_x * settings.frame_width
