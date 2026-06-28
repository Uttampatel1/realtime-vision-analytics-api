"""The end-to-end per-frame pipeline: detect -> track -> analytics.

A :class:`VisionPipeline` is stateful across frames (it owns the tracker and the
running analytics), so feed it frames in order. Use :func:`analyze_single_frame`
for a stateless one-shot detection (what the ``/analyze`` API endpoint uses).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analytics import SequenceAnalytics, line_x_pixels
from .config import Settings, get_settings
from .detection import Detection, build_detector
from .logging_utils import get_logger
from .tracking import CentroidTracker

log = get_logger(__name__)


@dataclass
class FrameResult:
    frame_index: int
    detections: list[Detection]
    track_ids: list[int]

    def as_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "count": len(self.detections),
            "detections": [d.as_dict() for d in self.detections],
            "track_ids": self.track_ids,
        }


class VisionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        zone: tuple[float, float, float, float] | None = None,
    ):
        self.settings = settings or get_settings()
        self.detector = build_detector(self.settings)
        self.tracker = CentroidTracker(self.settings)
        self.analytics = SequenceAnalytics(line_x_pixels(self.settings), zone=zone)
        self._frame_index = 0
        log.info(
            "VisionPipeline ready (detector=%s, zone=%s)",
            self.settings.detector, zone,
        )

    def process(self, frame: np.ndarray) -> FrameResult:
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)
        self.analytics.update(len(detections), tracks)
        result = FrameResult(
            frame_index=self._frame_index,
            detections=detections,
            track_ids=sorted(tracks.keys()),
        )
        self._frame_index += 1
        return result

    def process_sequence(self, frames: list[np.ndarray]) -> list[FrameResult]:
        return [self.process(f) for f in frames]

    def summary(self) -> dict:
        return self.analytics.summary()


def analyze_single_frame(
    frame: np.ndarray, settings: Settings | None = None
) -> dict:
    """Stateless single-image detection (no tracking)."""
    settings = settings or get_settings()
    detector = build_detector(settings)
    detections = detector.detect(frame)
    return {
        "count": len(detections),
        "detections": [d.as_dict() for d in detections],
    }
