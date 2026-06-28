"""A minimal centroid tracker.

Assigns a stable integer ID to each object across frames by greedily matching
new detections to existing tracks by nearest centroid (within a distance
threshold). Tracks that go unmatched for ``max_disappeared`` frames are dropped.
This is the classic lightweight tracker used for counting/flow analytics when a
full Kalman/Re-ID tracker would be overkill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Settings, get_settings
from .detection import Detection


@dataclass
class Track:
    object_id: int
    centroid: tuple[float, float]
    disappeared: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)


class CentroidTracker:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._next_id = 0
        self.tracks: dict[int, Track] = {}

    def _register(self, centroid: tuple[float, float]) -> None:
        self.tracks[self._next_id] = Track(
            object_id=self._next_id, centroid=centroid, history=[centroid]
        )
        self._next_id += 1

    def _deregister(self, object_id: int) -> None:
        del self.tracks[object_id]

    def update(self, detections: list[Detection]) -> dict[int, Track]:
        """Advance the tracker by one frame and return the live tracks."""
        if not detections:
            for oid in list(self.tracks):
                self.tracks[oid].disappeared += 1
                if self.tracks[oid].disappeared > self.settings.max_disappeared:
                    self._deregister(oid)
            return self.tracks

        new_centroids = [d.centroid for d in detections]

        if not self.tracks:
            for c in new_centroids:
                self._register(c)
            return self.tracks

        ids = list(self.tracks)
        existing = np.array([self.tracks[i].centroid for i in ids])
        incoming = np.array(new_centroids)

        # Pairwise distances (tracks × detections).
        dist = np.linalg.norm(existing[:, None, :] - incoming[None, :, :], axis=2)

        matched_tracks, matched_dets = set(), set()
        # Greedy nearest-neighbour assignment.
        for r, c in sorted(
            ((r, c) for r in range(dist.shape[0]) for c in range(dist.shape[1])),
            key=lambda rc: dist[rc],
        ):
            if r in matched_tracks or c in matched_dets:
                continue
            if dist[r, c] > self.settings.max_track_distance:
                continue
            oid = ids[r]
            self.tracks[oid].centroid = new_centroids[c]
            self.tracks[oid].disappeared = 0
            self.tracks[oid].history.append(new_centroids[c])
            matched_tracks.add(r)
            matched_dets.add(c)

        # Unmatched existing tracks -> aged; dropped if stale.
        for r, oid in enumerate(ids):
            if r not in matched_tracks:
                self.tracks[oid].disappeared += 1
                if self.tracks[oid].disappeared > self.settings.max_disappeared:
                    self._deregister(oid)

        # Unmatched detections -> new objects.
        for c in range(len(new_centroids)):
            if c not in matched_dets:
                self._register(new_centroids[c])

        return self.tracks
