"""Deterministic synthetic video generator.

Produces a sequence of frames in which a handful of bright "objects" (filled
circles / rectangles) move across a dark scene at constant velocities. This is a
stand-in for a real camera feed — e.g. people walking through a corridor or
vehicles passing a checkpoint — and lets the whole pipeline run and be tested
offline with no datasets and no privacy concerns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings, get_settings


@dataclass
class MovingObject:
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    shape: str  # "circle" | "rect"

    def step(self) -> None:
        self.x += self.vx
        self.y += self.vy


def _draw(frame: np.ndarray, obj: MovingObject) -> None:
    h, w = frame.shape[:2]
    cx, cy, r = int(obj.x), int(obj.y), obj.radius
    if obj.shape == "circle":
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        frame[mask] = 255
    else:  # rectangle
        x0, x1 = max(0, cx - r), min(w, cx + r)
        y0, y1 = max(0, cy - r), min(h, cy + r)
        frame[y0:y1, x0:x1] = 255


def generate_sequence(
    n_frames: int = 60,
    n_objects: int = 3,
    settings: Settings | None = None,
) -> list[np.ndarray]:
    """Return a list of single-channel (grayscale) frames as uint8 arrays."""
    settings = settings or get_settings()
    rng = np.random.default_rng(settings.seed)
    w, h = settings.frame_width, settings.frame_height

    objects: list[MovingObject] = []
    for _ in range(n_objects):
        # Start near the left edge, move rightward across the counting line.
        y = float(rng.integers(int(0.2 * h), int(0.8 * h)))
        objects.append(
            MovingObject(
                x=float(rng.integers(0, int(0.15 * w))),
                y=y,
                vx=float(rng.uniform(0.6, 1.2)) * (w / n_frames),
                vy=float(rng.uniform(-0.3, 0.3)) * (h / n_frames),
                radius=int(rng.integers(12, 22)),
                shape=rng.choice(["circle", "rect"]),
            )
        )

    frames = []
    for _ in range(n_frames):
        frame = np.zeros((h, w), dtype=np.uint8)
        # A little sensor noise so detection isn't trivially clean.
        frame += rng.integers(0, 12, (h, w), dtype=np.uint8)
        for obj in objects:
            _draw(frame, obj)
            obj.step()
        frames.append(frame)
    return frames


def to_bgr(frame: np.ndarray) -> np.ndarray:
    """Stack a grayscale frame into 3 channels (for annotation / encoding)."""
    return np.repeat(frame[:, :, None], 3, axis=2)
