"""Run the analytics pipeline over a synthetic sequence and save annotated frames.

Usage::

    python run_demo.py
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from src.analytics import line_x_pixels
from src.config import get_settings
from src.pipeline import VisionPipeline
from src.synthetic import generate_sequence, to_bgr


def main() -> None:
    settings = get_settings()
    frames = generate_sequence(n_frames=60, n_objects=4, settings=settings)
    # A demo region-of-interest in the middle of the frame for occupancy/dwell.
    zone = (
        settings.frame_width * 0.35, settings.frame_height * 0.25,
        settings.frame_width * 0.65, settings.frame_height * 0.75,
    )
    # An arbitrary polygon ROI (a diagonal lane) + per-track speed estimation.
    lane = [
        (settings.frame_width * 0.10, settings.frame_height * 0.20),
        (settings.frame_width * 0.55, settings.frame_height * 0.15),
        (settings.frame_width * 0.90, settings.frame_height * 0.80),
        (settings.frame_width * 0.30, settings.frame_height * 0.85),
    ]
    pipeline = VisionPipeline(
        settings,
        zone=zone,
        polygons=[{"name": "diagonal_lane", "points": lane}],
        track_speed=True,
    )

    os.makedirs(settings.data_dir, exist_ok=True)
    line_x = int(line_x_pixels(settings))
    zx1, zy1, zx2, zy2 = (int(v) for v in zone)
    lane_pts = np.array([[int(x), int(y)] for x, y in lane], dtype=np.int32)
    saved = []
    for frame in frames:
        result = pipeline.process(frame)
        canvas = to_bgr(frame).copy()
        cv2.line(canvas, (line_x, 0), (line_x, settings.frame_height), (0, 0, 255), 2)
        cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), (255, 128, 0), 2)  # ROI zone
        cv2.polylines(canvas, [lane_pts], True, (200, 0, 200), 2)  # polygon lane
        for det in result.detections:
            cv2.rectangle(
                canvas, (det.x, det.y), (det.x + det.w, det.y + det.h),
                (0, 255, 0), 2,
            )
        cv2.putText(
            canvas, f"frame {result.frame_index}  objects {len(result.detections)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )
        # Save a few sample frames for the README.
        if result.frame_index in (0, 20, 40, 59):
            path = os.path.join(settings.data_dir, f"frame_{result.frame_index:03d}.png")
            cv2.imwrite(path, canvas)
            saved.append(path)

    summary = pipeline.summary()
    print(json.dumps(summary, indent=2))
    print("\nSaved sample annotated frames:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
