"""Run the analytics pipeline over a synthetic sequence and save annotated frames.

Usage::

    python run_demo.py
"""
from __future__ import annotations

import json
import os

import cv2

from src.analytics import line_x_pixels
from src.config import get_settings
from src.pipeline import VisionPipeline
from src.synthetic import generate_sequence, to_bgr


def main() -> None:
    settings = get_settings()
    frames = generate_sequence(n_frames=60, n_objects=4, settings=settings)
    pipeline = VisionPipeline(settings)

    os.makedirs(settings.data_dir, exist_ok=True)
    line_x = int(line_x_pixels(settings))
    saved = []
    for frame in frames:
        result = pipeline.process(frame)
        canvas = to_bgr(frame).copy()
        cv2.line(canvas, (line_x, 0), (line_x, settings.frame_height), (0, 0, 255), 2)
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
