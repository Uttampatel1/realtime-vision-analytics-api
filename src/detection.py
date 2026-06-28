"""Object detectors behind a single interface.

``Detector.detect(frame) -> list[Detection]`` where a frame is an ``H×W`` (gray)
or ``H×W×3`` (BGR) uint8 array. The default :class:`BlobDetector` uses classical
OpenCV (threshold + contours) so it needs no model and is fully deterministic.

An optional :class:`OnnxDetector` shows how a learned object detector would slot
into the same interface; it's only constructed when ``DETECTOR=onnx``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings, get_settings


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str = "object"

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "score": round(self.score, 3), "label": self.label,
        }


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


class BlobDetector:
    """Threshold the frame and return bounding boxes of bright connected blobs."""

    name = "blob"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = _to_gray(frame)
        _, binary = cv2.threshold(
            gray, self.settings.binary_threshold, 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        dets: list[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.settings.min_blob_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Confidence proxy: fill ratio of the bounding box.
            score = float(min(1.0, area / max(w * h, 1)))
            dets.append(Detection(int(x), int(y), int(w), int(h), score))
        return dets


class OnnxDetector:  # pragma: no cover - optional, requires a model file
    """Object detection via ONNX Runtime (optional)."""

    name = "onnx"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.onnx_model_path:
            raise ValueError("DETECTOR=onnx requires ONNX_MODEL_PATH")
        import onnxruntime as ort

        self.session = ort.InferenceSession(self.settings.onnx_model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError(
            "Wire your model's pre/post-processing here. The BlobDetector is the "
            "reference implementation of the Detector interface."
        )


def build_detector(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.detector == "onnx":
        return OnnxDetector(settings)
    return BlobDetector(settings)
