"""Configuration from environment / ``.env``.

The service ships with a deterministic **classical-CV** detector (OpenCV contour
blobs) so it runs offline with no model download. An optional ONNX object-detector
backend can be enabled by pointing ``ONNX_MODEL_PATH`` at a model file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Vision-pipeline settings resolved from the environment."""

    # "blob"  -> OpenCV contour detector (default, offline, deterministic)
    # "onnx"  -> ONNX Runtime object detector (requires ONNX_MODEL_PATH)
    detector: str = _get("DETECTOR", "blob")
    onnx_model_path: str = _get("ONNX_MODEL_PATH", "")

    frame_width: int = int(_get("FRAME_WIDTH", "640"))
    frame_height: int = int(_get("FRAME_HEIGHT", "480"))

    # Blob detector: ignore contours smaller than this area (pixels).
    min_blob_area: int = int(_get("MIN_BLOB_AREA", "150"))
    # Binarisation threshold for the grayscale frame (0-255).
    binary_threshold: int = int(_get("BINARY_THRESHOLD", "60"))

    # Centroid tracker: max pixel distance to match an object across frames,
    # and how many missed frames before an object is dropped.
    max_track_distance: float = float(_get("MAX_TRACK_DISTANCE", "60"))
    max_disappeared: int = int(_get("MAX_DISAPPEARED", "8"))

    # Virtual counting line as a fraction of frame width (vertical line).
    counting_line_x: float = float(_get("COUNTING_LINE_X", "0.5"))

    data_dir: str = _get("DATA_DIR", "data")
    seed: int = int(_get("SEED", "42"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
