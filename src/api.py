"""FastAPI service exposing the vision-analytics pipeline.

Endpoints
---------
* ``GET  /health``            — liveness + active detector.
* ``GET  /config``            — effective settings.
* ``POST /analyze``           — upload an image, get object detections + count.
* ``POST /analyze/sequence``  — run the full detect→track→count pipeline over a
  freshly generated synthetic sequence and return the analytics summary.

Run with::

    uvicorn src.api:app --reload
"""
from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from .config import get_settings
from .pipeline import VisionPipeline, analyze_single_frame
from .synthetic import generate_sequence

app = FastAPI(title="Real-Time Vision Analytics API", version="1.0.0")


class SequenceRequest(BaseModel):
    n_frames: int = 60
    n_objects: int = 3


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "detector": get_settings().detector}


@app.get("/config")
def config() -> dict:
    s = get_settings()
    return {
        "detector": s.detector,
        "frame_width": s.frame_width,
        "frame_height": s.frame_height,
        "min_blob_area": s.min_blob_area,
        "counting_line_x": s.counting_line_x,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    """Detect objects in a single uploaded image."""
    raw = await file.read()
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "could not decode image"}
    result = analyze_single_frame(frame)
    result["filename"] = file.filename
    return result


@app.post("/analyze/sequence")
def analyze_sequence(req: SequenceRequest) -> dict:
    """Run the full pipeline over a synthetic sequence and return analytics."""
    frames = generate_sequence(req.n_frames, req.n_objects)
    pipeline = VisionPipeline()
    results = pipeline.process_sequence(frames)
    return {
        "summary": pipeline.summary(),
        "per_frame": [r.as_dict() for r in results],
    }
