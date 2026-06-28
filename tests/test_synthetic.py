from __future__ import annotations

import numpy as np

from src.config import Settings
from src.synthetic import generate_sequence, to_bgr


def _settings() -> Settings:
    return Settings(frame_width=320, frame_height=240, seed=3)


def test_sequence_shape_and_dtype():
    frames = generate_sequence(n_frames=10, n_objects=2, settings=_settings())
    assert len(frames) == 10
    for f in frames:
        assert f.shape == (240, 320)
        assert f.dtype == np.uint8


def test_objects_are_bright_against_dark_background():
    frames = generate_sequence(n_frames=5, n_objects=3, settings=_settings())
    # Bright object pixels (255) must exist; background is near zero.
    assert (frames[0] == 255).sum() > 0
    assert frames[0].mean() < 60


def test_determinism():
    a = generate_sequence(5, 3, _settings())
    b = generate_sequence(5, 3, _settings())
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_to_bgr_has_three_channels():
    frames = generate_sequence(1, 1, _settings())
    bgr = to_bgr(frames[0])
    assert bgr.shape == (240, 320, 3)
