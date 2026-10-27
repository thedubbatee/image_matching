"""Shared data types and image utilities for Harris–Laplace, DoG and SIFT.

Conventions for all stages:
    x, y  : coordinates in the ORIGINAL image (pixels).
    sigma : Gaussian feature scale in ORIGINAL-image pixels.
    angle : dominant orientation in degrees, or None if not yet assigned.

The descriptor should consume these common fields, regardless of detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Keypoint:
    """A feature location, independent of the algorithm that detected it."""

    x: float
    y: float
    sigma: float
    response: float = 0.0  # Harris strength or DoG contrast (not comparable).
    angle: float | None = None


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or OpenCV-style BGR/BGRA image to float32 [0, 1].

    Accepted inputs are uint8 images or float arrays with intensities in
    [0, 1] or [0, 255]. Three-channel input is assumed to be BGR.
    """
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        gray = pixels
    elif pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        # OpenCV channel order: blue, green, red; ignore optional alpha.
        gray = (0.114 * pixels[..., 0] +
                0.587 * pixels[..., 1] +
                0.299 * pixels[..., 2])
    else:
        raise ValueError("Expected a 2D grayscale or 3/4-channel BGR image")

    if gray.shape[0] < 8 or gray.shape[1] < 8:
        raise ValueError("Image must be at least 8 by 8 pixels")

    gray = np.asarray(gray, dtype=np.float32)
    if not np.all(np.isfinite(gray)):
        raise ValueError("Image contains NaN or infinite pixel values")
    
    if np.min(gray) < 0 or np.max(gray) > 255:
        raise ValueError("Pixel values must be in [0, 1] or [0, 255]")

    # uint8 images are always interpreted as 0..255, even if very dark.
    if np.issubdtype(pixels.dtype, np.integer) or np.max(gray) > 1.0:
        gray /= 255.0
    return gray


def to_opencv_keypoints(points: Sequence[Keypoint]) -> list:
    """Optional conversion for OpenCV drawing, SIFT and benchmarking.

    OpenCV's size is a diameter, while our sigma is a Gaussian standard
    deviation. size=2*sigma is a convenient *adapter convention*, not a
    guarantee of identical descriptor support-region scaling.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install opencv-python for OpenCV conversion") from exc

    return [cv2.KeyPoint(
        x=float(p.x), y=float(p.y), size=2.0 * float(p.sigma),
        angle=-1.0 if p.angle is None else float(p.angle),
        response=float(p.response),
    ) for p in points]
