"""Shared data types and image utilities for Harris-Laplace, DoG and SIFT.

Conventions for all stages:
    x, y  : coordinates in the original image.
    sigma : Gaussian feature scale in original-image pixels.
    angle : dominant orientation in radians, or None if not yet assigned.

The descriptor should consume these common fields, regardless of detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Keypoint:
    """A feature location"""
    x: float # Col
    y: float # Row
    sigma: float # Scale
    response: float = 0.0  # Harris strength or DoG contrast (not comparable).
    angle: float | None = None # Radians


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a 2D image with a separable Gaussian kernel.
     Edges are treated as reflections (e.g. abc becomes cbabcba).
    The result is a np array of floats
    """
    pixels = np.asarray(image, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError("Gaussian blur expects a 2D image")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Gaussian sigma must be finite and non-negative")
    if sigma == 0:
        return pixels.copy()

    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights = (weights / weights.sum()).astype(np.float32)

    height, width = pixels.shape
    padded = np.pad(pixels, ((0, 0), (radius, radius)), mode="reflect")
    horizontal = np.zeros_like(pixels)
    for offset, weight in enumerate(weights):
        horizontal += weight * padded[:, offset:offset + width]

    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="reflect")
    blurred = np.zeros_like(pixels)
    for offset, weight in enumerate(weights):
        blurred += weight * padded[offset:offset + height, :]
    return blurred


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a greyscale or OpenCV-style BGR/BGRA image to greyscale, in float32 from[0, 1].

    Accepted inputs are uint8 images or float arrays with intensities in
    [0, 1] or [0, 255].
    """
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        gray = pixels
    elif pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        # OpenCV channel order: blue, green, red;
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

    # uint8 images are always interpreted as 0..255.
    if np.issubdtype(pixels.dtype, np.integer) or np.max(gray) > 1.0:
        gray /= 255.0
    return gray


def to_opencv_keypoints(points: Sequence[Keypoint]) -> list:
    """Optional conversion for OpenCV drawing, SIFT and benchmarking.

    OpenCV's size is a diameter, while our sigma is a Gaussian standard
    deviation. size=2*sigma is a convenient way to convert but doesn't
    guarantee the same scaling. Angles are
    converted from radians to OpenCV, and None becomes -1.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install opencv-python for OpenCV conversion") from exc

    return [cv2.KeyPoint(
        x=float(p.x), y=float(p.y), size=2.0 * float(p.sigma),
        angle=-1.0 if p.angle is None else float(np.degrees(p.angle) % 360.0),
        response=float(p.response),
    ) for p in points]
