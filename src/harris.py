"""Harris corner detector with Difference of Gaussians scale selection.

Finds harris keypoints at several scales, then selects characteristic
scales using the magnitude of adjacent Gaussian differences.

Images are assumed to be greyscale or OpenCV-style BGR (H, W, 3) arrays.
"""

from __future__ import annotations

import numpy as np
from detector_helpers import Keypoint, gaussian_blur, to_grayscale, to_opencv_keypoints
from scipy.ndimage import gaussian_filter, maximum_filter


def _harris_response(gray: np.ndarray, sigma_i: float, k: float,
                     differentiation_ratio: float) -> np.ndarray:
    """Scale-adapted Harris response using the Gaussian second moment matrix.

    M = G(sigma_I) * [[Lx^2, Lx*Ly], [Lx*Ly, Ly^2]],
    where Lx, Ly are scale-normalised derivatives of G(sigma_D)*I,
    sigma_D = differentiation_ratio * sigma_I, and R = det(M)-k tr(M)^2.
    """
    sigma_d = differentiation_ratio * sigma_i
    # Gaussian derivatives, multiplied by sigma_D to normalise their scale.
    ix = gaussian_filter(gray, sigma_d, order=(0, 1), mode='reflect') * sigma_d
    iy = gaussian_filter(gray, sigma_d, order=(1, 0), mode='reflect') * sigma_d
    a = gaussian_blur(ix * ix, sigma_i)
    b = gaussian_blur(iy * iy, sigma_i)
    c = gaussian_blur(ix * iy, sigma_i)
    return a * b - c * c - k * (a + b) ** 2


class HarrisDetector:
    """Harris corner detector with DoG scale selection.

    Parameters:
      sigma0: first Gaussian integration scale in input pixels.
      scale_factor: geometric step between successive sigma levels; 1.4 in paper.
      num_scales: number of scales; first and last are DoG guard levels.
      k: Harris free parameter, normally 0.04-0.06.
      differentiation_ratio: sigma_D / sigma_I; 0.7 in the paper.
      harris_threshold: relative to the largest positive Harris response PER SCALE.
      max_points: max returned keypoints (after nonmaximum suppression).
      dedup_radius: avoid duplicate corners across neighbouring scales;
    """

    def __init__(self, sigma0: float = 1.2, scale_factor: float = 1.4,
                 num_scales: int = 12, k: float = 0.04,
                 differentiation_ratio: float = 0.7,
                 harris_threshold: float = 0.01,
                 max_points: int | None = 500,
                 dedup_radius: float = 1.0):
        if sigma0 <= 0 or scale_factor <= 1 or num_scales < 3:
            raise ValueError('Require sigma0>0, scale_factor>1, num_scales>=3')
        if not 0 < k < 0.25 or differentiation_ratio <= 0:
            raise ValueError('Require 0<k<0.25 and differentiation_ratio>0')
        if not 0 <= harris_threshold < 1:
            raise ValueError('Harris threshold must lie in [0,1)')
        if max_points is not None and max_points < 1:
            raise ValueError('max_points must be positive or None')
        if dedup_radius < 0:
            raise ValueError('dedup_radius must be nonnegative')

        self.scales = tuple(float(sigma0 * scale_factor ** i)
                            for i in range(num_scales))
        self.k = k
        self.differentiation_ratio = differentiation_ratio
        self.harris_threshold = harris_threshold
        self.max_points = max_points
        self.dedup_radius = dedup_radius
        self.scale_factor = scale_factor

    def detect_points(self, image: np.ndarray) -> list[Keypoint]:
        """Detect Harris corners, sorting by decreasing strength of response.
        For each detected harris corner, select the scale that is a local max across the scale axis.
        """
        gray = to_grayscale(image)
        h, w = gray.shape

        gaussians = [gaussian_blur(gray, sigma)
                     for sigma in (*self.scales, self.scales[-1] * self.scale_factor)]
        # Adds an extra scale for checking scale maximum, but doesn't take keypoints in that extra scale
        dogs = [np.abs(gaussians[i] - gaussians[i + 1])
                for i in range(len(self.scales))]

        candidates: list[Keypoint] = []

        for i in range(1, len(self.scales) - 1):
            sigma = self.scales[i] * (self.scale_factor ** 0.5)
            response = _harris_response(gray, sigma, self.k,
                                        self.differentiation_ratio)
            top = float(np.max(response)) # highest harris response
            if top <= 0:
                continue
            is_spatial_maximum = (response == maximum_filter(
                response, size=3, mode='nearest')) # boolean array of maximums
            is_corner = is_spatial_maximum & (
                # finds maximums above a threshold, relative to the greatest harris response found
                response > max(0.0, self.harris_threshold * top))

            curr, prev, nxt = (dogs[i], dogs[i-1], dogs[i+1])
            is_scale_maximum = (curr > prev) & (curr > nxt)
            # filter for points that are local maxima in scale space
            mask = is_corner & is_scale_maximum

            # filter out points near the border, which rely on reflected pixels
            border = max(2, int(np.ceil(3 * sigma)))
            if border * 2 >= min(h, w):
                continue

            mask[:border, :] = False
            mask[-border:, :] = False
            mask[:, :border] = False
            mask[:, -border:] = False

            ys, xs = np.nonzero(mask)
            candidates.extend(
                Keypoint(float(x), float(y), sigma,
                         float(response[y, x]))
                for y, x in zip(ys, xs))

        candidates.sort(key=lambda p: (-p.response, p.y, p.x, p.sigma))
        if not self.dedup_radius:
            # just return the found points if we don't care about deduplicating
            return (candidates[:self.max_points]
                    if self.max_points is not None else candidates)

        selected: list[Keypoint] = [] # list of selected points
        # grid of selected points so we can search by location quicker
        cell_size = max(2.0, 2 * self.scales[-2] * max(1.0, self.dedup_radius))

        # self.scales[-2] is the largest scale we check for keypoints
        grid: dict[tuple[int, int], list[Keypoint]] = {}
        for point in candidates:
            cx = int(point.x // cell_size)
            cy = int(point.y // cell_size)
            duplicate = False
            # We break the image into cells, and only check neighbouring cells for duplicates.
            for gy in range(cy - 1, cy + 2):
                for gx in range(cx - 1, cx + 2):
                    for other in grid.get((gx, gy), []):
                        # Keep keypoints detected at widely different scales distinct.
                        if max(point.sigma, other.sigma) > (
                            self.scale_factor ** 2.1) * min(point.sigma, other.sigma):
                            continue

                        dist_sq = ((point.x - other.x) ** 2 +
                                   (point.y - other.y) ** 2)
                        # duplicate if distance is below min_dist
                        min_dist = self.dedup_radius * min(point.sigma, other.sigma)
                        if dist_sq < min_dist * min_dist:
                            duplicate = True
                            break
                    if duplicate:
                        break
                if duplicate:
                    break
            if not duplicate:
                selected.append(point)
                grid.setdefault((cx, cy), []).append(point)
                # if we've found the max_points number of points, finish up.
                if self.max_points is not None and len(selected) >= self.max_points:
                    break
        return selected

    def detect(self, image: np.ndarray) -> list[Keypoint]:
        """Return Keypoints for SIFT descriptor.

        The positions and sigma are in original-image pixels. Orientation
        is assigned by the SIFT algorithm
        """
        return self.detect_points(image)

    def detect_cv2(self, image: np.ndarray):
        """Returns points in openCV format for testing purposes."""
        return to_opencv_keypoints(self.detect(image))
