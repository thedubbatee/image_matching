
from __future__ import annotations

import cv2
import numpy as np

from detector_helpers import Keypoint, to_grayscale


class DoGDetector:
    """Difference of Gaussians scale-space keypoint detector."""

    def __init__(
        self,
        num_octaves: int = 4,
        scales_per_octave: int = 3,
        sigma: float = 1.6,
        contrast_threshold: float = 0.03,
        edge_threshold: float = 10.0,
    ):
        if num_octaves < 1:
            raise ValueError("num_octaves must be at least 1")
        if scales_per_octave < 1:
            raise ValueError("scales_per_octave must be at least 1")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if contrast_threshold < 0:
            raise ValueError("contrast_threshold must be non-negative")
        if edge_threshold <= 0:
            raise ValueError("edge_threshold must be positive")

        self.num_octaves = num_octaves
        self.scales_per_octave = scales_per_octave
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold

    def detect(self, image: np.ndarray) -> list[Keypoint]:
        gray = to_grayscale(image)

        # Establish the initial Gaussian scale.
        # Input images are assumed to have an initial blur of sigma 0.5.
        assumed_blur = 0.5
        if self.sigma < assumed_blur:
            raise ValueError(
                "sigma must be at least the assumed input blur (0.5)"
            )

        initial_sigma = np.sqrt(self.sigma**2 - assumed_blur**2)
        base = cv2.GaussianBlur(
            gray, (0, 0), sigmaX=initial_sigma
        )

        k = 2.0 ** (1.0 / self.scales_per_octave)
        num_gaussian = self.scales_per_octave + 3
        keypoints: list[Keypoint] = []

        for octave in range(self.num_octaves):
            height, width = base.shape

            if min(height, width) < 8:
                break

            # Build the Gaussian pyramid for this octave.
            gaussian_images = [base]

            for level in range(1, num_gaussian):
                previous_sigma = self.sigma * k ** (level - 1)
                current_sigma = self.sigma * k ** level
                incremental_sigma = np.sqrt(
                    current_sigma**2 - previous_sigma**2
                )

                blurred = cv2.GaussianBlur(
                    gaussian_images[-1],
                    (0, 0),
                    sigmaX=incremental_sigma,
                )
                gaussian_images.append(blurred)

            # D[i] = G[i+1] - G[i].
            dogs = [
                gaussian_images[i + 1] - gaussian_images[i]
                for i in range(num_gaussian - 1)
            ]

            # An extremum must be compared against all 26 neighbours:
            # 8 in the scale below, 8 in the same scale, and 8 above.
            kernel = np.ones((3, 3), dtype=np.uint8)

            for s in range(1, len(dogs) - 1):
                prev_dog = dogs[s - 1]
                curr_dog = dogs[s]
                next_dog = dogs[s + 1]

                max_neighbour = np.maximum(
                    np.maximum(
                        cv2.dilate(prev_dog, kernel),
                        cv2.dilate(curr_dog, kernel),
                    ),
                    cv2.dilate(next_dog, kernel),
                )

                min_neighbour = np.minimum(
                    np.minimum(
                        cv2.erode(prev_dog, kernel),
                        cv2.erode(curr_dog, kernel),
                    ),
                    cv2.erode(next_dog, kernel),
                )

                # Strict comparison avoids many duplicate detections
                # on flat regions and tied neighbouring values.
                extrema = (
                    ((curr_dog > max_neighbour) |
                     (curr_dog < min_neighbour))
                )

                # Exclude the image border because spatial derivatives
                # require pixels on both sides.
                extrema[:1, :] = False
                extrema[-1:, :] = False
                extrema[:, :1] = False
                extrema[:, -1:] = False

                # Reject low-contrast candidates.
                extrema &= (
                    np.abs(curr_dog) >= self.contrast_threshold
                )

                ys, xs = np.nonzero(extrema)

                for y, x in zip(ys, xs):
                    # Spatial Hessian of the DoG image.
                    center = curr_dog[y, x]

                    dxx = (
                        curr_dog[y, x + 1]
                        + curr_dog[y, x - 1]
                        - 2.0 * center
                    )
                    dyy = (
                        curr_dog[y + 1, x]
                        + curr_dog[y - 1, x]
                        - 2.0 * center
                    )
                    dxy = (
                        curr_dog[y + 1, x + 1]
                        - curr_dog[y + 1, x - 1]
                        - curr_dog[y - 1, x + 1]
                        + curr_dog[y - 1, x - 1]
                    ) / 4.0

                    determinant = dxx * dyy - dxy**2
                    trace = dxx + dyy

                    # Reject saddle points and degenerate curvature.
                    if determinant <= 0:
                        continue

                    curvature_ratio = trace**2 / determinant
                    edge_limit = (
                        (self.edge_threshold + 1.0) ** 2
                        / self.edge_threshold
                    )

                    # Reject edge-like responses.
                    if curvature_ratio >= edge_limit:
                        continue

                    octave_scale = 2.0**octave

                    # dogs[s] is between Gaussian levels s and s+1.
                    # Use the geometric mean of their scales.
                    local_sigma = self.sigma * k ** (s + 0.5)
                    original_sigma = local_sigma * octave_scale

                    keypoints.append(
                        Keypoint(
                            x=float(x * octave_scale),
                            y=float(y * octave_scale),
                            sigma=float(original_sigma),
                            response=float(abs(center)),
                        )
                    )

            # The image at level scales_per_octave has twice the
            # base blur, so downsampling it starts the next octave.
            next_base = gaussian_images[self.scales_per_octave]
            base = cv2.resize(
                next_base,
                (next_base.shape[1] // 2, next_base.shape[0] // 2),
                interpolation=cv2.INTER_NEAREST,
            )

        return keypoints
