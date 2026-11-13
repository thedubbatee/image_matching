"""End-to-end checks for detector -> SIFT descriptor -> matcher pipelines.

Run from the repository root with:
    pytest -q
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


# The project modules currently use sibling imports (for example,
# ``from dog import DoGDetector``), so add their directory to the import path.
FEATURE_DETECTOR_DIR = Path(__file__).resolve().parents[1] / "feature_detector"
sys.path.insert(0, str(FEATURE_DETECTOR_DIR))

from dog import DoGDetector
from harris import HarrisLaplaceDetector
from matcher import BFMatcher
from sift import sift_describe


def make_test_image() -> np.ndarray:
    """Build a repeatable, textured image with corners and blobs."""
    rng = np.random.default_rng(3821)
    image = np.zeros((256, 256), dtype=np.uint8)

    # Fixed geometric features provide clear corners and scale-space blobs.
    cv2.rectangle(image, (28, 32), (92, 96), 220, thickness=-1)
    cv2.rectangle(image, (148, 28), (218, 78), 150, thickness=4)
    cv2.circle(image, (66, 186), 24, 255, thickness=-1)
    cv2.circle(image, (184, 174), 34, 190, thickness=5)
    cv2.line(image, (112, 116), (230, 228), 255, thickness=5)
    cv2.putText(image, "CV", (104, 72), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 180, thickness=2, lineType=cv2.LINE_AA)

    # Add stable texture so descriptors have distinctive local patterns.
    texture = rng.integers(0, 45, size=image.shape, dtype=np.uint8)
    return cv2.add(image, texture)


def make_test_images() -> dict[str, np.ndarray]:
    """Return several deterministic scenes with different feature patterns."""
    images = {"mixed_shapes_texture": make_test_image()}

    geometry = np.full((256, 256), 24, dtype=np.uint8)
    for row in range(3):
        for col in range(3):
            x, y = 24 + col * 76, 24 + row * 76
            value = 90 + ((row * 3 + col) % 5) * 30
            if (row + col) % 2:
                cv2.circle(geometry, (x + 24, y + 24), 18, value, -1)
                cv2.line(geometry, (x + 5, y + 45), (x + 45, y + 5), 240, 2)
            else:
                cv2.rectangle(geometry, (x, y), (x + 48, y + 48), value, 3)
                cv2.line(geometry, (x + 4, y + 4), (x + 44, y + 44), 230, 2)
    images["repeated_geometry"] = geometry

    rng = np.random.default_rng(2026)
    texture = rng.integers(20, 230, size=(256, 256), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.7)
    cv2.rectangle(texture, (44, 44), (212, 212), 255, 3)
    cv2.circle(texture, (128, 128), 48, 0, 4)
    images["random_texture_with_contours"] = texture

    text_edges = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(text_edges, "SCALE", (18, 92), cv2.FONT_HERSHEY_COMPLEX,
                1.0, 220, 3, cv2.LINE_AA)
    cv2.putText(text_edges, "MATCH", (18, 166), cv2.FONT_HERSHEY_COMPLEX,
                1.0, 180, 3, cv2.LINE_AA)
    cv2.polylines(text_edges,
                  [np.array([[24, 208], [80, 188], [138, 222], [220, 190]],
                            dtype=np.int32)], False, 255, 4)
    images["text_and_edges"] = text_edges
    return images


def rotate_90(image: np.ndarray) -> np.ndarray:
    """Rotate counterclockwise by 90 degrees (exact pixel permutation)."""
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_point_90(x: float, y: float, height: int) -> tuple[float, float]:
    """Map an (x, y) point through the 90-degree counterclockwise rotation."""
    return y, height - 1 - x


def match_geometry(keypoints_a, keypoints_b, matches, height: int) -> tuple[int, float]:
    """Return (match count, fraction within 5 px of the known rotation)."""
    if not matches:
        return 0, 0.0

    errors = []
    for match in matches:
        first = keypoints_a[match.queryIdx if hasattr(match, "queryIdx") else match.query_idx]
        second = keypoints_b[match.trainIdx if hasattr(match, "trainIdx") else match.train_idx]
        first_x, first_y = first.pt if hasattr(first, "pt") else (first.x, first.y)
        second_x, second_y = second.pt if hasattr(second, "pt") else (second.x, second.y)
        expected_x, expected_y = rotate_point_90(first_x, first_y, height)
        errors.append(np.hypot(second_x - expected_x, second_y - expected_y))
    return len(matches), float(np.mean(np.asarray(errors) <= 5.0))


def opencv_sift_pipeline(image: np.ndarray, rotated: np.ndarray):
    """Run OpenCV's built-in detector + SIFT descriptor + cross-check matcher."""
    sift = cv2.SIFT_create()
    keypoints_a, descriptors_a = sift.detectAndCompute(image, None)
    keypoints_b, descriptors_b = sift.detectAndCompute(rotated, None)
    if descriptors_a is None or descriptors_b is None:
        return keypoints_a, keypoints_b, []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(
        matcher.match(descriptors_a, descriptors_b),
        key=lambda match: match.distance,
    )
    return keypoints_a, keypoints_b, matches


@pytest.mark.parametrize(
    "detector",
    [HarrisLaplaceDetector(), DoGDetector()],
    ids=["harris_laplace", "dog"],
)
def test_detector_sift_match_pipeline_is_rotation_consistent(detector):
    image = make_test_image()
    rotated = rotate_90(image)

    keypoints_a = detector.detect(image)
    keypoints_b = detector.detect(rotated)
    described_a, descriptors_a = sift_describe(image, keypoints_a)
    described_b, descriptors_b = sift_describe(rotated, keypoints_b)

    # Descriptors must correspond one-to-one with described keypoints and be
    # valid 128-dimensional SIFT vectors.
    assert descriptors_a.shape == (len(described_a), 128)
    assert descriptors_b.shape == (len(described_b), 128)
    assert np.isfinite(descriptors_a).all()
    assert np.isfinite(descriptors_b).all()
    assert len(described_a) > 0
    assert len(described_b) > 0

    matches = BFMatcher().match(descriptors_a, descriptors_b)
    assert len(matches) >= 5, "expected several repeatable matches after rotation"

    # Check geometric consistency with the known image transformation. Some
    # false matches are expected, so use a robust inlier fraction and tolerance.
    errors = []
    for match in matches:
        first = described_a[match.query_idx]
        second = described_b[match.train_idx]
        expected_x, expected_y = rotate_point_90(
            first.x, first.y, image.shape[0]
        )
        errors.append(np.hypot(second.x - expected_x, second.y - expected_y))

    errors = np.asarray(errors)
    inlier_fraction = np.mean(errors <= 5.0)
    assert inlier_fraction >= 0.5, (
        f"only {inlier_fraction:.0%} of {len(matches)} matches were within "
        "5 pixels of the known rotation"
    )


@pytest.mark.parametrize(
    "detector",
    [HarrisLaplaceDetector(), DoGDetector()],
    ids=["harris_laplace", "dog"],
)
def test_descriptor_arrays_are_empty_with_no_keypoints(detector):
    image = make_test_image()
    keypoints, descriptors = sift_describe(image, [])

    assert keypoints == []
    assert descriptors.shape == (0, 128)
    assert detector.detect(np.zeros_like(image)) == []


@pytest.mark.parametrize(
    "detector",
    [HarrisLaplaceDetector(), DoGDetector()],
    ids=["harris_laplace", "dog"],
)
def test_custom_pipeline_against_opencv_sift(detector):
    """Compare custom detector+SIFT to OpenCV's integrated SIFT pipeline.

    This reports match count and known-transform inlier fraction for both
    methods. It deliberately does not require the custom pipeline to beat or
    equal OpenCV: the detector and descriptor implementations differ.
    """
    for image_name, image in make_test_images().items():
        rotated = rotate_90(image)

        custom_a, custom_desc_a = sift_describe(image, detector.detect(image))
        custom_b, custom_desc_b = sift_describe(rotated, detector.detect(rotated))
        custom_matches = BFMatcher().match(custom_desc_a, custom_desc_b)
        custom_count, custom_inliers = match_geometry(
            custom_a, custom_b, custom_matches, image.shape[0]
        )

        cv_a, cv_b, cv_matches = opencv_sift_pipeline(image, rotated)
        cv_count, cv_inliers = match_geometry(
            cv_a, cv_b, cv_matches, image.shape[0]
        )

        # Ensure the baseline ran and custom pipeline found at least some
        # matches. Quality is reported rather than requiring equality.
        assert cv_count > 0, f"OpenCV SIFT found no matches for {image_name}"
        assert custom_count > 0, (
            f"{type(detector).__name__} found no matches for {image_name}"
        )
        print(
            f"{image_name} | {type(detector).__name__}: "
            f"custom {custom_count} matches / {custom_inliers:.1%} inliers; "
            f"OpenCV SIFT {cv_count} matches / {cv_inliers:.1%} inliers"
        )
