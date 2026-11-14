import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import detector_helpers
import harris
import sift
from dog import DoGDetector


class GaussianBlurTests(unittest.TestCase):
    def test_impulse_has_gaussian_weights_and_preserves_brightness(self):
        blur = getattr(detector_helpers, "gaussian_blur", None)
        self.assertIsNotNone(blur)

        image = np.zeros((9, 9), dtype=np.float32)
        image[4, 4] = 1.0
        result = blur(image, 1.0)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(result[4, 5] / result[4, 4]),
                               float(np.exp(-0.5)), places=5)
        self.assertAlmostEqual(float(result[5, 4]), float(result[4, 5]), places=6)

    def test_constant_image_stays_constant_at_borders(self):
        blur = getattr(detector_helpers, "gaussian_blur", None)
        self.assertIsNotNone(blur)

        image = np.full((8, 8), 0.7, dtype=np.float32)
        result = blur(image, 1.4)

        np.testing.assert_allclose(result, image, rtol=0, atol=1e-6)

    def test_dog_detection_does_not_need_opencv_gaussian_blur(self):
        image = np.random.default_rng(12).integers(0, 256, (128, 128), dtype=np.uint8)
        with patch.object(cv2, "GaussianBlur", side_effect=AssertionError(
                "DoG still called OpenCV GaussianBlur")):
            points = DoGDetector(num_octaves=1, contrast_threshold=0.005).detect(image)

        self.assertGreater(len(points), 0)

    def test_harris_uses_scipy_only_for_gaussian_derivatives(self):
        image = np.random.default_rng(7).uniform(0, 255, (128, 128))
        scipy_gaussian_filter = harris.gaussian_filter

        def derivative_only(*args, **kwargs):
            if kwargs.get("order", 0) == 0:
                raise AssertionError("Harris still uses SciPy for ordinary blur")
            return scipy_gaussian_filter(*args, **kwargs)

        with patch.object(harris, "gaussian_filter", side_effect=derivative_only):
            points = harris.HarrisDetector(
                num_scales=5, max_points=50).detect_points(image)

        self.assertGreater(len(points), 0)

    def test_sift_describes_keypoint_without_scipy_blur(self):
        image = np.random.default_rng(9).integers(0, 256, (96, 96), dtype=np.uint8)
        point = detector_helpers.Keypoint(48, 48, 2.0, angle=0.0)

        with patch.object(sift, "gaussian_filter", side_effect=AssertionError(
                "SIFT still called SciPy gaussian_filter"), create=True):
            keypoints, descriptors = sift.sift_describe(image, [point])

        self.assertEqual(len(keypoints), 1)
        self.assertEqual(descriptors.shape, (1, 128))
        self.assertTrue(np.all(np.isfinite(descriptors)))
        self.assertAlmostEqual(float(np.linalg.norm(descriptors[0])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
