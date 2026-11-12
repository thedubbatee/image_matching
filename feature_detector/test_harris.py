import unittest
from unittest.mock import patch

import numpy as np

from harris import HarrisLaplaceDetector
from sift import sift_describe


class HarrisDoGTests(unittest.TestCase):
    def test_scale_is_selected_by_difference_of_gaussians(self):
        detector = HarrisLaplaceDetector(num_scales=9, dedup_radius=0)
        y, x = np.mgrid[:128, :128]
        image = np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / 32).astype(np.float32)
        corner_response = np.zeros_like(image)
        corner_response[64, 64] = 1

        # Isolate scale selection at a fixed Harris corner. For this blob,
        # adjacent Gaussian differences peak at level 3; LoG peaks at 4.
        with patch('harris._harris_response', return_value=corner_response):
            points = detector.detect(image)

        self.assertEqual(len(points), 1)
        self.assertEqual((points[0].x, points[0].y), (64.0, 64.0))
        self.assertAlmostEqual(points[0].sigma, detector.scales[3])

    def test_harris_points_can_be_described_by_sift(self):
        rng = np.random.default_rng(3)
        blocks = rng.integers(0, 2, (16, 16)).astype(np.float32)
        image = np.kron(blocks, np.ones((8, 8), dtype=np.float32))
        points = HarrisLaplaceDetector(max_points=20).detect(image)
        self.assertTrue(points)
        self.assertTrue(all(point.response > 0 and point.sigma > 0 for point in points))

        described, descriptors = sift_describe(image, points)
        self.assertTrue(described)
        self.assertEqual(descriptors.shape, (len(described), 128))
        self.assertTrue(np.all(np.isfinite(descriptors)))


if __name__ == '__main__':
    unittest.main()
