import unittest

import numpy as np

from verification.verify_detectors import compare_features, summarize


class ComparisonTests(unittest.TestCase):
    def test_inclusive_distance_and_scale_boundaries(self):
        result = compare_features(
            np.array([[0, 0, 8], [10, 0, 12.5], [20, 0, 10]]),
            np.array([[3, 0, 10], [13, 0, 10], [23.01, 0, 10]]))
        self.assertEqual(result['correct'], [True, True, False])

    def test_nearest_position_is_chosen_before_testing_scale(self):
        result = compare_features(np.array([[0, 0, 10]]),
                                  np.array([[1, 0, 20], [2, 0, 10]]))
        self.assertEqual(result['correct'], [False])
        self.assertEqual(result['reference_index'], [0])

    def test_empty_detections_cannot_pass(self):
        for ours, reference in [([], []), ([[0, 0, 1]], [])]:
            result = compare_features(np.array(ours), np.array(reference))
            self.assertFalse(summarize([result])['passed'])

    def test_threshold_uses_pooled_feature_counts(self):
        report = summarize([{'correct': [True] * 9}, {'correct': [False]}])
        self.assertEqual(report['accuracy'], 0.9)
        self.assertTrue(report['passed'])

    def test_invalid_scales_rejected(self):
        with self.assertRaises(ValueError):
            compare_features(np.array([[0, 0, 0]]), np.array([[0, 0, 1]]))


if __name__ == '__main__':
    unittest.main()
