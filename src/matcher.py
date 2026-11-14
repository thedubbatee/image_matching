"""Brute-force feature descriptor matcher with cross-checking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import cv2
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class Match:
    """Matched descriptor pair between two images."""

    query_idx: int
    train_idx: int
    distance: float


class BFMatcher:
    """Brute-force L2 descriptor matcher with mutual cross-checking"""

    def __init__(self, cross_check: bool = True, ratio: float | None = None):
        if ratio is not None and not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.cross_check = cross_check
        self.ratio = ratio

    def match(
        self,
        query_descriptors: np.ndarray,
        train_descriptors: np.ndarray,
    ) -> list[Match]:
        """Finds best match for each descriptor, sorted by distance"""
        if query_descriptors is None or train_descriptors is None:
            return []

        desc1 = np.asarray(query_descriptors, dtype=np.float32)
        desc2 = np.asarray(train_descriptors, dtype=np.float32)

        if len(desc1) == 0 or len(desc2) == 0:
            return []

        dist_matrix = cdist(desc1, desc2, metric="euclidean")
        best_train_for_query = np.argmin(dist_matrix, axis=1)
        best_dist = dist_matrix[np.arange(len(desc1)), best_train_for_query]

        keep = np.ones(len(desc1), dtype=bool)

        if self.cross_check:
            best_query_for_train = np.argmin(dist_matrix, axis=0)
            keep &= best_query_for_train[best_train_for_query] == np.arange(len(desc1))

        if self.ratio is not None and dist_matrix.shape[1] >= 2:
            # Second-smallest distance in each row (partition is faster than a full sort).
            second_dist = np.partition(dist_matrix, 1, axis=1)[:, 1]
            keep &= best_dist < self.ratio * second_dist

        matches = [Match(int(i), int(best_train_for_query[i]), float(best_dist[i]))
                   for i in np.nonzero(keep)[0]]
        matches.sort(key=lambda m: m.distance)
        return matches

    def match_cv2(
        self,
        query_descriptors: np.ndarray,
        train_descriptors: np.ndarray,
    ) -> list:
        """Return matches converted to opencv cv2.DMatch objects."""
        return to_opencv_matches(
            self.match(query_descriptors, train_descriptors)
        )


def to_opencv_matches(matches: Sequence[Match]) -> list:
    """Convert Match objects to cv2.DMatch for opencv visualization."""
    opencv_matches = []
    
    for match in matches:
        dmatch = cv2.DMatch(match.query_idx, match.train_idx, match.distance)
        opencv_matches.append(dmatch)
        
    return opencv_matches