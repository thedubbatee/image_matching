"""Compare Harris smoothing on a few HPatches image pairs.

Run: .venv/bin/python benchmarks/harris_hpatches.py
Pass --dataset PATH to use another HPatches sequences directory.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import harris


DEFAULT_DATASET = (Path(__file__).resolve().parents[1] / "src/verification/"
                   ".hpatches-cache/hpatches-sequences-release")


def load_image(path: Path, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read HPatches image: {path}")
    height, width = image.shape
    factor = min(1.0, max_side / max(height, width))
    if factor < 1.0:
        image = cv2.resize(image, (round(width * factor), round(height * factor)),
                           interpolation=cv2.INTER_AREA)
    scaled_height, scaled_width = image.shape
    transform = np.diag([scaled_width / width, scaled_height / height, 1.0])
    return image, transform


def coordinates(points: list) -> np.ndarray:
    return np.array([(p.x, p.y) for p in points], dtype=np.float64).reshape(-1, 2)


def project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def inside(points: np.ndarray, image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    return (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) &
            (points[:, 0] < width) & (points[:, 1] >= 0) &
            (points[:, 1] < height))


def repeatability(reference: list, target: list, homography: np.ndarray,
                  ref_image: np.ndarray, target_image: np.ndarray,
                  radius: float) -> tuple[int, int]:
    ref_xy, target_xy = coordinates(reference), coordinates(target)
    if not len(ref_xy) or not len(target_xy):
        return 0, 0

    mapped_ref = project(ref_xy, homography)
    ref_visible = mapped_ref[inside(mapped_ref, target_image)]
    mapped_target = project(target_xy, np.linalg.inv(homography))
    target_visible = target_xy[inside(mapped_target, ref_image)]
    denominator = min(len(ref_visible), len(target_visible))
    if not denominator:
        return 0, 0

    tree = cKDTree(target_visible)
    candidates = []
    for ref_index, near in enumerate(tree.query_ball_point(ref_visible, radius)):
        for target_index in near:
            distance = np.linalg.norm(ref_visible[ref_index] - target_visible[target_index])
            candidates.append((distance, ref_index, target_index))

    used_ref, used_target = set(), set()
    for _, ref_index, target_index in sorted(candidates):
        if ref_index not in used_ref and target_index not in used_target:
            used_ref.add(ref_index)
            used_target.add(target_index)
    return len(used_ref), denominator


def old_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """The SciPy smoothing Harris used before the shared blur helper."""
    return gaussian_filter(image, sigma, mode="reflect")


def timed_detection(image: np.ndarray, blur, repeats: int) -> tuple[list, float]:
    detector = harris.HarrisDetector()
    elapsed = []
    with patch.object(harris, "gaussian_blur", blur):
        for _ in range(repeats):
            start = perf_counter()
            points = detector.detect_points(image)
            elapsed.append(perf_counter() - start)
    return points, statistics.median(elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pairs", nargs="+", default=["i_fruits:2", "v_bark:2"],
                        help="HPatches scene and target image, for example v_bark:2")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.max_side < 8 or args.radius <= 0 or args.repeats < 1:
        parser.error("max-side must be >= 8, radius > 0, and repeats >= 1")

    cv2.setNumThreads(1)
    scores = {"old": [], "new": []}
    times = {"old": [], "new": []}
    for pair in args.pairs:
        scene, target_number = pair.rsplit(":", 1)
        directory = args.dataset / scene
        ref_image, ref_scale = load_image(directory / "1.ppm", args.max_side)
        target_image, target_scale = load_image(
            directory / f"{target_number}.ppm", args.max_side)
        homography = np.loadtxt(directory / f"H_1_{target_number}")
        homography = target_scale @ homography @ np.linalg.inv(ref_scale)

        row = {}
        for name, blur in (("old", old_blur), ("new", harris.gaussian_blur)):
            reference, ref_time = timed_detection(ref_image, blur, args.repeats)
            target, target_time = timed_detection(target_image, blur, args.repeats)
            matched, possible = repeatability(
                reference, target, homography, ref_image, target_image, args.radius)
            score = matched / possible if possible else 0.0
            scores[name].append(score)
            times[name].extend((ref_time, target_time))
            row[name] = (score, matched, possible, len(reference), len(target))

        print(f"{scene} 1->{target_number}:", flush=True)
        for name in ("old", "new"):
            score, matched, possible, count_ref, count_target = row[name]
            print(f"  {name}: {score:.1%} repeatability ({matched}/{possible}), "
                  f"keypoints {count_ref}/{count_target}", flush=True)

    print("Summary (mean across pairs; median detection seconds per image):")
    for name in ("old", "new"):
        print(f"  {name}: {statistics.mean(scores[name]):.1%} repeatability, "
              f"{statistics.median(times[name]):.3f} s/image")
    print(f"  new/old runtime: "
          f"{statistics.median(times['new']) / statistics.median(times['old']):.2f}x")


if __name__ == "__main__":
    main()
