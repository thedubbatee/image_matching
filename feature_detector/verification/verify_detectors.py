#!/usr/bin/env python3
"""Compare custom detectors to OpenCV on unresized HPatches sequence images.

This is a detector agreement check, not the official descriptor benchmark.
Each custom feature selects its nearest OpenCV feature by Euclidean position
first; references may be reused. Scale ratio = custom sigma / reference sigma.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from scipy.spatial import cKDTree

from dog import DoGDetector
from harris import HarrisLaplaceDetector


def compare_features(ours: np.ndarray, reference: np.ndarray) -> dict:
    """Inputs contain rows (x, y, sigma) in original-image pixels."""
    arrays = []
    for points in (ours, reference):
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            points = np.empty((0, 3))
        if (points.ndim != 2 or points.shape[1] != 3 or
                not np.isfinite(points).all() or np.any(points[:, 2] <= 0)):
            raise ValueError('Features must be finite (x, y, positive sigma) rows')
        arrays.append(points)
    ours, reference = arrays
    count = len(ours)
    if not count or not len(reference):
        return dict(reference_count=len(reference), reference_index=[None] * count,
                    distance_px=[None] * count, scale_ratio=[None] * count,
                    correct=[False] * count)
    distances, indices = cKDTree(reference[:, :2]).query(ours[:, :2])
    ratios = ours[:, 2] / reference[indices, 2]
    correct = (distances <= 3.0) & (ratios >= 0.8) & (ratios <= 1.25)
    return dict(reference_count=len(reference), reference_index=indices.tolist(),
                distance_px=distances.tolist(), scale_ratio=ratios.tolist(),
                correct=correct.tolist())


def summarize(results: list[dict]) -> dict:
    total = sum(len(row['correct']) for row in results)
    correct = sum(sum(row['correct']) for row in results)
    accuracy = correct / total if total else 0.0
    return dict(features=total, correct=correct, accuracy=accuracy,
                passed=total > 0 and accuracy >= 0.9)


def make_detectors(names: list[str]) -> dict:
    detectors = {}
    for name in names:
        if name == 'dog':
            custom = DoGDetector()
            reference = cv2.SIFT_create(
                nOctaveLayers=custom.scales_per_octave,
                contrastThreshold=custom.contrast_threshold,
                edgeThreshold=custom.edge_threshold, sigma=custom.sigma)
            divisor = 2.0  # SIFT size = 2 * Gaussian sigma.
        else:
            module = getattr(cv2, 'xfeatures2d', None)
            factory = getattr(module, 'HarrisLaplaceFeatureDetector_create', None)
            if factory is None:
                raise RuntimeError(
                    'Harris–Laplace requires opencv-contrib-python. Install it in '
                    'a separate virtual environment with numpy and scipy; do not '
                    'install multiple OpenCV wheel variants together.')
            custom = HarrisLaplaceDetector()
            reference = factory(maxCorners=5000)
            # OpenCV source: size = 3 * 2**(octave-1) * si * 2.
            # Our sigma is the Harris integration scale si in image pixels.
            divisor = 6.0
        detectors[name] = (custom, reference, divisor)
    return detectors


def find_images(root: Path) -> list[Path]:
    """Require full, six-image sequences; reject extracted patch mosaics."""
    if (root / 'hpatches-sequences-release').is_dir():
        root = root / 'hpatches-sequences-release'
    sequences = sorted(p for p in root.iterdir()
                       if p.is_dir() and p.name.startswith(('i_', 'v_')))
    if not sequences:
        raise ValueError(f'No HPatches sequence directories found in {root}')
    images = []
    for sequence in sequences:
        expected = [sequence / f'{i}.ppm' for i in range(1, 7)]
        missing = [str(p) for p in expected if not p.is_file()]
        if missing:
            raise ValueError(f'Incomplete full-image sequence: {missing[0]} missing')
        images.extend(expected)
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path,
                        help='Extracted hpatches-sequences-release directory')
    parser.add_argument('--detector', choices=['dog', 'harris', 'both'], default='both')
    parser.add_argument('--limit', type=int, help='Smoke test on first N images only')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('detector_verification.json'))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    try:
        images = find_images(args.dataset)
        available_images = len(images)
        if args.limit:
            images = images[:args.limit]
        names = ['dog', 'harris'] if args.detector == 'both' else [args.detector]
        detectors = make_detectors(names)
        rows = {name: [] for name in names}
        for index, path in enumerate(images, 1):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f'Cannot decode {path}')
            for name, (custom, reference, divisor) in detectors.items():
                points = custom.detect(image)
                baseline = reference.detect(image, None)
                ours = np.array([(p.x, p.y, p.sigma) for p in points])
                theirs = np.array([(*p.pt, p.size / divisor) for p in baseline])
                result = compare_features(ours, theirs)
                result['image'] = str(path)
                result['summary'] = summarize([result])
                rows[name].append(result)
                summary = result['summary']
                print(f'[{index}/{len(images)}] {path.parent.name}/{path.name} '
                      f'{name}: {summary["correct"]}/{summary["features"]} '
                      f'({summary["accuracy"]:.1%})', flush=True)
        summaries = {name: summarize(results) for name, results in rows.items()}
        report = dict(opencv_version=cv2.__version__, dataset=str(args.dataset.resolve()),
                      available_images=available_images, evaluated_images=len(images),
                      limited=len(images) < available_images,
                      protocol=dict(radius_px=3, scale_ratio=[0.8, 1.25],
                                    required_accuracy=0.9, matching='nearest spatial; many-to-one',
                                    reference_size_divisor={'dog': 2, 'harris': 6}),
                      summary=summaries, images=rows)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        for name, result in summaries.items():
            print(f'{name}: {"PASS" if result["passed"] else "FAIL"} '
                  f'{result["accuracy"]:.2%} ({result["correct"]}/{result["features"]})')
        print(f'Report: {args.output}')
        return 0 if all(s['passed'] for s in summaries.values()) else 1
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
