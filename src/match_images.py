"""Match two images with the project's DoG + SIFT-style pipeline.

From the repository root:
    python3 feature_detector/match_images.py image_a.jpg image_b.jpg
"""

import argparse
from pathlib import Path

import cv2

from dog import DoGDetector
from harris import HarrisDetector
from matcher import BFMatcher
from sift import sift_describe


REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_IMAGE_DIMENSION = 1200
MAX_KEYPOINTS = 500


def load_and_resize(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    height, width = image.shape
    largest_dimension = max(height, width)
    if largest_dimension > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / largest_dimension
        new_size = (round(width * scale), round(height * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def describe_image(image, detector):
    keypoints = detector.detect(image)
    # The custom descriptor is implemented with per-pixel Python loops. Keep
    # the strongest points to make this demo practical on large photographs.
    keypoints = sorted(keypoints, key=lambda point: point.response, reverse=True)
    keypoints = keypoints[:MAX_KEYPOINTS]
    return sift_describe(image, keypoints)


def main():
    parser = argparse.ArgumentParser(
        description="Detect and match features in two images."
    )
    parser.add_argument(
        "image_a", nargs="?", type=Path,
        default=REPO_ROOT / "image_a.jpg",
        help="first image (default: image_a.jpg in the repo root)",
    )
    parser.add_argument(
        "image_b", nargs="?", type=Path,
        default=REPO_ROOT / "image_b.jpg",
        help="second image (default: image_b.jpg in the repo root)",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "matches.jpg",
        help="path for the match visualization (default: matches.jpg in the repo root)",
    )
    args = parser.parse_args()

    image_a = load_and_resize(args.image_a)
    image_b = load_and_resize(args.image_b)
    detector = DoGDetector()
    #detector = HarrisDetector()  # Uncomment to use Harris-Laplace detector instead

    keypoints_a, descriptors_a = describe_image(image_a, detector)
    keypoints_b, descriptors_b = describe_image(image_b, detector)
    matches = BFMatcher(ratio=0.6).match(descriptors_a, descriptors_b)
    print(
        f"Image A: {len(keypoints_a)} described keypoints; "
        f"image B: {len(keypoints_b)} described keypoints; "
        f"mutual matches: {len(matches)}"
    )

    cv_keypoints_a = [
        cv2.KeyPoint(float(point.x), float(point.y), float(2 * point.sigma))
        for point in keypoints_a
    ]
    cv_keypoints_b = [
        cv2.KeyPoint(float(point.x), float(point.y), float(2 * point.sigma))
        for point in keypoints_b
    ]
    cv_matches = [
        cv2.DMatch(match.query_idx, match.train_idx, float(match.distance))
        for match in matches[:50]
    ]
    visualization = cv2.drawMatches(
        image_a, cv_keypoints_a,
        image_b, cv_keypoints_b,
        cv_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), visualization):
        raise OSError(f"Could not write output image: {args.output}")
    print(f"Saved match visualization to {args.output}")


if __name__ == "__main__":
    main()
