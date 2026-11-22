"""Transformation-Based Feature Detector Correctness Testing (Task 2).

Instead of comparing against OpenCV's implementations (which only measures
similarity to OpenCV's internal design choices), this script evaluates detector
correctness directly across known geometric/photometric transformations:

  1. Given a reference image I_1 and a transformed image I_k with known 3x3
     ground-truth homography matrix H_{1,k} (e.g. from each 6-image sequence
     1.ppm..6.ppm and H_1_2..H_1_6 in the HPatches dataset):
  2. Detect keypoints in I_1 and I_k using our unmodified DoGDetector and
     HarrisLaplaceDetector.
  3. Project each reference keypoint p_1 = (x_1, y_1) onto the transformed image
     coordinate frame via p_1' = Pi(H_{1,k} [x_1, y_1, 1]^T), and scale its
     characteristic scale sigma_1 by the local Jacobian scale factor of H_{1,k}:
         sigma_1' = sigma_1 * sqrt(|det J_H(x_1, y_1)|).
  4. Filter keypoints to the shared field-of-view (FOV) so out-of-frame points
     after warping are not penalised.
  5. Apply the exact same correctness criteria from Table 1 of the Progression Check:
     - Position Match: a detected keypoint in I_k lies within 3.0 pixels of p_1'.
     - Scale Match: the scale ratio sigma_1' / sigma_k lies within [0.8, 1.25].
     - Position + Scale Match: both criteria hold simultaneously for the same
       keypoint in I_k.

Performance Note:
  - Target images I_2..I_6 are detected ONCE per image (not inside the detector
    comparison loop), and spatial neighbour lookups use `scipy.spatial.cKDTree`
    (O(N log N) and O(1) memory instead of O(N^2) dense `cdist` matrices that
    exhaust RAM on high-resolution HPatches images with 10,000+ DoG keypoints).
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree

from detector_helpers import Keypoint
from dog import DoGDetector
from harris import HarrisLaplaceDetector


POS_RADIUS_PX = 3.0
SCALE_RATIO_MIN = 0.8
SCALE_RATIO_MAX = 1.25


def project_keypoints_with_homography(
    keypoints: Sequence[Keypoint],
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map reference keypoints (x, y, sigma) through 3x3 homography H.

    Returns:
      projected_xy: (N, 2) array of expected (x', y') coordinates on the transformed image.
      projected_sigma: (N,) array of expected scales sigma' on the transformed image,
        scaled by the local area change sqrt(|det J_H(x, y)|) = sqrt(|det(H)| / |w|^3).
    """
    if not keypoints:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    pts = np.array([[kp.x, kp.y, 1.0] for kp in keypoints], dtype=np.float64)
    sigmas = np.array([kp.sigma for kp in keypoints], dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)

    warped = pts @ H.T
    w = np.where(np.abs(warped[:, 2]) > 1e-9, warped[:, 2], 1e-9)
    proj_xy = warped[:, :2] / w[:, None]

    # Exact local scale factor from the 2x2 Jacobian determinant of the homography at (x, y)
    det_H = float(np.linalg.det(H))
    local_scale = np.sqrt(np.abs(det_H) / np.maximum(np.abs(w) ** 3, 1e-12))
    proj_sigma = sigmas * local_scale

    return proj_xy, proj_sigma


def evaluate_pair_transformation_correctness(
    ref_keypoints: Sequence[Keypoint],
    transformed_keypoints: Sequence[Keypoint],
    H_1_to_k: np.ndarray,
    transformed_shape: tuple[int, int],
    pos_radius: float = POS_RADIUS_PX,
    scale_bounds: tuple[float, float] = (SCALE_RATIO_MIN, SCALE_RATIO_MAX),
) -> tuple[int, int, int, int]:
    """Compare reference keypoints projected by H_1_to_k against transformed_keypoints.

    Uses a 2D KD-Tree (`scipy.spatial.cKDTree`) instead of a dense N x M distance
    matrix so pairs with 15,000+ DoG keypoints evaluate in <5 ms and O(N) memory.

    Returns:
      (total_in_fov, pos_matches, scale_matches, pos_and_scale_matches)
    """
    if not ref_keypoints:
        return 0, 0, 0, 0

    proj_xy, proj_sigma = project_keypoints_with_homography(
        ref_keypoints, H_1_to_k
    )

    # Keep only reference keypoints whose expected position lies inside the transformed image
    h_t, w_t = transformed_shape[:2]
    in_bounds = (
        (proj_xy[:, 0] >= 0.0)
        & (proj_xy[:, 0] < w_t)
        & (proj_xy[:, 1] >= 0.0)
        & (proj_xy[:, 1] < h_t)
    )
    proj_xy = proj_xy[in_bounds]
    proj_sigma = proj_sigma[in_bounds]
    total_in_fov = int(len(proj_xy))

    if total_in_fov == 0 or not transformed_keypoints:
        return total_in_fov, 0, 0, 0

    trans_xy = np.array(
        [[kp.x, kp.y] for kp in transformed_keypoints], dtype=np.float64
    )
    trans_sigma = np.maximum(
        np.array([kp.sigma for kp in transformed_keypoints], dtype=np.float64),
        1e-7,
    )

    s_min, s_max = scale_bounds
    tree = cKDTree(trans_xy)

    # Nearest spatial neighbour for each projected reference keypoint
    _, nearest_idx = tree.query(proj_xy, k=1)
    nearest_ratios = proj_sigma / trans_sigma[nearest_idx]
    nearest_scale_ok = (nearest_ratios >= s_min) & (nearest_ratios <= s_max)

    # All candidate neighbours within `pos_radius` (3.0 px)
    neighbours_list = tree.query_ball_point(proj_xy, r=pos_radius)

    pos_matches = 0
    scale_matches = 0
    pos_and_scale_matches = 0

    for i, nbrs in enumerate(neighbours_list):
        has_pos = len(nbrs) > 0
        has_joint = False
        if has_pos:
            pos_matches += 1
            ratios = proj_sigma[i] / trans_sigma[nbrs]
            has_joint = bool(np.any((ratios >= s_min) & (ratios <= s_max)))
            if has_joint:
                pos_and_scale_matches += 1
        if nearest_scale_ok[i] or has_joint:
            scale_matches += 1

    return total_in_fov, pos_matches, scale_matches, pos_and_scale_matches


def _find_image_file(seq_dir: Path, idx: int) -> Path | None:
    for ext in (".ppm", ".png", ".jpg", ".jpeg", ".pgm"):
        p = seq_dir / f"{idx}{ext}"
        if p.is_file():
            return p
    return None


def _process_single_sequence(
    seq_dir_str: str,
) -> tuple[str, dict[str, dict[str, int]], float]:
    """Worker function to evaluate all 6 images (1.ppm..6.ppm) in one HPatches sequence."""
    t0 = time.time()
    seq_dir = Path(seq_dir_str)
    detectors = {
        "DoGDetector": DoGDetector(),
        "HarrisLaplaceDetector": HarrisLaplaceDetector(),
    }
    seq_totals = {
        name: {"total": 0, "pos": 0, "scale": 0, "both": 0}
        for name in detectors
    }

    ref_path = _find_image_file(seq_dir, 1)
    if ref_path is None:
        return seq_dir.name, seq_totals, 0.0
    img1 = cv2.imread(str(ref_path))
    if img1 is None:
        return seq_dir.name, seq_totals, 0.0

    ref_kps = {name: det.detect(img1) for name, det in detectors.items()}

    for k in range(2, 7):
        tgt_path = _find_image_file(seq_dir, k)
        h_path = seq_dir / f"H_1_{k}"
        if tgt_path is None or not h_path.is_file():
            continue
        img_k = cv2.imread(str(tgt_path))
        H_1_k = np.loadtxt(str(h_path), dtype=np.float64)
        if img_k is None or H_1_k.shape != (3, 3):
            continue

        for name, det in detectors.items():
            kp_k = det.detect(img_k)
            tot, pos, sc, both = evaluate_pair_transformation_correctness(
                ref_kps[name], kp_k, H_1_k, img_k.shape[:2]
            )
            seq_totals[name]["total"] += tot
            seq_totals[name]["pos"] += pos
            seq_totals[name]["scale"] += sc
            seq_totals[name]["both"] += both

    return seq_dir.name, seq_totals, time.time() - t0


def generate_synthetic_test_images(
    base_image: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Build a small self-contained set of known transformations (illumination, rotation, scale)
    so correctness_test.py works out-of-the-box even when HPatches is not yet downloaded.
    """
    if base_image is None:
        H_img, W_img = 240, 320
        base_image = np.full((H_img, W_img, 3), 35, dtype=np.uint8)
        for r in range(30, 210, 36):
            for c in range(40, 280, 36):
                if ((r // 36) + (c // 36)) % 2 == 0:
                    cv2.rectangle(
                        base_image, (c, r), (c + 24, r + 24), (220, 220, 220), -1
                    )
                else:
                    cv2.circle(
                        base_image, (c + 12, r + 12), 9, (185, 200, 235), -1
                    )
        base_image = cv2.GaussianBlur(base_image, (3, 3), sigmaX=0.8)

    h, w = base_image.shape[:2]
    center = (w / 2.0, h / 2.0)

    pairs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    specs = [
        ("illumination_shift", 0.0, 1.0, 0.85, -15.0),
        ("rot_5deg", 5.0, 1.0, 1.0, 0.0),
        ("rot_10deg_scale_1.1", 10.0, 1.1, 1.0, 0.0),
        ("rot_-8deg_scale_0.9", -8.0, 0.9, 1.0, 0.0),
        ("rot_15deg_scale_1.2", 15.0, 1.2, 1.0, 0.0),
    ]
    for name, angle_deg, scale, alpha, beta in specs:
        M_2x3 = cv2.getRotationMatrix2D(center, angle_deg, scale)
        H_mat = np.vstack([M_2x3, [0.0, 0.0, 1.0]])
        warped = cv2.warpPerspective(
            base_image,
            H_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        warped = np.clip(warped.astype(np.float32) * alpha + beta, 0, 255).astype(
            np.uint8
        )
        pairs.append((name, base_image, warped, H_mat))

    return pairs


def run_correctness_benchmark(
    dataset_dir: str | Path | None = None,
    workers: int | None = None,
    max_sequences: int | None = None,
) -> None:
    """Run the transformation-based correctness test and print Table 1 formatted results."""
    totals = {
        "DoGDetector": {"total": 0, "pos": 0, "scale": 0, "both": 0},
        "HarrisLaplaceDetector": {"total": 0, "pos": 0, "scale": 0, "both": 0},
    }

    if dataset_dir is not None and Path(dataset_dir).is_dir():
        root = Path(dataset_dir)
        seq_dirs = sorted(
            d
            for d in root.iterdir()
            if d.is_dir() and _find_image_file(d, 1) is not None
        )
        if max_sequences is not None:
            seq_dirs = seq_dirs[:max_sequences]

        num_seqs = len(seq_dirs)
        num_workers = workers or max(1, (os.cpu_count() or 2) - 1)
        print(
            f"Running transformation correctness test on {num_seqs} HPatches sequences "
            f"using {num_workers} parallel CPU workers...",
            flush=True,
        )
        start_time = time.time()

        seq_paths = [str(d) for d in seq_dirs]
        if num_workers == 1:
            iterator = map(_process_single_sequence, seq_paths)
            for seq_idx, (seq_name, seq_totals, seq_sec) in enumerate(
                iterator, start=1
            ):
                for name in totals:
                    for k in ("total", "pos", "scale", "both"):
                        totals[name][k] += seq_totals[name][k]
                elapsed = time.time() - start_time
                eta = (elapsed / seq_idx) * (num_seqs - seq_idx)
                dog_t = totals["DoGDetector"]
                hl_t = totals["HarrisLaplaceDetector"]
                dog_both = (
                    100.0 * dog_t["both"] / dog_t["total"]
                    if dog_t["total"]
                    else 0.0
                )
                hl_both = (
                    100.0 * hl_t["both"] / hl_t["total"]
                    if hl_t["total"]
                    else 0.0
                )
                print(
                    f"[{seq_idx:3d}/{num_seqs}] {seq_name:<20} "
                    f"| {seq_sec:4.1f}s (ETA: {eta/60:4.1f}m) "
                    f"| Pos+Scale -> DoG: {dog_both:5.1f}%, Harris: {hl_both:5.1f}%",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                for seq_idx, (seq_name, seq_totals, seq_sec) in enumerate(
                    pool.map(_process_single_sequence, seq_paths), start=1
                ):
                    for name in totals:
                        for k in ("total", "pos", "scale", "both"):
                            totals[name][k] += seq_totals[name][k]
                    elapsed = time.time() - start_time
                    eta = (elapsed / seq_idx) * (num_seqs - seq_idx)
                    dog_t = totals["DoGDetector"]
                    hl_t = totals["HarrisLaplaceDetector"]
                    dog_both = (
                        100.0 * dog_t["both"] / dog_t["total"]
                        if dog_t["total"]
                        else 0.0
                    )
                    hl_both = (
                        100.0 * hl_t["both"] / hl_t["total"]
                        if hl_t["total"]
                        else 0.0
                    )
                    print(
                        f"[{seq_idx:3d}/{num_seqs}] {seq_name:<20} "
                        f"| {seq_sec:4.1f}s (ETA: {eta/60:4.1f}m) "
                        f"| Pos+Scale -> DoG: {dog_both:5.1f}%, Harris: {hl_both:5.1f}%",
                        flush=True,
                    )
    else:
        detectors = {
            "DoGDetector": DoGDetector(),
            "HarrisLaplaceDetector": HarrisLaplaceDetector(),
        }
        base_img = cv2.imread("image.jpg")
        pairs = generate_synthetic_test_images(base_img)
        print(
            f"Running transformation correctness test on {len(pairs)} known image transformations "
            "(pass --dataset /path/to/hpatches to run on full HPatches)...",
            flush=True,
        )
        kp1_cache = {
            name: det.detect(pairs[0][1]) for name, det in detectors.items()
        }
        for pair_idx, (pair_name, _, img_k, H_1_k) in enumerate(
            pairs, start=1
        ):
            for name, det in detectors.items():
                kp_k = det.detect(img_k)
                tot, pos, sc, both = evaluate_pair_transformation_correctness(
                    kp1_cache[name], kp_k, H_1_k, img_k.shape[:2]
                )
                totals[name]["total"] += tot
                totals[name]["pos"] += pos
                totals[name]["scale"] += sc
                totals[name]["both"] += both
            print(f"[{pair_idx}/{len(pairs)}] Finished {pair_name}", flush=True)

    # Print results in the exact format of Table 1 from the Progression Check
    dog_t = totals["DoGDetector"]
    hl_t = totals["HarrisLaplaceDetector"]

    def _fmt(count: int, total: int) -> str:
        pct = (100.0 * count / total) if total > 0 else 0.0
        return f"{pct:5.1f}% ({count}/{total})"

    print("\nTable 1 (Updated). Transformation-based correctness of feature detectors:")
    print("-" * 86)
    print(
        f"{'Match Type':<24} | {'DoG (Image vs Transformed)':<27} | {'HarrisLaplace (Image vs Transformed)':<27}"
    )
    print("-" * 86)
    print(
        f"{'Position Match':<24} | {_fmt(dog_t['pos'], dog_t['total']):<27} | {_fmt(hl_t['pos'], hl_t['total']):<27}"
    )
    print(
        f"{'Scale Match':<24} | {_fmt(dog_t['scale'], dog_t['total']):<27} | {_fmt(hl_t['scale'], hl_t['total']):<27}"
    )
    print(
        f"{'Position + Scale Match':<24} | {_fmt(dog_t['both'], dog_t['total']):<27} | {_fmt(hl_t['both'], hl_t['total']):<27}"
    )
    print("-" * 86)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test DoG and Harris-Laplace detector correctness across known image transformations."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to HPatches dataset root (contains sequence folders with 1.ppm..6.ppm and H_1_2..H_1_6).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel CPU processes (default: CPU count - 1).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional limit on number of HPatches sequences to test.",
    )
    args = parser.parse_args()
    run_correctness_benchmark(
        args.dataset, workers=args.workers, max_sequences=args.max_sequences
    )