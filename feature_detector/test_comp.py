"""Compatibility tests: Harris-Laplace / DoG detectors -> SIFT descriptor -> matcher.

Run from the project folder (next to harris.py, dog.py, sift.py, ...):

    python test_compatibility.py                 # synthetic RGB test image
    python test_compatibility.py --image img.jpg # your own image (loaded as BGR)
    python test_compatibility.py --defaults      # use the detectors' default settings (slow)

What it checks, for each detector:
  1. The RGB test image converts to a float32 [0, 1] greyscale image.
  2. The detector returns shared Keypoint objects with valid fields
     (finite, inside the image, sigma > 0, angle not yet assigned).
  3. sift_describe accepts the detector's keypoints with a BGR image
     and with a greyscale image.
  4. The SIFT output is consistent: one keypoint per descriptor row,
     (N, 128) float32, finite, non-negative, unit length, clipped values,
     angles in radians [0, 2*pi), and every output keypoint comes from an
     input keypoint (same x, y, sigma, response).
  5. to_opencv_keypoints converts the output (radians -> degrees).
  6. End to end: the image rotated by exactly 90 degrees is matched against
     the original with BFMatcher. Correct matches must land where the
     rotation says they should, and their angles must differ by 90 degrees.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback

import cv2
import numpy as np

from detector_helpers import Keypoint, to_grayscale, to_opencv_keypoints
from dog import DoGDetector
from harris import HarrisLaplaceDetector
from matcher import BFMatcher
from sift import sift_describe

# --------------------------------------------------------------------------
# Tiny test harness
# --------------------------------------------------------------------------

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(condition)


def run_step(name: str, fn):
    """Run fn(); record a FAIL with the error instead of crashing the script."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - we want every failure reported
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        where = f"{tb.filename.split('/')[-1]}:{tb.lineno}"
        check(name, False, f"{type(exc).__name__} at {where}: {exc}")
        return None


# --------------------------------------------------------------------------
# Test image
# --------------------------------------------------------------------------

def make_rgb_test_image(size: int = 256, seed: int = 0) -> np.ndarray:
    """A synthetic RGB uint8 scene with corners and blobs at several scales."""
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]

    # Smooth colour background gradient.
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 60 + 60 * xx / w          # R
    img[..., 1] = 70 + 50 * yy / h          # G
    img[..., 2] = 110 - 40 * xx / w         # B
    img = img.astype(np.uint8)

    s = size / 256.0  # scale shapes with the image size

    def S(v):
        return int(round(v * s))

    # Filled rectangles (strong corners), colours given in RGB.
    cv2.rectangle(img, (S(20), S(20)), (S(90), S(70)), (230, 40, 40), -1)
    cv2.rectangle(img, (S(150), S(30)), (S(230), S(110)), (40, 200, 60), -1)
    cv2.rectangle(img, (S(170), S(50)), (S(205), S(85)), (250, 240, 60), -1)
    cv2.rectangle(img, (S(30), S(160)), (S(70), S(230)), (30, 30, 30), -1)

    # Circles (blobs for the DoG detector) at a few sizes.
    for (cx, cy, r, col) in [(120, 130, 22, (240, 240, 240)),
                             (200, 180, 12, (20, 20, 160)),
                             (95, 205, 7, (255, 140, 0)),
                             (225, 225, 5, (250, 250, 250)),
                             (60, 120, 9, (180, 0, 180))]:
        cv2.circle(img, (S(cx), S(cy)), max(2, S(r)), col, -1, lineType=cv2.LINE_AA)

    # A triangle (corners that are not axis aligned).
    tri = np.array([[S(130), S(170)], [S(175), S(240)], [S(105), S(235)]], np.int32)
    cv2.fillPoly(img, [tri], (0, 160, 200), lineType=cv2.LINE_AA)

    # Small checkerboard patch.
    cb = S(8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                x0, y0 = S(110) + j * cb, S(25) + i * cb
                cv2.rectangle(img, (x0, y0), (x0 + cb - 1, y0 + cb - 1), (255, 255, 255), -1)

    # Mild noise so the image is not perfectly piecewise constant.
    noise = rng.normal(0, 2.0, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img  # RGB


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------

def test_grayscale(rgb: np.ndarray, bgr: np.ndarray) -> np.ndarray | None:
    print("\n== Greyscale conversion ==")
    gray = run_step("to_grayscale(BGR image)", lambda: to_grayscale(bgr))
    if gray is None:
        return None
    check("greyscale is 2-D with the image's height/width",
          gray.ndim == 2 and gray.shape == bgr.shape[:2], f"shape {gray.shape}")
    check("greyscale is float32", gray.dtype == np.float32, str(gray.dtype))
    check("greyscale values lie in [0, 1]",
          float(gray.min()) >= 0.0 and float(gray.max()) <= 1.0,
          f"min {gray.min():.3f}, max {gray.max():.3f}")
    # Compare with OpenCV's own RGB -> grey conversion (same luma weights).
    ref = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    err = float(np.abs(gray - ref).max())
    check("BGR channel order is handled correctly (matches cv2 RGB2GRAY)",
          err < 2.0 / 255.0, f"max abs diff {err:.4f}")
    return gray


def test_detector_output(name: str, kps, shape) -> bool:
    h, w = shape
    ok = check(f"{name}: detect() returns a list", isinstance(kps, list), type(kps).__name__)
    ok &= check(f"{name}: found at least one keypoint", len(kps) > 0, f"{len(kps)} keypoints")
    if not kps:
        return False
    ok &= check(f"{name}: every item is a detector_helpers.Keypoint",
                all(isinstance(k, Keypoint) for k in kps))
    finite = all(all(math.isfinite(float(v)) for v in (k.x, k.y, k.sigma, k.response)) for k in kps)
    ok &= check(f"{name}: x, y, sigma, response are finite", finite)
    inside = all(0 <= k.x <= w - 1 and 0 <= k.y <= h - 1 for k in kps)
    ok &= check(f"{name}: all keypoints lie inside the image (x=col, y=row)", inside)
    ok &= check(f"{name}: sigma > 0", all(k.sigma > 0 for k in kps),
                f"sigma range {min(k.sigma for k in kps):.2f}-{max(k.sigma for k in kps):.2f}")
    ok &= check(f"{name}: angle not assigned yet (None)", all(k.angle is None for k in kps))
    return ok


def test_sift_output(name: str, in_kps, out_kps, descs) -> bool:
    ok = check(f"{name}: one output keypoint per descriptor row",
               len(out_kps) == len(descs), f"{len(out_kps)} keypoints, {len(descs)} rows")
    ok &= check(f"{name}: descriptors are (N, 128) float32",
                descs.ndim == 2 and descs.shape[1] == 128 and descs.dtype == np.float32,
                f"{descs.shape} {descs.dtype}")
    if len(descs) == 0:
        check(f"{name}: produced at least one descriptor", False, "empty output")
        return False
    ok &= check(f"{name}: produced at least one descriptor", True, f"{len(descs)} descriptors")
    ok &= check(f"{name}: descriptors finite and non-negative",
                bool(np.all(np.isfinite(descs)) and descs.min() >= 0))
    norms = np.linalg.norm(descs, axis=1)
    ok &= check(f"{name}: descriptors are unit length",
                bool(np.allclose(norms, 1.0, atol=1e-3)),
                f"norm range {norms.min():.4f}-{norms.max():.4f}")
    # After clip at 0.2 and renormalising, entries can grow a little above 0.2
    # but should stay well below 1.
    ok &= check(f"{name}: clipping applied (max entry < 0.5)",
                float(descs.max()) < 0.5, f"max entry {descs.max():.3f}")
    angles = [k.angle for k in out_kps]
    ok &= check(f"{name}: every output keypoint has an angle in radians [0, 2*pi)",
                all(a is not None and 0 <= a < 2 * math.pi for a in angles))
    source = {(k.x, k.y, k.sigma, k.response) for k in in_kps}
    ok &= check(f"{name}: output keypoints come from the detector's keypoints",
                all((k.x, k.y, k.sigma, k.response) in source for k in out_kps))
    ok &= check(f"{name}: orientation kept at least one angle for most keypoints",
                len({(k.x, k.y, k.sigma) for k in out_kps}) >= 0.5 * len(in_kps),
                f"{len({(k.x, k.y, k.sigma) for k in out_kps})}/{len(in_kps)} keypoints oriented")
    return ok


def test_opencv_conversion(name: str, out_kps) -> None:
    cv_kps = run_step(f"{name}: to_opencv_keypoints(SIFT output)",
                      lambda: to_opencv_keypoints(out_kps))
    if cv_kps is None:
        return
    check(f"{name}: OpenCV keypoint count matches", len(cv_kps) == len(out_kps))
    deg_ok = all(abs(c.angle - math.degrees(k.angle) % 360.0) < 1e-3
                 for c, k in zip(cv_kps, out_kps))
    check(f"{name}: OpenCV angles are the radians converted to degrees", deg_ok)
    check(f"{name}: OpenCV angles in [0, 360)", all(0 <= c.angle < 360 for c in cv_kps))


def rot90_point(x: float, y: float, w: int) -> tuple[float, float]:
    """Where pixel (x=col, y=row) lands after np.rot90(img, k=1)."""
    return y, (w - 1) - x


def test_rotation_matching(name: str, detector, bgr: np.ndarray,
                           kps0, desc0) -> None:
    h, w = bgr.shape[:2]
    rotated = np.ascontiguousarray(np.rot90(bgr, k=1))

    def describe_rotated():
        k = detector.detect(rotated)
        return sift_describe(rotated, k)

    res = run_step(f"{name}: detect + describe on the 90-degree rotated image", describe_rotated)
    if res is None:
        return
    kps1, desc1 = res
    matches = run_step(f"{name}: BFMatcher.match on SIFT descriptors",
                       lambda: BFMatcher().match(desc0, desc1))
    if matches is None:
        return
    if not check(f"{name}: matcher returned matches", len(matches) > 0, f"{len(matches)} matches"):
        return

    inliers, angle_errs = 0, []
    for m in matches:
        a, b = kps0[m.query_idx], kps1[m.train_idx]
        ex, ey = rot90_point(a.x, a.y, w)
        # DoG coordinates are quantised to the octave grid, so allow a
        # tolerance that grows with scale.
        tol = max(3.0, 0.5 * a.sigma)
        if math.hypot(b.x - ex, b.y - ey) <= tol:
            inliers += 1
            # Rotating the image by +90 deg (rot90, y down) turns gradient
            # angles by -90 deg, so b.angle should equal a.angle - pi/2.
            d = (b.angle - (a.angle - math.pi / 2)) % (2 * math.pi)
            angle_errs.append(math.degrees(min(d, 2 * math.pi - d)))

    ratio = inliers / len(matches)
    check(f"{name}: matches land where the rotation predicts",
          ratio >= 0.5, f"{inliers}/{len(matches)} geometrically correct ({ratio:.0%})")
    if angle_errs:
        med = float(np.median(angle_errs))
        check(f"{name}: matched angles differ by 90 degrees (radians used consistently)",
              med < 15.0, f"median angle error {med:.1f} deg")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="optional image file (read with cv2.imread as BGR)")
    parser.add_argument("--size", type=int, default=256, help="synthetic image size (default 256)")
    parser.add_argument("--defaults", action="store_true",
                        help="use the detectors' default settings instead of faster test settings")
    args = parser.parse_args()

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"Could not read {args.image}")
            return 2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"Using {args.image} ({bgr.shape[1]}x{bgr.shape[0]})")
    else:
        rgb = make_rgb_test_image(args.size)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # pipeline expects OpenCV BGR
        cv2.imwrite("test_image.png", bgr)
        print(f"Using synthetic RGB test image {args.size}x{args.size} (saved as test_image.png)")

    gray = test_grayscale(rgb, bgr)

    if args.defaults:
        detectors = [("Harris-Laplace", HarrisLaplaceDetector()), ("DoG", DoGDetector())]
    else:
        # Smaller top scales keep the pure-Python SIFT loops quick.
        detectors = [("Harris-Laplace", HarrisLaplaceDetector(num_scales=8, max_points=150)),
                     ("DoG", DoGDetector(num_octaves=3))]

    for name, detector in detectors:
        print(f"\n== {name} -> SIFT ==")
        t0 = time.perf_counter()
        kps = run_step(f"{name}: detect(BGR image)", lambda: detector.detect(bgr))
        if kps is None or not test_detector_output(name, kps, bgr.shape[:2]):
            print(f"  -> skipping SIFT tests for {name}")
            continue

        res_bgr = run_step(f"{name}: sift_describe(BGR image, keypoints)",
                           lambda: sift_describe(bgr, kps))
        res_gray = None
        if gray is not None:
            res_gray = run_step(f"{name}: sift_describe(greyscale image, keypoints)",
                                lambda: sift_describe(gray, kps))

        res = res_bgr if res_bgr is not None else res_gray
        if res is None:
            print(f"  -> sift_describe failed for every input; skipping the rest of {name}")
            continue
        if res_bgr is not None and res_gray is not None:
            check(f"{name}: BGR and greyscale input give the same descriptors",
                  len(res_bgr[1]) == len(res_gray[1])
                  and bool(np.allclose(res_bgr[1], res_gray[1], atol=1e-5)))

        out_kps, descs = res
        test_sift_output(name, kps, out_kps, descs)
        test_opencv_conversion(name, out_kps)
        test_rotation_matching(name, detector, bgr if res_bgr is not None else gray,
                               out_kps, descs)
        print(f"  ({time.perf_counter() - t0:.1f}s)")

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n== Summary: {len(RESULTS) - len(failed)} passed, {len(failed)} failed ==")
    for name, _, detail in failed:
        print(f"  FAIL {name}" + (f": {detail}" if detail else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())