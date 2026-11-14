"""OpenCV SIFT baseline for the HPatches patch benchmark.

Same output layout as hpatches_extract.py:

    <out>/<sequence>/<type>.csv     (one 128-D descriptor per row)

Usage:
    python hpatches_extract_opencv.py PATH/TO/hpatches-release PATH/TO/descriptors/opencv-sift
    python hpatches_extract_opencv.py ... --rootsift

This follows the toolkit's own python/extract_opencv_sift.py, with its bugs
fixed for current OpenCV/Python (cv2.SIFT_create instead of
cv2.xfeatures2d.SIFT_create, integer patch counts, explicit angle):

  * one keypoint at the patch centre (32.5, 32.5),
  * size = 2 * 32.5 / 5.303 (the toolkit's choice: OpenCV's descriptor window
    radius is then exactly the 65 px patch width; equivalent to our --sigma 6.13),
  * angle = 0, because HPatches patches are already oriented.

Because the keypoint is in octave 0, OpenCV computes the descriptor on the
patch blurred to sigma 1.6 (its base level), not to the keypoint's scale.
To mirror that with our descriptor use:  --sigma 6.13 --blur 1.6

OpenCV stores descriptors scaled to 0..255 (about 512 x unit length); the
benchmark only ranks distances, so the scale makes no difference.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

PATCH = 65
TYPES = ["ref"] + [f"{t}{i}" for t in "eht" for i in range(1, 6)]
CENTRE = PATCH / 2.0
SIZE = 2 * CENTRE / 5.303


def describe_patches(patches: np.ndarray, rootsift: bool, size: float = SIZE) -> np.ndarray:
    sift = cv2.SIFT_create()
    out = np.zeros((len(patches), 128), np.float32)
    for i, p in enumerate(patches):
        kp = [cv2.KeyPoint(x=CENTRE, y=CENTRE, size=size, angle=0.0)]
        _, d = sift.compute(np.ascontiguousarray(p), kp)
        if d is None or len(d) != 1:
            raise RuntimeError(f"OpenCV returned {0 if d is None else len(d)} descriptors for patch {i}")
        out[i] = d[0]
    if rootsift:
        out = np.sqrt(out / np.maximum(out.sum(axis=1, keepdims=True), 1e-7))
    return out


def load_patches(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None or im.shape[1] != PATCH or im.shape[0] % PATCH:
        raise ValueError(f"{path} is not a stack of {PATCH}x{PATCH} patches")
    return im.reshape(-1, PATCH, PATCH)


def process_sequence(seq_dir: str, out_root: str, rootsift: bool) -> str:
    cv2.setNumThreads(1)  # one process per sequence already uses every core
    name = os.path.basename(os.path.normpath(seq_dir))
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    for t in TYPES:
        out_csv = os.path.join(out_dir, t + ".csv")
        if os.path.exists(out_csv):
            continue  # resume support
        desc = describe_patches(load_patches(os.path.join(seq_dir, t + ".png")), rootsift)
        tmp = out_csv + ".tmp"
        np.savetxt(tmp, desc, delimiter=",", fmt="%.6f")
        os.replace(tmp, out_csv)
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hpatches_root", help="folder containing the 116 i_*/v_* sequence folders")
    ap.add_argument("out_dir", help="output folder, e.g. hpatches-benchmark/data/descriptors/opencv-sift")
    ap.add_argument("--rootsift", action="store_true", help="apply RootSIFT (L1-normalise, then square root)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    if not hasattr(cv2, "SIFT_create"):
        print("This OpenCV has no cv2.SIFT_create; install opencv-python >= 4.4")
        return 2
    seq_dirs = sorted(os.path.join(args.hpatches_root, d) for d in os.listdir(args.hpatches_root)
                      if os.path.isdir(os.path.join(args.hpatches_root, d))
                      and d[:2] in ("i_", "v_"))
    if not seq_dirs:
        print(f"No i_*/v_* sequence folders in {args.hpatches_root}")
        return 2
    print(f"OpenCV {cv2.__version__}: {len(seq_dirs)} sequences, rootsift={args.rootsift}, "
          f"writing to {args.out_dir}")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_sequence, s, args.out_dir, args.rootsift) for s in seq_dirs]
        for k, f in enumerate(futures, 1):
            print(f"[{k:3d}/{len(seq_dirs)}] {f.result()}  ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())