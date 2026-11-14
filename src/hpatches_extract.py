"""Compute our SIFT descriptor for every patch of the HPatches patch dataset.

Writes the files the official HPatches benchmark expects:

    <out>/<sequence>/<type>.csv     (one 128-D descriptor per row, comma separated)

where <type> is ref, e1..e5, h1..h5, t1..t5, and row i of every file is the
descriptor of patch i of <sequence>/<type>.png.

Usage:
    python hpatches_extract.py PATH/TO/hpatches-release PATH/TO/descriptors/our-sift
    python hpatches_extract.py ... --sigma 6.13 --workers 8
    python hpatches_extract.py ... --sigma 6.13 --blur 1.6 --rootsift

The start-up check always compares the default settings (blur=auto, no
RootSIFT) against sift.py, since those are what sift.py itself computes.

How a patch is described
------------------------
HPatches patches are 65x65, already centred on the feature, already scaled to a
fixed size and already rotated to the feature's dominant orientation. So each
patch is described as ONE keypoint at the patch centre with:

  * angle = 0 (no orientation assignment; the patch is already oriented),
  * a fixed sigma for every patch (default 65/12, so the 4x4 cell grid,
    4 cells * 3*sigma wide, exactly spans the patch).

Exactly one descriptor is produced per patch, which the benchmark requires
(sift_describe can return 0 or several per keypoint, so it is not used here).

The maths is identical to sift.generate_descriptor + sift.GradientCache, but
vectorised: every patch has the same keypoint geometry, so the sample offsets,
Gaussian weights and spatial bin weights are computed once and applied to all
patches with a single matrix product. `--check` compares the two
implementations on real patches before running.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from detector_helpers import Keypoint, to_grayscale
from sift import (DESC_BINS, DESC_MAG_CLIP, DESC_SCALE_FACTOR, DESC_WIDTH,
                  GradientCache, generate_descriptor)

PATCH = 65
TYPES = ["ref"] + [f"{t}{i}" for t in "eht" for i in range(1, 6)]


# --------------------------------------------------------------------------
# Vectorised descriptor (same maths as sift.generate_descriptor)
# --------------------------------------------------------------------------

class PatchDescriptor:
    """SIFT descriptor for a stack of same-size patches, keypoint at the centre."""

    def __init__(self, sigma: float, angle: float = 0.0, size: int = PATCH,
                 blur: float | None = None, rootsift: bool = False):
        """blur: total blur (sigma) of the image the gradients are taken from.
        None = what sift.py does (the keypoint sigma, quantised by GradientCache).
        rootsift: L1-normalise then square-root each descriptor (Arandjelovic 2012).
        """
        self.sigma, self.angle, self.size, self.rootsift = sigma, angle, size, rootsift

        # Blur level exactly as GradientCache.get chooses it (or as overridden).
        gc = GradientCache(np.zeros((8, 8), np.float32))
        target = gc._quantise(sigma) if blur is None else blur
        self.blur = float(np.sqrt(max(target ** 2 - gc.assumed_blur ** 2, 0.01)))

        # Fixed sample geometry (same loop bounds and tests as generate_descriptor).
        H = W = size
        cx = cy = int(round(size // 2))
        cos_t, sin_t = np.cos(angle), np.sin(angle)
        hist_width = DESC_SCALE_FACTOR * sigma
        radius = int(round(hist_width * np.sqrt(2) * (DESC_WIDTH + 1) * 0.5))
        weight_sigma = 0.5 * DESC_WIDTH

        dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        dx, dy = dx.ravel().astype(np.float64), dy.ravel().astype(np.float64)
        rx = (cos_t * dx + sin_t * dy) / hist_width
        ry = (-sin_t * dx + cos_t * dy) / hist_width
        cbin = rx + DESC_WIDTH / 2 - 0.5
        rbin = ry + DESC_WIDTH / 2 - 0.5
        px = cx + dx.astype(int)
        py = cy + dy.astype(int)
        keep = ((-1 < rbin) & (rbin < DESC_WIDTH) & (-1 < cbin) & (cbin < DESC_WIDTH) &
                (0 < px) & (px < W - 1) & (0 < py) & (py < H - 1))
        rx, ry, rbin, cbin, px, py = (a[keep] for a in (rx, ry, rbin, cbin, px, py))

        w = np.exp(-(rx ** 2 + ry ** 2) / (2 * weight_sigma ** 2))
        r0, c0 = np.floor(rbin).astype(int), np.floor(cbin).astype(int)
        dr, dc = rbin - r0, cbin - c0

        n = len(px)
        side = DESC_WIDTH + 2
        S = np.zeros((n, side * side))  # sample -> (row, col) cell weight
        rows = np.arange(n)
        for ir, wr in ((0, 1 - dr), (1, dr)):
            for ic, wc in ((0, 1 - dc), (1, dc)):
                cell = (r0 + 1 + ir) * side + (c0 + 1 + ic)
                np.add.at(S, (rows, cell), w * wr * wc)

        self.pix = py * W + px  # flat pixel index of each sample
        self.S = S

    def __call__(self, patches: np.ndarray) -> np.ndarray:
        """patches: (P, size, size) uint8 -> (P, 128) float32 descriptors."""
        P = len(patches)
        img = patches.astype(np.float32) / np.float32(255.0)  # == to_grayscale for uint8

        # Match gaussian_blur's reflected edges and 3-sigma kernel radius.
        radius = int(np.ceil(3 * self.blur))
        L = gaussian_filter(img, sigma=(0, self.blur, self.blur),
                            mode="mirror", radius=(0, radius, radius))
        gx = np.zeros_like(L)
        gy = np.zeros_like(L)
        gx[:, :, 1:-1] = L[:, :, 2:] - L[:, :, :-2]
        gy[:, 1:-1, :] = L[:, 2:, :] - L[:, :-2, :]
        mag = np.hypot(gx, gy).reshape(P, -1)[:, self.pix].astype(np.float64)
        ori = np.mod(np.arctan2(gy, gx), 2 * np.pi).reshape(P, -1)[:, self.pix]

        theta = (ori - self.angle) % (2 * np.pi)
        obin = theta * DESC_BINS / (2 * np.pi)
        o0 = np.floor(obin).astype(int)
        do = obin - o0

        n = mag.shape[1]
        # Per-sample orientation contributions. The two bins of a sample are
        # always different, so plain assignment is the same as accumulating.
        O = np.zeros((P, n, DESC_BINS))
        np.put_along_axis(O, (o0 % DESC_BINS)[..., None], (mag * (1 - do))[..., None], axis=2)
        np.put_along_axis(O, ((o0 + 1) % DESC_BINS)[..., None], (mag * do)[..., None], axis=2)

        # Spatial (trilinear) weights: (P, 8, n) @ (n, 36) -> (P, 8, 36).
        side = DESC_WIDTH + 2
        hist = (O.transpose(0, 2, 1) @ self.S).transpose(0, 2, 1)
        hist = hist.reshape(P, side, side, DESC_BINS)
        desc = hist[:, 1:-1, 1:-1, :].reshape(P, -1)

        desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-7)
        desc = np.minimum(desc, DESC_MAG_CLIP)
        desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-7)
        if self.rootsift:
            desc = np.sqrt(desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-7))
        return desc.astype(np.float32)


def reference_descriptor(patch: np.ndarray, sigma: float, angle: float = 0.0) -> np.ndarray:
    """The original, unvectorised path through sift.py (slow; used for --check)."""
    gray = to_grayscale(patch)
    c = float(patch.shape[0] // 2)
    return generate_descriptor(Keypoint(x=c, y=c, sigma=sigma, angle=angle),
                               GradientCache(gray))


# --------------------------------------------------------------------------
# Dataset I/O
# --------------------------------------------------------------------------

def load_patches(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None or im.shape[1] != PATCH or im.shape[0] % PATCH:
        raise ValueError(f"{path} is not a stack of {PATCH}x{PATCH} patches")
    return im.reshape(-1, PATCH, PATCH)


def process_sequence(seq_dir: str, out_root: str, sigma: float, batch: int,
                     blur: float | None, rootsift: bool) -> str:
    name = os.path.basename(os.path.normpath(seq_dir))
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    describe = PatchDescriptor(sigma, blur=blur, rootsift=rootsift)
    for t in TYPES:
        out_csv = os.path.join(out_dir, t + ".csv")
        if os.path.exists(out_csv):
            continue  # resume support
        patches = load_patches(os.path.join(seq_dir, t + ".png"))
        desc = np.vstack([describe(patches[i:i + batch])
                          for i in range(0, len(patches), batch)])
        tmp = out_csv + ".tmp"
        np.savetxt(tmp, desc, delimiter=",", fmt="%.6f")
        os.replace(tmp, out_csv)  # never leave a half-written csv behind
    return name


def check_against_reference(seq_dirs: list[str], sigma: float, n: int = 40) -> bool:
    rng = np.random.default_rng(0)
    patches = load_patches(os.path.join(seq_dirs[0], "ref.png"))
    pick = rng.choice(len(patches), size=min(n, len(patches)), replace=False)
    fast = PatchDescriptor(sigma)(patches[pick])
    slow = np.stack([reference_descriptor(patches[i], sigma) for i in pick])
    err = float(np.abs(fast - slow).max())
    ok = err < 1e-4
    print(f"check vs sift.generate_descriptor on {len(pick)} patches: "
          f"max abs diff {err:.2e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hpatches_root", help="folder containing the 116 i_*/v_* sequence folders")
    ap.add_argument("out_dir", help="output folder, e.g. hpatches-benchmark/data/descriptors/our-sift")
    ap.add_argument("--sigma", type=float, default=PATCH / (4 * DESC_SCALE_FACTOR),
                    help="keypoint sigma in patch pixels (default 65/12 = 5.42)")
    ap.add_argument("--blur", type=float, default=None,
                    help="total blur sigma of the image gradients are taken from "
                         "(default: the keypoint sigma, as sift.py does; "
                         "1.6 = what OpenCV uses for these patches)")
    ap.add_argument("--rootsift", action="store_true",
                    help="apply RootSIFT (L1-normalise, then square root)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--batch", type=int, default=256, help="patches per vectorised batch")
    ap.add_argument("--no-check", action="store_true", help="skip the comparison with sift.py")
    args = ap.parse_args()

    seq_dirs = sorted(os.path.join(args.hpatches_root, d) for d in os.listdir(args.hpatches_root)
                      if os.path.isdir(os.path.join(args.hpatches_root, d))
                      and d[:2] in ("i_", "v_"))
    if not seq_dirs:
        print(f"No i_*/v_* sequence folders in {args.hpatches_root}")
        return 2
    print(f"{len(seq_dirs)} sequences, sigma={args.sigma:.3f}, "
          f"blur={'auto' if args.blur is None else args.blur}, rootsift={args.rootsift}, "
          f"writing to {args.out_dir}")
    if len(seq_dirs) != 116:
        print("warning: the full HPatches release has 116 sequences; "
              "the benchmark needs all of them")

    if not args.no_check and not check_against_reference(seq_dirs, args.sigma):
        return 1

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_sequence, s, args.out_dir, args.sigma, args.batch,
                               args.blur, args.rootsift)
                   for s in seq_dirs]
        for k, f in enumerate(futures, 1):
            name = f.result()
            print(f"[{k:3d}/{len(seq_dirs)}] {name}  ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
