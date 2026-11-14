import numpy as np
from typing import List, Optional, Tuple
from dataclasses import replace

from detector_helpers import Keypoint, gaussian_blur, to_grayscale

"""
SIFT Feature Descriptor

Algorithm:
1. Normalise both Harris-Laplace and Difference of Gaussian keypoints into a single type
2. Build a Gaussian scale space so each keypoint is described on an image smoothed
   to its own scale
3. Precompute gradient magnitude and orientation per scale level
4. Assign an orientation to each keypoint
5. Construct the 128-D descriptor using 4x4 sub-blocks
6. Cap + normalise the descriptor
"""


# Constants - (Taken from Lowe 2004)
ORI_BINS = 36 # histogram has 36 bins of 10 degrees each
ORI_SIGMA_FACTOR = 1.5 # weighted sigma = 1.5 * keypoint sigma
ORI_RADIUS_FACTOR = 3.0 # window radius = 3 * weighted sigma
ORI_PEAK_RATIO = 0.8 # any histogram peak at 80% or more of the max becomes its own keypoint

DESC_WIDTH = 4 # 4x4 spatial cells
DESC_BINS = 8 # 8 orientation bins per cell
DESC_SCALE_FACTOR = 3.0 # each cell is 3 * sigma pixels wide
DESC_MAG_CLIP = 0.2 # Capped value after first normalisation of descriptor

def harris_laplace_keypoint(points: np.ndarray, sigma_is_derivation=False, s=0.7) -> List[Keypoint]:
    """
    points: (N, 3) keypoints generated from harris-laplace feature detector
    sigma_is_derivation=False is correct for our implementation, the integration sigma is provided.

    returns: a standardised keypoint (Keypoint instance) that the sift algorithm can use to compute a descriptor.
    """
    keypoints = []
    for x, y, sig in points:
        if sigma_is_derivation:
            sig = sig / s
        keypoints.append(Keypoint(x=float(x), y=float(y), sigma=float(sig)))

    return keypoints


def difference_of_gaussians_keypoint(points: np.ndarray, sigma_0: float = 1.6, n_layers: int = 3, 
                                     octave_unsampled: bool = True) -> List[Keypoint]:
    """
    points: (N, 3) keypoints generated from difference of gaussians feature detector
    num
    returns: a standardised keypoint (Keypoint instance) that the sift algorithm can use to compute a descriptor.
    """
    keypoints = []
    for p in points:
        x, y, octave, layer = p[:4]
        frac = p[4] if len(p) > 4 else 0.0 # Sub-level offset from refinement
        resp = p[5] if len(p) > 5 else 0.0

        # scale within the octave, then up to the original image
        sig_oct = sigma_0 * 2.0 ** ((layer + frac) / n_layers)
        factor = 2.0 ** octave
        if octave_unsampled:
            factor *= 0.5

        keypoints.append(Keypoint(x=x * factor, y=y * factor, sigma=sig_oct * factor, response=float(resp)))

    return keypoints

class GradientCache:
    """
    Blurs the image to the scale each keypoint needs and caches gradients.
    Simple version: quantise sigma to a log grid and blur the full image once
    per level.
    """
    def __init__(self, image: np.ndarray, assumed_blur: float = 0.5,
                 levels_per_octave: int = 6):
        self.image = image.astype(np.float32)
        if self.image.max() > 1.0:
            self.image /= 255.0
        self.assumed_blur = assumed_blur
        self.levels_per_octave = levels_per_octave
        self._cache = {}
 
    def _quantise(self, sigma: float) -> float:
        k = np.round(np.log2(sigma) * self.levels_per_octave)
        return 2.0 ** (k / self.levels_per_octave)
 
    def get(self, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (magnitude, orientation in [0, 2pi)) at blur level sigma."""
        sq = self._quantise(sigma)
        if sq not in self._cache:
            extra = np.sqrt(max(sq ** 2 - self.assumed_blur ** 2, 0.01))
            L = gaussian_blur(self.image, extra)
            dx = np.zeros_like(L)
            dy = np.zeros_like(L)
            dx[:, 1:-1] = L[:, 2:] - L[:, :-2]
            dy[1:-1, :] = L[2:, :] - L[:-2, :]
            mag = np.hypot(dx, dy)
            ori = np.mod(np.arctan2(dy, dx), 2 * np.pi)
            self._cache[sq] = (mag, ori)
        return self._cache[sq]
    
def assign_orientations(keypoint: Keypoint, gradients: GradientCache) -> List[Keypoint]:
    mag, ori = gradients.get(keypoint.sigma)
    H, W = mag.shape
    weighted_sig = ORI_SIGMA_FACTOR * keypoint.sigma
    radius = int(round(ORI_RADIUS_FACTOR * weighted_sig))
    cx, cy = int(round(keypoint.x)), int(round(keypoint.y))

    # Window around the keypoint, clipped to the image
    x0, x1 = max(cx - radius, 0), min(cx + radius, W - 1)
    y0, y1 = max(cy - radius, 0), min(cy + radius, H - 1)
    hist = np.zeros(ORI_BINS)

    if x0 <= x1 and y0 <= y1:
        dy, dx = np.mgrid[y0 - cy:y1 - cy + 1, x0 - cx:x1 - cx + 1]
        w = np.exp(-(dx * dx + dy * dy) / (2 * weighted_sig ** 2))
        o = ori[y0:y1 + 1, x0:x1 + 1]
        b = (ORI_BINS * o / (2 * np.pi)).astype(int) % ORI_BINS
        hist = np.bincount(b.ravel(), weights=(w * mag[y0:y1 + 1, x0:x1 + 1]).ravel(),
                           minlength=ORI_BINS).astype(np.float64)

    # smoothing
    for _ in range(6):
        hist = (np.roll(hist, 1) + hist + np.roll(hist, -1)) / 3

    out = []

    # drop the keypoint if the histogram is empty (completely flat with no gradient)
    if hist.max() <= 0:
        return out

    peak_thresh = ORI_PEAK_RATIO * hist.max()
    for i in range(ORI_BINS):
        l, r = hist[(i - 1) % ORI_BINS], hist[(i + 1) % ORI_BINS]
        if hist[i] > l and hist[i] > r and hist[i] >= peak_thresh:
            # parabolic interpolation of peak position
            offset = 0.5 * (l - r) / (l - 2 * hist[i] + r)
            b = (i + offset) % ORI_BINS
            angle = 2 * np.pi * b / ORI_BINS
            out.append(replace(keypoint, angle=angle))

    return out

def generate_descriptor(keypoint: Keypoint, gradients: GradientCache) -> Optional[np.ndarray]:
    mag, ori = gradients.get(keypoint.sigma)
    H, W = mag.shape
    cos_t, sin_t = np.cos(keypoint.angle), np.sin(keypoint.angle)

    hist_width = DESC_SCALE_FACTOR * keypoint.sigma # number of pixels in the subblock
    radius = int(round(hist_width * np.sqrt(2) * (DESC_WIDTH + 1) * 0.5))
    weight_sigma = 0.5 * DESC_WIDTH
    side = DESC_WIDTH + 2
    hist = np.zeros(side * side * DESC_BINS)

    # Window around the keypoint, clipped to pixels strictly inside the image
    cx, cy = int(round(keypoint.x)), int(round(keypoint.y))
    x0, x1 = max(cx - radius, 1), min(cx + radius, W - 2)
    y0, y1 = max(cy - radius, 1), min(cy + radius, H - 2)

    if x0 <= x1 and y0 <= y1:
        # rotate every offset in the window into the keypoint frame
        dy, dx = np.mgrid[y0 - cy:y1 - cy + 1, x0 - cx:x1 - cx + 1]
        rx = ( cos_t * dx + sin_t * dy) / hist_width
        ry = (-sin_t * dx + cos_t * dy) / hist_width

        cbin = rx + DESC_WIDTH / 2 - 0.5
        rbin = ry + DESC_WIDTH / 2 - 0.5
        keep = (-1 < rbin) & (rbin < DESC_WIDTH) & (-1 < cbin) & (cbin < DESC_WIDTH)

        if keep.any():
            rx, ry, rbin, cbin = rx[keep], ry[keep], rbin[keep], cbin[keep]
            o = ori[y0:y1 + 1, x0:x1 + 1][keep]
            m = mag[y0:y1 + 1, x0:x1 + 1][keep]

            # orientation relative to keypoint
            theta = (o - keypoint.angle) % (2 * np.pi)
            obin = theta * DESC_BINS / (2 * np.pi)
            m = np.exp(-(rx ** 2 + ry ** 2) / (2 * weight_sigma ** 2)) * m

            # trilinear interpolation into the (row, col, orientation) histogram
            r0, c0, o0 = (np.floor(rbin).astype(int), np.floor(cbin).astype(int),
                          np.floor(obin).astype(int))
            dr, dc, do = rbin - r0, cbin - c0, obin - o0
            for ir, wr in ((0, 1 - dr), (1, dr)):
                for ic, wc in ((0, 1 - dc), (1, dc)):
                    for io, wo in ((0, 1 - do), (1, do)):
                        idx = ((r0 + 1 + ir) * side + (c0 + 1 + ic)) * DESC_BINS \
                              + (o0 + io) % DESC_BINS
                        hist += np.bincount(idx, weights=m * wr * wc * wo,
                                            minlength=hist.size)

    # generate the 128-D vector from histograms
    desc = hist.reshape(side, side, DESC_BINS)[1:-1, 1:-1, :].ravel()

    # normalisation
    desc /= max(np.linalg.norm(desc), 1e-7)
    desc = np.minimum(desc, DESC_MAG_CLIP)
    desc /= max(np.linalg.norm(desc), 1e-7)

    return desc.astype(np.float32)

def sift_describe(image: np.ndarray, keypoints: List[Keypoint]) -> Tuple[List[Keypoint], np.ndarray]:
    """
    image: a H x W greyscale image.
    keypoints: a list of standardised keypoints generated from either DoG or HL detectors

    returns: a tuple of a list of keypoints (one for each descriptor), and a np.darray
    """

    gray = to_grayscale(image)
    
    # one gradient cache shared by all keypoints, so each scale is blurred once
    gradients = GradientCache(gray)
    
    out_kps, descs = [], []
    for kp in keypoints:
        oriented = [kp] if kp.angle is not None else assign_orientations(kp, gradients)
        for okp in oriented:
            d = generate_descriptor(okp, gradients)
            if d is not None:
                out_kps.append(okp)
                descs.append(d)

    descs = np.vstack(descs) if descs else np.empty((0, 128), np.float32)

    return out_kps, descs




    
